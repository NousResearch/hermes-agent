"""Tests for the CompactionResult metrics dataclass."""

from agent.compaction_result import CompactionResult


def test_compaction_result_records_basic_fields():
    r = CompactionResult(
        original_messages=50,
        compacted_messages=12,
        original_tokens=80_000,
        compacted_tokens=8_000,
        operations_deduped=4,
        triggered_by="token",
    )
    assert r.original_messages == 50
    assert r.compacted_messages == 12
    assert r.token_reduction_pct == 90.0
    assert r.message_reduction_pct == 76.0


def test_compaction_result_handles_zero_original():
    r = CompactionResult(
        original_messages=0,
        compacted_messages=0,
        original_tokens=0,
        compacted_tokens=0,
        operations_deduped=0,
        triggered_by="token",
    )
    assert r.token_reduction_pct == 0.0
    assert r.message_reduction_pct == 0.0


def test_compaction_result_summary_line():
    r = CompactionResult(
        original_messages=50,
        compacted_messages=12,
        original_tokens=80_000,
        compacted_tokens=8_000,
        operations_deduped=4,
        triggered_by="token",
    )
    line = r.summary_line()
    assert "80,000 → 8,000" in line
    assert "90%" in line
    assert "deduped 4" in line
    assert "trigger=token" in line


class TestCompressorPopulatesResult:
    """compress() must populate ``self.last_compaction_result`` after each run."""

    def _compressor(self):
        from agent.context_compressor import ContextCompressor
        c = ContextCompressor.__new__(ContextCompressor)
        c.protect_first_n = 1
        c.protect_last_n = 2
        c.tail_token_budget = 300       # ← tight budget forces real middle region
        c.context_length = 200_000
        c.threshold_percent = 0.50
        c.threshold_tokens = 100_000
        c.threshold_absolute_max = None
        c.summary_target_ratio = 0.20
        c.max_summary_tokens = 8000
        c.dedup_operations = False
        c.anchor_first_assistant = False
        c.message_threshold = None
        c.turn_threshold = None
        c.quiet_mode = True
        c.compression_count = 0
        c.last_prompt_tokens = 50_000
        c._previous_summary = None
        c._summary_failure_cooldown_until = 0.0
        c._last_compression_savings_pct = 100.0
        c._ineffective_compression_count = 0
        c._last_summary_dropped_count = 0
        c._last_summary_fallback_used = False
        c._last_summary_error = None
        c._last_aux_model_failure_error = None
        c._last_aux_model_failure_model = None
        c._last_trigger = "token"
        c._last_op_deduped = 0
        c.summary_model = None
        c.model = "test"
        c.provider = "test"
        c.base_url = ""
        c.api_key = ""
        c.api_mode = "chat_completions"
        c.last_compaction_result = None
        return c

    def test_compress_populates_last_result(self, monkeypatch):
        from agent.context_compressor import ContextCompressor
        from agent.compaction_result import CompactionResult

        c = self._compressor()
        # Stub the LLM call to return a deterministic summary
        monkeypatch.setattr(
            ContextCompressor, "_generate_summary",
            lambda self, turns, focus_topic=None: "## Goal\nTest summary.",
        )
        # 12 messages, mixed roles. With protect_first_n=1, protect_last_n=2,
        # tail_token_budget=300, the middle region spans roughly idx 1..9
        # — non-empty so compress() actually runs.
        msgs = (
            [{"role": "system", "content": "sys"}]
            + [
                m for i in range(5)
                for m in (
                    {"role": "user", "content": f"u{i}"},
                    {"role": "assistant", "content": f"reply {i}"},
                )
            ]
            + [{"role": "user", "content": "final"}]
        )
        assert len(msgs) == 12

        out = c.compress(msgs, current_tokens=120_000)

        assert isinstance(c.last_compaction_result, CompactionResult)
        assert c.last_compaction_result.original_messages == len(msgs)
        assert c.last_compaction_result.compacted_messages == len(out)
        assert c.last_compaction_result.triggered_by == "token"
        # Middle was actually compressed (out should be shorter than in)
        assert len(out) < len(msgs)
