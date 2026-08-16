"""Tests for the infinite compaction loop fix (issue #40803).

When summary_target_ratio is large enough that the entire transcript fits
within soft_ceiling, the backward walk in _find_tail_cut_by_tokens never
breaks early.  Without the fix this produces either a no-op compression
(compress_start >= compress_end) or a single-message compression whose
summary-of-one overhead saves 0 tokens — both of which cause the
compressor to fire on every subsequent turn with no progress.

The fix adds two safeguards:
1. _find_tail_cut_by_tokens: when the whole transcript fits in soft_ceiling,
   re-walk with the raw (non-inflated) budget to find a meaningful cut.
2. compress(): when compress_start >= compress_end, defer retries via the
   structural no-op backoff (#93022) so should_compress() anti-thrashing
   fires without burning anti-thrash strikes on transcript-shape facts.
"""

from unittest.mock import patch, MagicMock

import time

from agent.context_compressor import ContextCompressor, _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_compressor(**kwargs) -> ContextCompressor:
    defaults = dict(
        model="test-model",
        threshold_percent=0.65,
        protect_first_n=2,
        protect_last_n=3,
        quiet_mode=True,
    )
    defaults.update(kwargs)
    # NOTE: 96K < 512K, so the small-context floor raises the effective
    # threshold_percent to 0.75 → threshold_tokens = 72_000. Tests use
    # 73_000 as the "over threshold" probe value.
    with patch("agent.context_compressor.get_model_context_length", return_value=96000):
        return ContextCompressor(**defaults)


def _build_session(n_turns: int, words_per_turn: int = 20) -> list:
    """Build a multi-turn conversation with a system prompt."""
    base_text = " ".join(["a"] * words_per_turn)
    messages = [{"role": "system", "content": "You are a helpful agent."}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"{base_text} (user turn {i})"})
        messages.append({"role": "assistant", "content": f"{base_text} (assistant turn {i})"})
    return messages


# ---------------------------------------------------------------------------
# Test: compress_start >= compress_end registers as ineffective
# ---------------------------------------------------------------------------

class TestCompressNoOpRegistersIneffective:
    """When compress_start >= compress_end, the fix defers further attempts
    via the structural no-op backoff (#93022) so the anti-thrashing guard
    fires.

    We trigger this path by having _find_tail_cut_by_tokens return
    head_end (which makes compress_end = head_end + 1, same as
    compress_start after alignment)."""

    def test_no_op_arms_structural_backoff(self):
        """compress_start >= compress_end -> backoff armed, strikes untouched."""
        comp = _make_compressor(
            summary_target_ratio=0.45,
            config_context_length=96000,
        )
        # A large session that passes the min_for_compress check
        messages = _build_session(10, words_per_turn=10)
        comp.last_prompt_tokens = 73_000

        # Mock _find_tail_cut_by_tokens to return head_end,
        # causing compress_start >= compress_end
        original = comp._find_tail_cut_by_tokens
        comp._find_tail_cut_by_tokens = lambda msgs, he: he  # force no-op

        result = comp.compress(messages, current_tokens=73_000)

        assert len(result) == len(messages), (
            "no-op compression must return the transcript unchanged"
        )
        assert comp._ineffective_compression_count == 0, (
            "a structural impossibility is not an ineffective strike (#93022)"
        )
        assert comp._structural_no_op_backoff_until > time.monotonic(), (
            "structural no-op must arm the retry backoff"
        )


    def test_two_no_ops_block_should_compress(self):
        """After 2 no-op compressions, should_compress returns False."""
        comp = _make_compressor(
            summary_target_ratio=0.45,
            config_context_length=96000,
        )
        messages = _build_session(10, words_per_turn=10)
        comp.last_prompt_tokens = 73_000
        comp._find_tail_cut_by_tokens = lambda msgs, he: he  # force no-op

        comp.compress(messages, current_tokens=73_000)
        comp.compress(messages, current_tokens=73_000)

        assert comp._ineffective_compression_count == 0, (
            "structural no-ops defer via backoff instead of striking (#93022)"
        )
        assert not comp.should_compress(73_000), (
            "should_compress should return False while the structural backoff holds"
        )



# ---------------------------------------------------------------------------
# Test: _find_tail_cut_by_tokens raw-budget fallback
# ---------------------------------------------------------------------------

class TestTailCutRawBudgetFallback:
    """When the entire transcript fits within soft_ceiling, the fix
    re-walks with the raw budget to find a meaningful cut point."""

    def test_meaningful_cut_with_large_ratio(self):
        """With summary_target_ratio=0.45, _find_tail_cut_by_tokens still
        leaves a meaningful compressable region."""
        comp = _make_compressor(
            summary_target_ratio=0.45,
            config_context_length=96000,
        )
        messages = _build_session(20, words_per_turn=20)
        head_end = comp._protect_head_size(messages)
        head_end = comp._align_boundary_forward(messages, head_end)

        cut = comp._find_tail_cut_by_tokens(messages, head_end)

        n = len(messages)
        middle_size = cut - head_end
        assert middle_size >= 3, (
            f"Expected at least 3 messages in compressable region, got {middle_size} "
            f"(cut={cut}, head_end={head_end}, n={n})"
        )




# ---------------------------------------------------------------------------
# Test: Effective compression resets counter
# ---------------------------------------------------------------------------

class TestEffectiveCompressionResetsCounter:
    """When compression actually saves tokens, the ineffective counter resets."""

    def test_effective_compression_resets_counter(self):
        """After an effective compression, _ineffective_compression_count = 0."""
        comp = _make_compressor(
            summary_target_ratio=0.20,
            config_context_length=96000,
        )
        messages = _build_session(30, words_per_turn=100)
        comp._generate_summary = MagicMock(return_value="Compacted summary of earlier turns.")
        comp.last_prompt_tokens = 73_000

        comp.compress(messages, current_tokens=73_000)

        assert comp._ineffective_compression_count == 0, (
            f"Expected 0 ineffective compressions with effective compression, "
            f"got {comp._ineffective_compression_count}"
        )


# ---------------------------------------------------------------------------
# Test: anti-thrashing in should_compress
# ---------------------------------------------------------------------------

class TestAntiThrashing:
    """Directly test the should_compress anti-thrashing guard."""

    def test_ineffective_count_2_blocks(self):
        """_ineffective_compression_count >= 2 -> should_compress returns False."""
        comp = _make_compressor(config_context_length=96000)
        comp.last_prompt_tokens = 73_000
        comp._ineffective_compression_count = 2
        assert not comp.should_compress(73_000)




# ---------------------------------------------------------------------------
# Test: summary-LLM cooldown guard in should_compress (#11529)
# ---------------------------------------------------------------------------

class TestCooldownGuard:
    """should_compress() must skip compression while the summary LLM is in
    cooldown, otherwise a 429/transient failure re-fires _compress_context()
    every turn (inserting a fallback marker repeatedly) and freezes the CLI.
    """

    def test_active_cooldown_blocks(self):
        """A future cooldown deadline -> should_compress returns False even
        when tokens are over threshold."""
        comp = _make_compressor(config_context_length=96000)
        comp.last_prompt_tokens = 73_000
        comp._summary_failure_cooldown_until = time.monotonic() + 60
        assert not comp.should_compress(73_000)




# ---------------------------------------------------------------------------
# Test: #48621 — gpt-5.3-codex-spark short-session boundary
#
# Issue #48621 Bug 2 claims that a short high-token session (15-20 messages,
# ~90k tokens on a 128k model with protect_last_n=20) hits
# compress_start >= compress_end, causing a silent context wipe.  The
# raw-budget fallback added in the #40803 fix already mitigates this: the
# boundary logic always exposes a minimal compressible window.  This test
# locks that behavior in for the exact #48621 parameters.
# ---------------------------------------------------------------------------

class TestCodexSparkShortSessionBoundary:
    """Verify that gpt-5.3-codex-spark's short-session scenario always yields
    a non-empty compressible window (no silent wipe)."""

    def test_short_high_token_session_has_compressible_window(self):
        """16 messages with large tool outputs on a 128k model must leave
        a compressible middle (compress_start < compress_end)."""
        comp = _make_compressor(
            model="gpt-5.3-codex-spark",
            threshold_percent=0.70,
            protect_first_n=3,
            protect_last_n=20,
            config_context_length=128000,
        )
        # Build system + 3 head pairs + 3 tool groups (large outputs) + tail pair
        big_tool = "x" * 20000  # ~5k tokens each
        messages = [{"role": "system", "content": "You are a helpful agent."}]
        for i in range(3):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}"})
        for i in range(3):
            messages.append({"role": "user", "content": f"Run command {i}"})
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"tc{i}", "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }],
            })
            messages.append({"role": "tool", "tool_call_id": f"tc{i}", "content": big_tool})
        messages.append({"role": "user", "content": "Final question"})
        messages.append({"role": "assistant", "content": "Final answer"})

        head = comp._protect_head_size(messages)
        compress_start = comp._align_boundary_forward(messages, head)
        compress_end = comp._find_tail_cut_by_tokens(messages, compress_start)

        assert compress_start < compress_end, (
            f"No compressible window: start={compress_start}, end={compress_end}. "
            f"This would cause the silent context wipe described in #48621."
        )
        assert comp.has_content_to_compress(messages) is True


class TestPressureRealFloor:
    """Regression: Cyrillic-heavy sessions under-count in the rough estimate,
    letting real prompts ride the provider window (64,842→64,995 observed)
    while the pre-API gate saw sub-threshold pressure."""

    def _compressor(self, last_real, awaiting=False):
        class _C:
            last_real_prompt_tokens = last_real
            awaiting_real_usage_after_compression = awaiting
        return _C()

    def test_real_floor_lifts_undercounted_rough(self):
        from agent.conversation_loop import _pressure_with_real_floor
        assert _pressure_with_real_floor(self._compressor(64_842), 45_000) == 64_842

    def test_rough_wins_when_larger(self):
        from agent.conversation_loop import _pressure_with_real_floor
        assert _pressure_with_real_floor(self._compressor(30_000), 45_000) == 45_000

    def test_stale_real_ignored_right_after_compaction(self):
        from agent.conversation_loop import _pressure_with_real_floor
        compressor = self._compressor(64_842, awaiting=True)
        assert _pressure_with_real_floor(compressor, 20_000) == 20_000

    def test_zero_and_missing_real_are_safe(self):
        from agent.conversation_loop import _pressure_with_real_floor
        assert _pressure_with_real_floor(self._compressor(0), 10_000) == 10_000
        assert _pressure_with_real_floor(object(), 10_000) == 10_000
