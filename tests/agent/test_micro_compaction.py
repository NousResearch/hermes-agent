"""Tests for per-turn micro-compaction in ``ContextCompressor``.

Micro-compaction amortizes the cost of context compression: instead of one
long pause when the window fills, each turn folds the single oldest
un-absorbed exchange into a rolling summary.

The invariants that matter:

* one call absorbs exactly one exchange (assistant + its tool results), so
  the per-turn cost stays bounded;
* the absorbed span is replaced by a summary marker carrying the usual
  ``_compressed_summary`` metadata, so resume/handoff treat it like a batch
  summary;
* the cursor advances, so successive calls walk forward rather than
  re-summarising the same exchange;
* protected head and tail messages are never touched;
* an exchange the summarizer cannot handle is retried a bounded number of
  times and then skipped, so a poison exchange can't stall every turn.
"""

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    _MICRO_COMPACT_MAX_CONSECUTIVE_FAILURES,
)


def _compressor(summary="ROLLING SUMMARY") -> ContextCompressor:
    cc = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=1,
        protect_last_n=2,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    cc._micro_compact_enabled = True
    # Stand in for the auxiliary summarizer LLM.
    cc._micro_summarize_one = lambda _text: summary
    return cc


def _conversation(exchanges: int = 6) -> list:
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(exchanges):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "z" * 400})
    return msgs


def _summary_markers(messages: list) -> list:
    return [m for m in messages if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]


class TestMicroCompaction:
    def test_absorbs_one_exchange_and_leaves_a_summary_marker(self):
        cc = _compressor()
        messages = _conversation()

        result = cc._micro_compact(list(messages))

        # The absorbed assistant turn is gone from the transcript.
        assert any("answer 0" in str(m.get("content")) for m in messages)
        assert not any("answer 0" in str(m.get("content")) for m in result)
        markers = _summary_markers(result)
        assert len(markers) == 1
        assert "ROLLING SUMMARY" in markers[0]["content"]
        # The marker stands in for a user turn, like batch compaction's does.
        assert markers[0]["role"] == "user"

    def test_disabled_is_a_no_op(self):
        cc = _compressor()
        cc._micro_compact_enabled = False
        messages = _conversation()

        assert cc._micro_compact(list(messages)) == messages

    def test_cursor_advances_across_successive_turns(self):
        cc = _compressor()
        messages = _conversation(exchanges=8)

        first = cc._micro_compact(list(messages))
        cursor_after_first = cc._micro_compact_cursor
        second = cc._micro_compact(list(first))

        assert cursor_after_first > 0
        assert cc._micro_compact_cursor >= cursor_after_first
        # Still exactly one marker: the second pass merges into the rolling
        # summary rather than stacking a second summary block.
        assert len(_summary_markers(second)) == 1

    def test_protected_head_and_tail_survive(self):
        cc = _compressor()
        messages = _conversation()

        result = cc._micro_compact(list(messages))

        assert result[0] == messages[0], "system prompt must be preserved"
        assert result[-1] == messages[-1], "most recent turn must be preserved"

    def test_user_messages_are_never_absorbed(self):
        """User turns stay verbatim for the life of the session — by design.

        Assistant output is largely an account of what was done and survives
        summarising; the user's own words are the intent everything else is
        derived from and can't be reconstructed from it. So an exchange starts
        at the assistant message and the walk skips past user turns.
        """
        cc = _compressor()
        messages = _conversation(exchanges=10)
        originals = [m["content"] for m in messages if m["role"] == "user"]

        for _ in range(5):
            messages = cc._micro_compact(messages)

        surviving = [
            m["content"] for m in messages
            if m.get("role") == "user" and not m.get(COMPRESSED_SUMMARY_METADATA_KEY)
        ]
        assert surviving == originals, "user turns must survive verbatim"

    def test_cursor_is_derived_from_the_spliced_list(self):
        """The cursor must never carry over a pre-splice index.

        A splice collapses an assistant plus its tool results -- often several
        messages -- into one marker, so every later index shifts. Reusing
        ``exchange_end`` left the cursor pointing inside a *later* exchange's
        tool group; the next pass walked forward to the following assistant
        and skipped that exchange entirely, so on tool-bearing conversations
        roughly half the work silently never happened.

        Tool-free fixtures cannot catch this: the span is one message, so
        nothing shifts.
        """
        cc = _compressor()
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(8):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({
                "role": "assistant",
                "content": f"a{i}",
                "tool_calls": [
                    {"id": f"c{i}-{j}", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}}
                    for j in range(3)
                ],
            })
            for j in range(3):
                msgs.append({"role": "tool", "tool_call_id": f"c{i}-{j}",
                             "content": "T" * 500})

        for _ in range(4):
            msgs = cc._micro_compact(msgs)
            marker_idx = next(
                i for i, m in enumerate(msgs)
                if m.get(COMPRESSED_SUMMARY_METADATA_KEY)
            )
            assert cc._micro_compact_cursor == marker_idx + 1, (
                "cursor must sit just past the marker in the spliced list"
            )

    def test_short_conversation_is_untouched(self):
        cc = _compressor()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        assert cc._micro_compact(list(messages)) == messages

    def test_summarizer_failure_leaves_conversation_intact(self):
        cc = _compressor()
        cc._micro_summarize_one = lambda _text: None
        messages = _conversation()

        result = cc._micro_compact(list(messages))

        assert result == messages
        assert cc._micro_compact_consecutive_failures == 1

    def test_poison_exchange_is_skipped_after_repeated_failures(self):
        """A repeatedly unsummarizable exchange must not stall every turn."""
        cc = _compressor()
        cc._micro_summarize_one = lambda _text: None
        messages = _conversation()

        for _ in range(_MICRO_COMPACT_MAX_CONSECUTIVE_FAILURES):
            cc._micro_compact(list(messages))

        # The cursor has moved past the stuck exchange and the strike count
        # is reset, so the next turn attempts new material.
        assert cc._micro_compact_cursor > 0
        assert cc._micro_compact_consecutive_failures == 0

    def test_repeated_compaction_shrinks_context_and_keeps_one_marker(self):
        """The whole point: successive turns must reduce the transcript.

        The rolling summary is cumulative, so an earlier marker's text is a
        subset of the current one. Keeping the earlier markers stacked
        near-duplicate copies (each with its own heading/end-marker
        scaffolding) and made the transcript grow every turn — the opposite
        of what compaction is for.
        """
        from agent.model_metadata import estimate_messages_tokens_rough

        cc = _compressor()
        # Cumulative summary, like the real summarizer produces.
        state = {"n": 0}

        def growing(_text):
            state["n"] += 1
            return "SUMMARY " + " ".join(f"ex{i}" for i in range(state["n"]))

        cc._micro_summarize_one = growing

        messages = _conversation(exchanges=12)
        before = estimate_messages_tokens_rough(messages)
        for _ in range(6):
            messages = cc._micro_compact(messages)
        after = estimate_messages_tokens_rough(messages)

        assert len(_summary_markers(messages)) == 1
        assert after < before, f"context grew: {before} -> {after}"

    def test_emits_content_free_token_telemetry(self, caplog):
        """Each pass logs one JSON line with the token accounting.

        Message counts barely move even when the saving is large, so the token
        fields are what make the effect measurable in a real session.
        """
        import json
        import logging

        cc = _compressor()
        messages = _conversation(exchanges=8)

        with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
            result = cc._micro_compact(messages)

        lines = [
            r.getMessage() for r in caplog.records
            if "micro compaction telemetry:" in r.getMessage()
        ]
        assert len(lines) == 1
        payload = json.loads(lines[0].split("micro compaction telemetry: ", 1)[1])

        assert payload["event"] == "micro_compaction"
        assert payload["outcome"] == "absorbed"
        assert payload["tokens_saved_total"] == -payload["tokens_delta"]
        assert payload["passes_total"] == 1
        assert payload["messages_after"] == len(result)
        assert payload["exchange_tokens"] > 0
        # Content-free: no transcript text may ride along in the payload.
        blob = json.dumps(payload)
        assert "answer 0" not in blob and "question 0" not in blob

    def test_telemetry_reports_occupancy_without_forcing_resolution(self, caplog):
        """Occupancy is the headline: how full the window is being kept.

        It must be read from the cached threshold only. The public
        ``threshold_tokens`` property resolves lazily and can fire a
        synchronous /models probe (#32221); telemetry must never be what
        blocks a turn, so an unresolved window reports null instead.
        """
        import json
        import logging

        cc = _compressor()
        cc.threshold_tokens = 10_000  # pin; also populates the cache
        messages = _conversation(exchanges=8)

        with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
            cc._micro_compact(messages)

        line = next(r.getMessage() for r in caplog.records
                    if "micro compaction telemetry:" in r.getMessage())
        payload = json.loads(line.split("micro compaction telemetry: ", 1)[1])

        assert payload["threshold_tokens"] == 10_000
        assert payload["occupancy_pct"] == pytest.approx(
            payload["tokens_after"] / 10_000 * 100, abs=0.1
        )

    def test_emitter_never_forces_window_resolution(self, caplog):
        """The emitter reads the cached threshold, never the property.

        In a real pass the threshold is already resolved by the time
        telemetry runs (the tail calculation needs it), so occupancy is
        normally populated. This pins the safety property directly: with the
        cache empty, emitting reports null rather than triggering the lazy
        resolution — which can issue a synchronous /models probe (#32221).
        """
        import json
        import logging

        cc = _compressor()
        cc._threshold_tokens = None
        cc._resolved_context_length = None

        def explode(self):  # pragma: no cover - must never be called
            raise AssertionError("telemetry forced context-length resolution")

        with patch.object(type(cc), "threshold_tokens",
                          property(explode, lambda s, v: None)):
            with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
                cc._emit_micro_compaction_telemetry(
                    outcome="absorbed",
                    messages_before=10,
                    messages_after=9,
                    tokens_before=500,
                    tokens_after=400,
                )

        line = next(r.getMessage() for r in caplog.records
                    if "micro compaction telemetry:" in r.getMessage())
        payload = json.loads(line.split("micro compaction telemetry: ", 1)[1])

        assert payload["occupancy_pct"] is None
        assert payload["threshold_tokens"] is None

    def test_first_pass_costs_marker_overhead_then_pays_it_back(self):
        """The first pass can grow the transcript; later passes recover it.

        Inserting the summary marker costs a fixed ~400 tokens of scaffolding
        (the compaction preamble, the historical heading and the end marker).
        On pass one that overhead is paid against a single absorbed exchange,
        so the net can be positive. From pass two on the marker is replaced
        rather than added, so the scaffolding is already paid for and each
        absorbed exchange is pure saving. Anyone reading a single turn's
        telemetry needs to know this before concluding it made things worse.
        """
        from agent.model_metadata import estimate_messages_tokens_rough

        cc = _compressor()
        messages = _conversation(exchanges=10)
        start = estimate_messages_tokens_rough(messages)

        messages = cc._micro_compact(messages)
        after_first = estimate_messages_tokens_rough(messages)

        for _ in range(5):
            messages = cc._micro_compact(messages)
        after_many = estimate_messages_tokens_rough(messages)

        assert after_first > start, "expected one-time marker overhead"
        assert after_many < after_first, "later passes must recover it"

    def test_cumulative_savings_accumulate_across_passes(self):
        cc = _compressor()
        messages = _conversation(exchanges=10)

        for _ in range(4):
            messages = cc._micro_compact(messages)

        assert cc._micro_compact_passes == 4
        assert cc._micro_compact_tokens_saved_total > 0

    def test_defrag_triggers_once_the_rolling_summary_grows(self):
        cc = _compressor(summary="FRESH DEFRAGGED SUMMARY")
        cc._micro_compact_rolling_summary = "x" * 40_000  # far over the threshold
        messages = _conversation(exchanges=8)

        assert cc._needs_defrag() is True
        result = cc._micro_compact(list(messages))

        assert cc._micro_compact_rolling_summary == "FRESH DEFRAGGED SUMMARY"
        markers = _summary_markers(result)
        assert len(markers) == 1
        assert "FRESH DEFRAGGED SUMMARY" in markers[0]["content"]
