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
        # The marker is assistant-role: an exchange is a full agent turn
        # bounded by user messages, so user → marker(assistant) → user keeps
        # strict alternation. A user-role marker produced user → user → user,
        # and the pre-request repair_message_sequence pass merged the marker
        # into the neighbouring real user turn — metadata gone, cursor lost.
        assert markers[0]["role"] == "assistant"

    def test_disabled_is_a_no_op(self):
        cc = _compressor()
        cc._micro_compact_enabled = False
        messages = _conversation()

        assert cc._micro_compact(list(messages)) == messages

    def test_is_off_unless_explicitly_enabled(self):
        # A pass rewrites already-sent history, breaking the prompt-cache
        # prefix, so nobody inherits it from an update: it stays off until
        # `compression.micro_compact` opts in.
        cc = ContextCompressor(
            model="test-model",
            threshold_percent=0.75,
            protect_first_n=1,
            protect_last_n=2,
            quiet_mode=True,
            config_context_length=40960,
            provider="test",
        )
        cc._micro_summarize_one = lambda _text: "ROLLING SUMMARY"
        messages = _conversation()

        assert cc._micro_compact_enabled is False
        assert cc._micro_compact(list(messages)) == messages

    def test_cadence_of_one_runs_every_turn(self):
        cc = _compressor()
        cc._micro_compact_every_n_turns = 1
        messages = _conversation(exchanges=8)

        first = cc._micro_compact(list(messages))
        second = cc._micro_compact(list(first))

        assert cc._micro_compact_cursor > 0
        assert len(_summary_markers(first)) == 1
        # Each turn absorbed something, so the transcript kept shrinking.
        assert len(second) < len(first)

    def test_cadence_skips_turns_until_a_pass_is_due(self):
        cc = _compressor()
        cc._micro_compact_every_n_turns = 3
        messages = _conversation(exchanges=8)

        first = cc._micro_compact(list(messages))
        second = cc._micro_compact(list(first))

        # Cache prefix untouched on the turns in between.
        assert _summary_markers(first) == []
        assert _summary_markers(second) == []
        assert cc._micro_compact_cursor == 0

        third = cc._micro_compact(list(second))

        assert len(_summary_markers(third)) == 1
        assert not any("answer 0" in str(m.get("content")) for m in third)
        # Counter rearmed for the next window.
        assert cc._micro_compact_turns_since_pass == 0

    def test_cadence_is_clamped_to_at_least_one(self):
        # A bogus 0 or negative must not disable compaction silently, nor
        # divide-by-zero: it degrades to "every turn".
        for bogus in (0, -5):
            cc = _compressor()
            cc._micro_compact_every_n_turns = bogus
            messages = _conversation(exchanges=8)

            result = cc._micro_compact(list(messages))

            assert len(_summary_markers(result)) == 1

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
        """Every byte the user typed stays in the transcript — by design.

        Assistant output is largely an account of what was done and survives
        summarising; the user's own words are the intent everything else is
        derived from and can't be reconstructed from it. So an exchange starts
        at the assistant message and the walk skips past user turns.

        The invariant is on user TEXT, not message-list shape: superseding an
        old marker leaves two real user turns adjacent, and they are merged
        (\\n\\n-joined, same as repair_message_sequence pass 2) to keep strict
        alternation. Text is never summarized or dropped.
        """
        cc = _compressor()
        messages = _conversation(exchanges=10)
        originals = [m["content"] for m in messages if m["role"] == "user"]

        for _ in range(5):
            messages = cc._micro_compact(messages)

        surviving_text = "\n\n".join(
            m["content"] for m in messages
            if m.get("role") == "user" and not m.get(COMPRESSED_SUMMARY_METADATA_KEY)
        )
        for original in originals:
            assert original in surviving_text, (
                f"user text {original!r} must survive verbatim"
            )

    def test_timeline_marker_absorbing_user_keeps_prompt_displayable(self):
        marker = (
            "[System: The active model for this chat has changed to k3. "
            "From this point forward, use this runtime metadata.]"
        )
        messages = [
            {
                "role": "user",
                "content": marker,
                "display_kind": "model_switch",
                "display_metadata": {"provider": "openrouter"},
            },
            {
                "role": "user",
                "content": "the newest real prompt",
                "api_content": "the newest real prompt\n\nplugin context",
            },
        ]

        [merged] = ContextCompressor._merge_adjacent_user_turns(messages)

        assert merged["content"] == "the newest real prompt"
        assert merged["api_content"] == f"{marker}\n\nthe newest real prompt\n\nplugin context"
        assert "display_kind" not in merged
        assert "display_metadata" not in merged

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

    def test_resume_does_not_destroy_the_accumulated_summary(self):
        """A resumed session must not throw away compacted history.

        The rolling summary lives in memory; a resumed process starts with an
        empty one while the marker holding every previous exchange is still in
        the transcript. Superseding on that first pass would replace the whole
        history with a summary of one exchange.
        """
        msgs = _conversation(exchanges=10)
        first = _compressor(summary="IMPORTANT HISTORY: decisions and paths")
        for _ in range(3):
            msgs = first._micro_compact(msgs)
        assert any("IMPORTANT HISTORY" in m["content"] for m in _summary_markers(msgs))

        # Fresh compressor over the same transcript = resume.
        resumed = _compressor(summary="MERGED: history plus newest exchange")
        assert resumed._micro_compact_rolling_summary == ""
        result = resumed._micro_compact(msgs)

        markers = _summary_markers(result)
        assert len(markers) == 1
        assert "MERGED" in markers[0]["content"]

    def test_resume_keeps_the_old_marker_when_rehydration_fails(self):
        """If the prior summary can't be recovered, it must not be dropped."""
        msgs = _conversation(exchanges=10)
        first = _compressor(summary="IMPORTANT HISTORY: decisions and paths")
        for _ in range(3):
            msgs = first._micro_compact(msgs)

        resumed = _compressor(summary="BRAND NEW SUMMARY")
        resumed._rolling_summary_from_marker = staticmethod(lambda _c: "")
        result = resumed._micro_compact(msgs)

        markers = _summary_markers(result)
        assert len(markers) == 2, "must retain the un-carried history"
        assert any("IMPORTANT HISTORY" in m["content"] for m in markers)

    def test_rolling_summary_round_trips_through_a_marker(self):
        cc = _compressor()
        cc._micro_compact_rolling_summary = "decisions: use the existing helper"
        msgs = _conversation(exchanges=6)
        result = cc._micro_compact(msgs)
        marker = _summary_markers(result)[0]

        assert (cc._rolling_summary_from_marker(marker["content"])
                == cc._micro_compact_rolling_summary)

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

        Inserting the summary marker costs a fixed block of scaffolding
        (``SUMMARY_PREFIX``, the historical heading and the end marker —
        currently ~450 tokens and grows when the preamble is lengthened).
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
        """Session-total savings go positive once marker overhead is paid back.

        The first pass inserts ``SUMMARY_PREFIX`` scaffolding (~450 tokens);
        with the current preamble that alone leaves the cumulative counter
        negative after only a few absorptions. Enough later passes must
        still recover it — that is the amortization contract.
        """
        cc = _compressor()
        messages = _conversation(exchanges=10)

        for _ in range(6):
            messages = cc._micro_compact(messages)

        assert cc._micro_compact_passes == 6
        assert cc._micro_compact_tokens_saved_total > 0

    def test_defrag_triggers_once_the_rolling_summary_grows(self):
        """Defrag rewrites the summary text and the marker — nothing else.

        The original implementation spliced the whole remaining middle (user
        turns included) into the marker, silently absorbing user messages.
        Defrag is now transcript-shape-neutral: same message list, same
        cursor, marker content rewritten in place.
        """
        cc = _compressor(summary="FRESH DEFRAGGED SUMMARY")
        messages = _conversation(exchanges=8)
        # Seed a real marker + oversized rolling summary, as after many passes.
        messages = cc._micro_compact(list(messages))
        cc._micro_compact_rolling_summary = "x" * 40_000  # far over the threshold
        cursor_before = cc._micro_compact_cursor
        shape_before = [m.get("role") for m in messages]

        assert cc._needs_defrag() is True
        result = cc._micro_compact(list(messages))

        assert cc._micro_compact_rolling_summary == "FRESH DEFRAGGED SUMMARY"
        markers = _summary_markers(result)
        assert len(markers) == 1
        assert "FRESH DEFRAGGED SUMMARY" in markers[0]["content"]
        # Shape-neutral: no messages absorbed or spliced, cursor unmoved.
        assert [m.get("role") for m in result] == shape_before
        assert cc._micro_compact_cursor == cursor_before

    def test_defrag_never_absorbs_user_messages(self):
        """Defrag must not touch user turns — the feature's core invariant.

        The original implementation serialized head..tail (user turns
        included) and spliced it away: 8 of 10 user prompts were destroyed in
        one pass. Defrag now only rewrites the rolling summary text.
        """
        cc = _compressor(summary="DEFRAGGED")
        messages = [{"role": "system", "content": "sys"}]
        for i in range(10):
            messages.append({"role": "user", "content": f"UNIQUE-USER-PROMPT-{i}"})
            messages.append({"role": "assistant", "content": f"answer {i} " + "z" * 400})

        cc._micro_compact_rolling_summary = "x" * 40_000  # force defrag
        result = cc._micro_compact(list(messages))

        surviving = [
            m["content"] for m in result
            if m.get("role") == "user" and not m.get(COMPRESSED_SUMMARY_METADATA_KEY)
        ]
        for i in range(10):
            assert any(f"UNIQUE-USER-PROMPT-{i}" in s for s in surviving), (
                f"user prompt {i} was absorbed by defrag"
            )

    def test_defrag_summarizes_only_the_summary_text(self):
        """The defrag aux call receives the rolling summary, not the transcript."""
        cc = _compressor()
        captured = {}

        def capture(text):
            captured["text"] = text
            return "DEFRAGGED"

        cc._micro_summarize_one = capture
        cc._micro_compact_rolling_summary = "OLD-SUMMARY " + "x" * 40_000
        messages = _conversation(exchanges=8)
        cc._micro_compact(list(messages))

        assert "OLD-SUMMARY" in captured["text"]
        assert "[USER]" not in captured["text"], (
            "defrag must never serialize transcript user turns"
        )

    def test_spliced_transcript_survives_repair_message_sequence(self):
        """The compacted transcript must survive the production repair pass.

        conversation_loop runs repair_message_sequence before EVERY API call.
        With the old user-role marker, splicing next to a real user turn made
        user → user → user; repair merged the marker into the real user
        message — metadata gone, cursor unrecoverable, summary text duplicated
        into the transcript on every later pass. Pin the integration: markers
        survive repair untouched, and no summary text leaks into real user
        messages.
        """
        from agent.agent_runtime_helpers import repair_message_sequence

        class _DummyAgent:
            session_id = "probe"
            _last_flushed_db_idx = 0

        cc = _compressor()
        messages = _conversation(exchanges=8)

        for _ in range(3):
            messages = cc._micro_compact(messages)
            repairs = repair_message_sequence(_DummyAgent(), messages)
            assert repairs == 0, (
                "micro-compacted transcript must already be alternation-valid"
            )
            markers = _summary_markers(messages)
            assert len(markers) == 1, "marker destroyed by repair pass"
            polluted = [
                m for m in messages
                if m.get("role") == "user"
                and not m.get(COMPRESSED_SUMMARY_METADATA_KEY)
                and "ROLLING SUMMARY" in str(m.get("content"))
            ]
            assert not polluted, "summary text leaked into a real user message"

    def test_spliced_transcript_has_no_consecutive_same_role_messages(self):
        """Alternation invariant, checked directly on tool-bearing turns."""
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
                    for j in range(2)
                ],
            })
            for j in range(2):
                msgs.append({"role": "tool", "tool_call_id": f"c{i}-{j}",
                             "content": "T" * 400})
            # Multi-iteration turn: a second assistant+tools group before the
            # next user message — the splice must absorb the WHOLE turn.
            msgs.append({"role": "assistant", "content": f"followup {i} " + "y" * 200})

        for _ in range(4):
            msgs = cc._micro_compact(msgs)
            for a, b in zip(msgs, msgs[1:]):
                ra, rb = a.get("role"), b.get("role")
                assert not (ra == rb and ra in ("user", "assistant")), (
                    f"consecutive {ra} messages after micro-compaction"
                )

    def test_marker_reports_no_user_provenance(self):
        """Micro markers absorb only assistant/tool content (#64650)."""
        from agent.context_compressor import COMPRESSED_SUMMARY_HAS_USER_TURN_KEY

        cc = _compressor()
        result = cc._micro_compact(_conversation(exchanges=6))
        marker = _summary_markers(result)[0]

        assert marker[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] is False

    def test_supersede_never_drops_a_batch_compaction_marker(self):
        """A batch marker holds history the rolling summary does NOT contain.

        Sequence: micro absorbs some exchanges (rolling summary = those k
        exchanges only), then batch compaction fires and its marker
        summarizes MORE (exchanges 1..m). The next micro pass's supersede
        must not treat the batch marker as redundant — dropping it destroys
        everything batch summarized beyond exchange k. Only micro-tagged
        markers (whose text is provably inside the rolling summary) may be
        superseded.
        """
        cc = _compressor(summary="MICRO SUMMARY (exchanges 1..k only)")
        msgs = _conversation(exchanges=8)
        msgs = cc._micro_compact(msgs)
        assert cc._micro_compact_rolling_summary

        # Simulate a batch-compaction marker replacing the middle (batch
        # markers carry the shared metadata key but NOT the micro tag).
        batch_marker = {
            "role": "user",
            "content": "[batch summary] CRITICAL HISTORY: exchanges 1..m",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        }
        micro_idx = next(
            i for i, m in enumerate(msgs)
            if m.get(COMPRESSED_SUMMARY_METADATA_KEY)
        )
        msgs = msgs[:micro_idx] + [batch_marker] + msgs[micro_idx + 3:]

        out = cc._micro_compact(msgs)

        assert any(
            "CRITICAL HISTORY" in str(m.get("content")) for m in out
        ), "batch-compaction summary destroyed by micro supersede"

    def test_defrag_never_rewrites_a_batch_compaction_marker(self):
        """Defrag rewrites only micro-tagged markers, never batch markers."""
        cc = _compressor(summary="DEFRAGGED")
        msgs = [{"role": "system", "content": "sys"}]
        msgs.append({
            "role": "user",
            "content": "[batch summary] CRITICAL HISTORY: exchanges 1..m",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        })
        for i in range(6):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i} " + "z" * 400})

        cc._micro_compact_rolling_summary = "x" * 40_000  # force defrag
        result = cc._micro_compact(list(msgs))

        batch = [m for m in result if "CRITICAL HISTORY" in str(m.get("content"))]
        assert batch, "batch marker content overwritten by defrag"

    def test_batch_compress_resets_micro_state(self):
        """compress() success path invalidates the stale rolling summary.

        Without the reset, the in-memory micro summary (exchanges 1..k)
        outlives a batch compaction whose marker covers 1..m — the next
        micro pass would then treat its stale summary as cumulative.
        """
        cc = _compressor()
        msgs = _conversation(exchanges=8)
        msgs = cc._micro_compact(msgs)
        assert cc._micro_compact_rolling_summary
        assert cc._micro_compact_cursor > 0

        cc.compress(msgs, force=True)

        assert cc._micro_compact_rolling_summary == ""
        assert cc._micro_compact_cursor == 0

    def test_persist_disabled_agent_never_micro_compacts(self):
        """finalize_turn must skip micro-compaction on isolated fork agents.

        The background-review fork sets _persist_disabled=True; running a
        pass there burns an aux-LLM call on a throwaway replay transcript
        and, if the compressor ever holds a DB binding, would
        archive_and_compact the CANONICAL session rows.
        """
        import inspect

        from agent import turn_finalizer

        src = inspect.getsource(turn_finalizer.finalize_turn)
        micro_block = src.split("Post-turn micro-compaction", 1)[1]
        # Scope to the micro block only: stop at the persist call that follows.
        micro_block = micro_block.split("agent._persist_session", 1)[0]
        assert "_persist_disabled" in micro_block, (
            "micro-compaction gate must check agent._persist_disabled"
        )

    def test_splice_preserves_db_persisted_stamps(self):
        """Surviving messages keep their _db_persisted stamps through a splice.

        Micro-compaction archives in place under the SAME session id, so the
        stamps on untouched messages stay accurate. Stripping them (as the
        batch path does for its child-session rotation) meant an
        archive_and_compact failure left every previously-persisted message
        unstamped and the next append-only flush re-inserted them all as
        duplicate active rows.
        """
        from agent.context_compressor import _DB_PERSISTED_MARKER

        cc = _compressor()
        messages = _conversation(exchanges=8)
        for m in messages:
            m[_DB_PERSISTED_MARKER] = True

        # No DB bound -> _sync_micro_compact_to_db no-ops (the failure shape).
        result = cc._micro_compact(messages)

        unstamped = [
            m for m in result
            if not m.get(_DB_PERSISTED_MARKER)
            and not m.get(COMPRESSED_SUMMARY_METADATA_KEY)
        ]
        assert not unstamped, (
            "splice must not strip _db_persisted from surviving messages"
        )


class TestDefragFlushCursorInvalidation:
    """Sibling of the finalize_turn pop site (#75170): defrag pops
    _DB_PERSISTED_MARKER from the live marker dict in place, so the bounded
    flush-scan cursor must be invalidated or the rewritten summary is
    identity-skipped and never re-persisted."""

    def _defrag_setup(self):
        from agent.context_compressor import _DB_PERSISTED_MARKER

        cc = _compressor(summary="FRESH DEFRAGGED SUMMARY")
        messages = _conversation(exchanges=8)
        messages = cc._micro_compact(list(messages))
        # Simulate an incremental flush having stamped the marker row.
        for m in messages:
            if m.get(COMPRESSED_SUMMARY_METADATA_KEY):
                m[_DB_PERSISTED_MARKER] = True
        cc._micro_compact_rolling_summary = "x" * 40_000  # force defrag
        return cc, messages

    def test_defrag_marker_pop_raises_invalidation_flag(self):
        from agent.context_compressor import _DB_PERSISTED_MARKER

        cc, messages = self._defrag_setup()
        assert cc._flush_scan_cursor_invalidated is False

        result = cc._micro_compact(list(messages))

        markers = _summary_markers(result)
        assert len(markers) == 1
        # The pop happened in place on the live dict...
        assert not markers[0].get(_DB_PERSISTED_MARKER)
        # ...so the compressor must flag the flush-scan cursor stale.
        assert cc._flush_scan_cursor_invalidated is True

    def test_no_defrag_no_flag(self):
        cc = _compressor()
        messages = _conversation(exchanges=8)
        cc._micro_compact(list(messages))
        assert cc._flush_scan_cursor_invalidated is False

    def test_finalizer_consumes_flag_and_invalidates_agent_cursor(self):
        """finalize_turn's micro-compaction block must translate the
        compressor flag into agent._db_flush_scan_prefix = None (and reset
        the flag) so the next flush re-examines the rewritten marker row."""
        import inspect

        from agent import turn_finalizer

        src = inspect.getsource(turn_finalizer.finalize_turn)
        micro_block = src.split("Post-turn micro-compaction", 1)[1]
        micro_block = micro_block.split("agent._persist_session", 1)[0]
        assert "_flush_scan_cursor_invalidated" in micro_block, (
            "finalize_turn must consume the compressor's cursor-invalidation "
            "flag raised by the defrag marker pop"
        )
        assert "agent._db_flush_scan_prefix = None" in micro_block, (
            "finalize_turn must invalidate the bounded flush-scan cursor "
            "when the defrag pop stripped a live marker's stamp"
        )
