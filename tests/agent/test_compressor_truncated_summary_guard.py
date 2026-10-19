"""Truncated compaction summaries must never become checkpoints.

Port of earendil-works/pi#7048 (commit 97fa14e39): a summarization response
whose ``finish_reason == "length"`` contains PARTIAL text — the generation
stopped on the output-token cap mid-summary. Persisting it as the compaction
checkpoint silently truncates the conversation's memory and feeds the cut-off
text back into every subsequent iterative-update prompt.

Covers all three compressor summarization sites:
  1. ``_generate_summary`` (main batch summary) — length stop raises, falls
     back to main model once, then ABORTS compression preserving messages.
  2. ``_micro_summarize_one`` (micro-compact rolling summary) — length stop
     discards the partial merge (returns None) so the exchange stays
     unabsorbed.
(The former third site, ``_build_chunk_digests``, was removed on main by
#96603 — lean digests now ride the single ``_generate_summary`` request, so
its guard is covered by site 1.)
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _is_summary_refusal,
    _response_finish_reason,
)


def _mock_response(content="a perfectly fine summary", finish_reason="stop"):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    resp.choices = [choice]
    return resp


def _msgs(n=12):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i} " + "x" * 50}
        for i in range(n)
    ]


class TestResponseFinishReason:
    def test_object_shaped(self):
        assert _response_finish_reason(_mock_response(finish_reason="length")) == "length"
        assert _response_finish_reason(_mock_response(finish_reason="stop")) == "stop"

    def test_dict_shaped(self):
        resp = {"choices": [{"message": {"content": "x"}, "finish_reason": "LENGTH"}]}
        assert _response_finish_reason(resp) == "length"

    def test_missing_field_is_empty(self):
        assert _response_finish_reason({"choices": [{"message": {"content": "x"}}]}) == ""
        assert _response_finish_reason({"choices": []}) == ""
        assert _response_finish_reason(None) == ""


class TestSummaryRefusalGuard:
    @pytest.mark.parametrize(
        "content",
        [
            "I can't produce this summary as requested. The instructions conflict with my operating rules.",
            "Sorry, I am unable to create a context checkpoint for this conversation.",
            "I'm sorry, but I can't provide the requested summary.",
            "I’m sorry, but I can’t provide the requested summary.",
            "I apologize, but I cannot create a summary of this conversation.",
            "As an AI, I cannot provide the requested context checkpoint.",
            "I can not provide a summary of this conversation.",
            "I can't summarise this conversation for you.",
        ],
    )
    def test_rejects_refusal_body(self, content):
        assert _is_summary_refusal(content) is True

    def test_accepts_structured_summary_that_records_a_refusal(self):
        summary = "## Goal\nPreserve the user's task.\n\n## Completed Actions\n1. Recorded that a provider refused an earlier request."
        assert _is_summary_refusal(summary) is False

    def test_narrowing_and_heading_exemption(self):
        # Narrow by design: a refusal that never mentions the summary/checkpoint is not ours to judge.
        assert _is_summary_refusal("I can't help with that request.") is False
        # A hedging preamble in front of a real templated summary is not a refusal.
        preamble = (
            "I cannot see the earlier turns, but here is the summary:\n"
            "## Goal\nFinish the task.\n\n## Completed Actions\n1. Did X."
        )
        assert _is_summary_refusal(preamble) is False


class TestGenerateSummaryTruncationGuard:
    def test_length_stop_is_rejected_and_aborts(self):
        """A length-stopped summary must not become a checkpoint; with no
        distinct aux model to fall back from, compression ABORTS and the
        session is preserved unchanged."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test", quiet_mode=True,
                protect_first_n=2, protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = _msgs()
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_mock_response("partial summary that got cut o", "length"),
        ):
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert result == msgs
        assert c._last_summary_truncated_failure is True
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False
        # The partial text must never be stored for iterative updates.
        assert c._previous_summary is None or "cut o" not in (c._previous_summary or "")

    def test_length_stop_falls_back_to_main_model_once(self):
        """With a distinct aux summary model, a length stop retries once on
        the main model (which may have a larger output budget) and succeeds."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="small-aux-model",
                quiet_mode=True,
            )
        truncated = _mock_response("partial...", "length")
        ok = _mock_response("full summary via main model", "stop")
        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[truncated, ok],
        ) as mock_call:
            result = c._generate_summary(_msgs(2))

        assert mock_call.call_count == 2
        assert result is not None
        assert "full summary via main model" in result
        assert c._last_summary_truncated_failure is False

    def test_repeated_truncation_escalates_cooldown_across_turns(self):
        """#69637: a later turn (e.g. an async delegation completion arriving after the cooldown lapsed)
        must not re-arm a fresh flat 30s attempt. Consecutive length-stopped summaries walk the durable
        60s -> 300s -> 900s ladder, one LLM call per turn, and the transcript is preserved every time."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)
        clock = [1000.0]
        recorded = []
        with (
            patch("agent.context_compressor.time.monotonic", side_effect=lambda: clock[0]),
            patch("agent.context_compressor.call_llm", return_value=_mock_response("partial...", "length")) as call,
        ):
            for _turn in range(4):
                msgs = _msgs()
                assert c.compress(msgs, current_tokens=999999) == msgs
                cooldown = c._summary_failure_cooldown_until - clock[0]
                recorded.append(round(cooldown))
                clock[0] += cooldown + 1  # the next turn arrives right after the cooldown lapses
        assert recorded == [60, 300, 900, 900]
        assert call.call_count == 4
        # Truncation keeps its own streak: the timeout ladder (which arms the deterministic stall
        # fallback via _prior_timeout_failures) is untouched.
        assert c._consecutive_timeout_failures == 0
        assert c._consecutive_truncation_failures == 4

    def test_other_transient_failures_keep_short_cooldown(self):
        """Control: empty-content stays on the flat 30s rung and does not bump the truncation streak."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)
        clock = [1000.0]
        with (
            patch("agent.context_compressor.time.monotonic", side_effect=lambda: clock[0]),
            patch("agent.context_compressor.call_llm", side_effect=RuntimeError("LLM returned empty content")),
        ):
            for _turn in range(2):
                c.compress(_msgs(), current_tokens=999999)
                assert round(c._summary_failure_cooldown_until - clock[0]) == 30
                clock[0] += 31
        assert c._consecutive_truncation_failures == 0

    def test_stop_finish_reason_still_succeeds(self):
        """Control: a normal stop-terminated summary is accepted unchanged."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_mock_response("complete summary", "stop"),
        ):
            result = c._generate_summary(_msgs(2))
        assert result is not None
        assert "complete summary" in result

    def test_refusal_body_is_rejected_and_never_becomes_previous_summary(self):
        """A stop-terminated refusal is not a usable compaction checkpoint."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test", quiet_mode=True,
                protect_first_n=2, protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = _msgs()
        refusal = "I can't produce this summary as requested. The instructions conflict with my operating rules."
        with patch("agent.context_compressor.call_llm", return_value=_mock_response(refusal, "stop")):
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert result == msgs
        assert c._last_summary_empty_content_failure is True
        assert c._last_compress_aborted is True
        assert c._previous_summary is None

    def test_provider_refusal_field_is_rejected_even_with_summary_shaped_content(self):
        """An explicit ``message.refusal`` wins over plausible-looking content (#118406)."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test", quiet_mode=True,
                protect_first_n=2, protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = _msgs()
        filler = "## Goal\nContinue the task.\n\n## Completed Actions\n1. Nothing yet."
        response = _mock_response(filler, "stop")
        response.choices[0].message.refusal = "policy refusal"
        with patch("agent.context_compressor.call_llm", return_value=response):
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert result == msgs
        assert c._last_summary_empty_content_failure is True
        assert c._last_compress_aborted is True
        assert "refusal content" in (c._last_summary_error or "")
        assert c._previous_summary is None

    def test_missing_finish_reason_still_succeeds(self):
        """Providers that omit finish_reason entirely must not be rejected."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        resp = {"choices": [{"message": {"content": "complete summary"}}]}
        with patch("agent.context_compressor.call_llm", return_value=resp):
            result = c._generate_summary(_msgs(2))
        assert result is not None
        assert "complete summary" in result

    def test_successful_summary_clears_truncated_flag(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        c._last_summary_truncated_failure = True
        c._summary_failure_cooldown_until = 0
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_mock_response("fine", "stop"),
        ):
            result = c._generate_summary(_msgs(2))
        assert result is not None
        assert c._last_summary_truncated_failure is False


class TestMicroSummarizeTruncationGuard:
    def test_length_stop_discards_partial_merge(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        c._micro_compact_rolling_summary = "existing rolling summary"
        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=_mock_response("partial merge tex", "length"),
        ):
            result = c._micro_summarize_one("user: hi\nassistant: hello")
        assert result is None

    def test_stop_finish_reason_merges(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        c._micro_compact_rolling_summary = "existing"
        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=_mock_response("merged summary", "stop"),
        ):
            result = c._micro_summarize_one("user: hi\nassistant: hello")
        assert result == "merged summary"

    def test_refusal_leaves_exchange_and_cursor_unchanged(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=1,
                protect_last_n=2,
                config_context_length=40960,
            )
        c._micro_compact_enabled = True
        messages = _msgs()
        assert c._next_exchange(messages) is not None
        cursor_before = c._micro_compact_cursor
        refusal = "I’m sorry, but I can’t provide the requested summary."
        response_content = f"<think>Check the applicable policy.</think>\n{refusal}"

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=_mock_response(response_content, "stop"),
        ):
            result = c._micro_compact(list(messages))

        assert result == messages
        assert c._micro_compact_cursor == cursor_before
        assert c._micro_compact_rolling_summary == ""
        assert not any(refusal in str(message.get("content")) for message in result)
