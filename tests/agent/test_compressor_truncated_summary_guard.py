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

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
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

    def test_length_stop_uses_configured_route_once_then_succeeds(self):
        """A default route may escape a deterministic output cap only via one different route."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="main-model", provider="primary", quiet_mode=True)
        truncated = _mock_response("partial...", "length")
        complete = _mock_response("complete summary", "stop")
        fallback = {"provider": "backup", "model": "backup-model", "timeout": 45}
        with (
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"fallback_chain": [fallback]},
            ),
            patch("agent.context_compressor.call_llm", side_effect=[truncated, complete]) as call,
        ):
            result = c._generate_summary(_msgs(2))

        assert result is not None
        assert call.call_count == 2
        assert "provider" not in call.call_args_list[0].kwargs
        assert call.call_args_list[1].kwargs["provider"] == "backup"
        assert call.call_args_list[1].kwargs["model"] == "backup-model"

    def test_repeated_default_route_truncation_enters_long_backoff(self):
        """Automatic compression must not replay an unchanged capped request every 30 seconds."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model", quiet_mode=True, protect_first_n=2, protect_last_n=2,
            )
        with patch(
            "agent.context_compressor.call_llm",
            return_value=_mock_response("partial...", "length"),
        ) as call:
            original = _msgs()
            assert c.compress(original, current_tokens=999999) == original
            # The automatic retry sees the persisted-in-memory cooldown and does not call the same route again.
            assert c._generate_summary(_msgs(2)) is None

        assert call.call_count == 1
        assert c._summary_failure_cooldown_until >= time.monotonic() + 899
        assert c._last_summary_truncated_failure is True

    def test_truncated_fallback_enters_terminal_backoff_without_losing_messages(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model", provider="primary", quiet_mode=True, protect_first_n=2, protect_last_n=2,
            )
        fallback = {"provider": "backup", "model": "backup-model"}
        with (
            patch(
                "agent.auxiliary_client._get_auxiliary_task_config",
                return_value={"fallback_chain": [fallback]},
            ),
            patch(
                "agent.context_compressor.call_llm",
                side_effect=[_mock_response("partial primary", "length"), _mock_response("partial backup", "length")],
            ) as call,
        ):
            original = _msgs()
            assert c.compress(original, current_tokens=999999) == original

        assert call.call_count == 2
        assert c._last_compress_aborted is True
        assert c._last_summary_truncated_failure is True
        assert c._summary_failure_cooldown_until >= time.monotonic() + 899

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
