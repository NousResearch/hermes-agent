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
        ) as mock_call:
            result = c.compress(msgs, current_tokens=999999, force=True)
            automatic_retry = c.compress(msgs, current_tokens=999999)

        assert result == msgs
        assert automatic_retry == msgs
        assert mock_call.call_count == 1
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

    def test_configured_auxiliary_route_falls_back_to_main_model_once(self):
        """A task-configured aux route must not be selected again for the retry."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model", provider="main-provider", base_url="https://main.example/v1",
                api_key="main-key", api_mode="chat_completions", quiet_mode=True,
            )

        calls = []

        def _call(**kwargs):
            calls.append(kwargs)
            route_info = kwargs["route_info"]
            if kwargs.get("provider") == "main-provider":
                route_info.update(provider="main-provider", model="main-model")
                return _mock_response("full summary via main model", "stop")
            route_info.update(provider="google", model="gemini-2.5-flash-lite")
            return _mock_response("partial configured-route summary", "length")

        with patch("agent.context_compressor.call_llm", side_effect=_call):
            result = c._generate_summary(_msgs(2))

        assert result is not None
        assert "full summary via main model" in result
        assert len(calls) == 2
        assert calls[1]["provider"] == "main-provider"
        assert calls[1]["model"] == "main-model"
        assert calls[1]["base_url"] == "https://main.example/v1"
        assert calls[1]["bypass_task_route"] is True

    def test_same_model_on_different_aux_provider_still_falls_back(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="shared-model", provider="main-provider", quiet_mode=True)

        calls = []

        def _call(**kwargs):
            calls.append(kwargs)
            kwargs["route_info"].update(
                provider="main-provider" if kwargs.get("bypass_task_route") else "aux-provider",
                model="shared-model",
            )
            return _mock_response(
                "full summary" if kwargs.get("bypass_task_route") else "partial summary",
                "stop" if kwargs.get("bypass_task_route") else "length",
            )

        with patch("agent.context_compressor.call_llm", side_effect=_call):
            result = c._generate_summary(_msgs(2))

        assert result is not None
        assert len(calls) == 2
        assert calls[1]["provider"] == "main-provider"
        assert calls[1]["model"] == "shared-model"
        assert calls[1]["bypass_task_route"] is True

    def test_same_provider_model_on_task_endpoint_still_falls_back(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model", provider="anthropic",
                base_url="https://main.example/v1", quiet_mode=True,
            )

        calls = []

        def _call(**kwargs):
            calls.append(kwargs)
            bypass = kwargs.get("bypass_task_route", False)
            kwargs["route_info"].update(
                provider="anthropic",
                model="main-model",
                task_endpoint_override="false" if bypass else "true",
            )
            return _mock_response(
                "full summary" if bypass else "partial summary",
                "stop" if bypass else "length",
            )

        with patch("agent.context_compressor.call_llm", side_effect=_call):
            result = c._generate_summary(_msgs(2))

        assert result is not None
        assert len(calls) == 2
        assert calls[1]["base_url"] == "https://main.example/v1"
        assert calls[1]["bypass_task_route"] is True

    def test_auto_main_runtime_retry_bypasses_configured_auxiliary_route(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="main-model", provider="", base_url="", quiet_mode=True)

        calls = []

        def _call(**kwargs):
            calls.append(kwargs)
            bypass = kwargs.get("bypass_task_route", False)
            kwargs["route_info"].update(
                provider="auto-main-provider" if bypass else "aux-provider",
                model="main-model" if bypass else "aux-model",
            )
            return _mock_response("full summary" if bypass else "partial summary", "stop" if bypass else "length")

        with patch("agent.context_compressor.call_llm", side_effect=_call):
            result = c._generate_summary(_msgs(2))

        assert result is not None
        assert len(calls) == 2
        assert calls[1]["provider"] == ""
        assert calls[1]["model"] == "main-model"
        assert calls[1]["base_url"] == ""
        assert calls[1]["bypass_task_route"] is True

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
