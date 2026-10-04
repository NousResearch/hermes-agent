"""Provider-private replay carriers must not trigger the thinking-only prefill.

A provider profile may declare ``native_reasoning_details_type``: a provider-private
replay carrier (``{"type": "<provider>.native_assistant", ...}``) attached to EVERY
response on that route via ``reasoning_details``. It is replay data, not model
reasoning. Counting it as "structured reasoning" made a carrier-only empty look
thinking-only, so ``recover_empty_response`` spent two ``_thinking_prefill`` calls
(whose stubs the thinking-only sanitizer then drops from the API copy) before the
first real empty-response retry.

These tests drive ``recover_empty_response`` directly and assert which ladder rung
a given empty response lands on.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent import turn_empty_response as ter

CARRIER = {"type": "acme.native_assistant", "data": "x"}


def _agent():
    """Minimal agent surface ``recover_empty_response`` touches up to the retry rung."""
    status = []
    return SimpleNamespace(
        model="m",
        provider="acme",
        api_mode="chat_completions",
        base_url=None,
        api_key=None,
        _current_streamed_assistant_text="",
        _has_content_after_think_block=lambda text: False,
        _strip_think_blocks=lambda text: text or "",
        _last_content_with_tools=None,
        _last_content_tools_all_housekeeping=False,
        _post_tool_empty_retried=False,
        _thinking_prefill_retries=0,
        _empty_content_retries=0,
        _fallback_chain=[],
        _buffer_diagnostic_status=status.append,
        _emit_diagnostic_status=status.append,
        _build_assistant_message=lambda msg, finish_reason: {
            "role": "assistant", "content": msg.content or "", "finish_reason": finish_reason,
        },
        _status=status,
    )


def _assistant_message(**fields):
    base = dict(content="", reasoning=None, reasoning_content=None, reasoning_details=None)
    base.update(fields)
    return SimpleNamespace(**base)


def _response():
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1_000, completion_tokens=0, total_tokens=1_000),
    )


def _recover(agent, assistant_message, messages):
    with patch.object(ter, "interruptible_backoff_sleep", lambda *a, **k: None):
        return ter.recover_empty_response(
            agent, assistant_message, _response(), "stop",
            final_response="", messages=messages, api_messages=list(messages),
            conversation_history=None, active_system_prompt="sys", api_call_count=1,
            turn_exit_reason=None, preflight_compression_blocked=False,
        )


class TestCarrierOnlyEmptyTakesRetryRung:
    def test_carrier_only_empty_skips_prefill_and_retries(self):
        agent = _agent()
        messages = [{"role": "user", "content": "hi"}]
        verdict = _recover(agent, _assistant_message(reasoning_details=[CARRIER]), messages)

        assert verdict.action == "continue"
        # Retry rung, not the prefill rung: no prefill stub appended, retry counted.
        assert agent._thinking_prefill_retries == 0
        assert agent._empty_content_retries == 1
        assert not any(m.get("_thinking_prefill") for m in messages)

    def test_real_reasoning_content_still_prefills(self):
        """Control: genuine reasoning must keep routing to the prefill rung."""
        agent = _agent()
        messages = [{"role": "user", "content": "hi"}]
        verdict = _recover(
            agent,
            _assistant_message(reasoning_content="thinking...", reasoning_details=[CARRIER]),
            messages,
        )

        assert verdict.action == "continue"
        assert agent._thinking_prefill_retries == 1
        assert agent._empty_content_retries == 0
        assert messages[-1].get("_thinking_prefill") is True

    def test_real_reasoning_detail_alongside_carrier_still_prefills(self):
        agent = _agent()
        messages = [{"role": "user", "content": "hi"}]
        details = [CARRIER, {"type": "reasoning.text", "text": "hm"}]
        _recover(agent, _assistant_message(reasoning_details=details), messages)

        assert agent._thinking_prefill_retries == 1
        assert agent._empty_content_retries == 0


class TestModelReasoningDetails:
    def test_carrier_only_is_not_reasoning(self):
        from agent.turn_empty_response import _model_reasoning_details
        assert _model_reasoning_details([CARRIER]) is False

    def test_carrier_plus_real_detail_is_reasoning(self):
        from agent.turn_empty_response import _model_reasoning_details
        assert _model_reasoning_details([CARRIER, {"type": "reasoning.text", "text": "hm"}]) is True

    def test_empty_and_none(self):
        from agent.turn_empty_response import _model_reasoning_details
        assert _model_reasoning_details(None) is False
        assert _model_reasoning_details([]) is False

    def test_object_shaped_details(self):
        from agent.turn_empty_response import _model_reasoning_details
        assert _model_reasoning_details([SimpleNamespace(type=CARRIER["type"])]) is False
        assert _model_reasoning_details([SimpleNamespace(type="reasoning.summary")]) is True

    def test_untyped_dict_is_reasoning(self):
        from agent.turn_empty_response import _model_reasoning_details
        assert _model_reasoning_details([{"text": "hm"}]) is True

    def test_single_non_list_detail(self):
        from agent.turn_empty_response import _model_reasoning_details
        assert _model_reasoning_details(CARRIER) is False
        assert _model_reasoning_details({"type": "reasoning.text"}) is True
