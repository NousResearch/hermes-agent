"""Thinking-only recovery escalates instead of re-sending identical requests.

Follow-up to the request-local recovery (8edd3120da): both completion reminders used the
same text, so the second retry was byte-identical to the first, and the plain empty-response
retries re-sent the original request three times. Each identical request makes a reasoning
model re-derive the same reasoning in new words (the "paraphrased reasoning" symptom).
"""

from __future__ import annotations

from types import SimpleNamespace

from agent import empty_response_guard
from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE, _THINKING_ONLY_FINAL_NUDGE
from agent.turn_empty_response import _retry_empty, recover_empty_response


class _FakeAgent:
    """Just enough of AIAgent for the reminder / retry rungs of the ladder."""

    def __init__(self):
        self._current_streamed_assistant_text = ""
        self._last_content_with_tools = None
        self._last_content_tools_all_housekeeping = False
        self._post_tool_empty_retried = False
        self._thinking_prefill_retries = 0
        self._empty_content_retries = 0
        self._empty_response_retry_hint = None
        self._ephemeral_reasoning_off = False
        self._fallback_chain = []
        self.statuses: list[str] = []
        self.model = "test-model"
        self.provider = "custom"
        self.base_url = "http://127.0.0.1:1/v1"
        self.api_key = "test"
        self.api_mode = "chat_completions"

    def _has_content_after_think_block(self, text):
        return bool((text or "").strip())

    def _strip_think_blocks(self, text):
        return text or ""

    def _buffer_diagnostic_status(self, text):
        self.statuses.append(text)

    _emit_diagnostic_status = _buffer_diagnostic_status


def _reasoning_only():
    return SimpleNamespace(
        content="", reasoning=None, reasoning_details=None, tool_calls=None,
        reasoning_content="Let me restate the task and plan the steps.",
    )


def _rung(agent, messages):
    return recover_empty_response(
        agent, _reasoning_only(), SimpleNamespace(usage=None), "stop", final_response="",
        messages=messages, api_messages=None, conversation_history=None,
        active_system_prompt=None, api_call_count=1, turn_exit_reason=None,
        preflight_compression_blocked=False,
    )


def test_second_reminder_differs_and_turns_reasoning_off():
    agent = _FakeAgent()
    messages = [{"role": "user", "content": "Say hello"}]

    assert _rung(agent, messages).action == "continue"
    assert agent._empty_response_retry_hint == _CODEX_INCOMPLETE_NUDGE
    assert agent._ephemeral_reasoning_off is False

    agent._empty_response_retry_hint = None  # response intake consumes the hint
    assert _rung(agent, messages).action == "continue"
    assert agent._empty_response_retry_hint == _THINKING_ONLY_FINAL_NUDGE
    assert _THINKING_ONLY_FINAL_NUDGE != _CODEX_INCOMPLETE_NUDGE
    assert agent._ephemeral_reasoning_off is True
    assert messages == [{"role": "user", "content": "Say hello"}]  # still request-local


def test_reasoning_only_plain_retry_capped_and_sent_without_thinking(monkeypatch):
    agent = _FakeAgent()
    agent._thinking_prefill_retries = 2
    monkeypatch.setattr("agent.turn_empty_response.interruptible_backoff_sleep", lambda *a, **k: None)

    action, _, _ = _retry_empty(
        agent, SimpleNamespace(usage=None), "stop", True, messages=[], conversation_history=None,
        api_call_count=3, observed_generation=True,
    )
    assert action == "continue"
    assert agent._ephemeral_reasoning_off is True

    agent._ephemeral_reasoning_off = False
    action, _, _ = _retry_empty(
        agent, SimpleNamespace(usage=None), "stop", True, messages=[], conversation_history=None,
        api_call_count=4, observed_generation=True,
    )
    assert action is None
    assert agent._empty_content_retries == empty_response_guard.REASONING_ONLY_RETRY_BUDGET


def test_truly_empty_retry_keeps_budget_and_reasoning(monkeypatch):
    agent = _FakeAgent()
    monkeypatch.setattr("agent.turn_empty_response.interruptible_backoff_sleep", lambda *a, **k: None)
    action, _, _ = _retry_empty(
        agent, SimpleNamespace(usage=None), "stop", True, messages=[], conversation_history=None,
        api_call_count=1, observed_generation=False,
    )
    assert action == "continue"
    assert f"(1/{empty_response_guard.DEFAULT_EMPTY_RETRY_BUDGET})" in agent.statuses[-1]
    assert agent._ephemeral_reasoning_off is False


def test_reasoning_off_override_is_one_shot_on_the_wire():
    from agent.chat_completion_helpers import _reasoning_config_for_wire

    agent = SimpleNamespace(reasoning_config={"enabled": True, "effort": "medium"}, _ephemeral_reasoning_off=True)
    assert _reasoning_config_for_wire(agent) == {"enabled": False, "effort": "none"}
    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "medium"}


def test_post_tool_reminder_is_its_own_block_on_anthropic_messages():
    from agent.turn_empty_response import add_empty_response_retry_hint

    tool_tail = [
        {"role": "user", "content": "Read the file"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
                                                            "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "The secret word is PELICAN."},
    ]

    agent = _FakeAgent()
    agent.api_mode = "anthropic_messages"
    agent._empty_response_retry_hint = _CODEX_INCOMPLETE_NUDGE
    sent = [dict(m) for m in tool_tail]
    add_empty_response_retry_hint(agent, sent)
    assert sent[2]["content"] == "The secret word is PELICAN."  # tool output untouched
    assert sent[3] == {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE}

    from agent.anthropic_message_convert import convert_messages_to_anthropic

    _, converted = convert_messages_to_anthropic(sent)
    last = converted[-1]
    assert last["role"] == "user"
    kinds = [b.get("type") for b in last["content"]]
    assert kinds[0] == "tool_result" and "text" in kinds[1:]  # one user turn: tool_result, then the reminder


def test_post_tool_reminder_stays_on_the_tail_for_chat_completions():
    from agent.turn_empty_response import add_empty_response_retry_hint

    agent = _FakeAgent()  # api_mode chat_completions
    agent._empty_response_retry_hint = _CODEX_INCOMPLETE_NUDGE
    sent = [{"role": "user", "content": "q"}, {"role": "tool", "tool_call_id": "t1", "content": "out"}]
    add_empty_response_retry_hint(agent, sent)
    assert [m["role"] for m in sent] == ["user", "tool"]
    assert sent[-1]["content"].endswith(_CODEX_INCOMPLETE_NUDGE)


def test_reasoning_off_reaches_request_on_reasoning_capable_route():
    from agent.transports.chat_completions import ChatCompletionsTransport

    kwargs = ChatCompletionsTransport().build_kwargs(
        "some/reasoning-model", [{"role": "user", "content": "hi"}], None,
        reasoning_config={"enabled": False, "effort": "none"}, supports_reasoning=True,
    )
    assert kwargs["extra_body"]["reasoning"] == {"enabled": False, "effort": "none"}
