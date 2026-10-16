"""Regression tests for runtime-notification empty-reply quiet exit (#139299).

Empty responses on runtime-notification turns (``process_complete``,
``async_delegation_complete`` display kind) end the turn quietly: no nudge, no
paid retries, no provider switch, no ``(empty)`` sentinel. Ordinary turns and
notification turns where the model already ran a tool keep the full ladder.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import turn_empty_response as ter


def _agent():
    a = MagicMock()
    a.model = "test/model"
    a.provider = "test"
    a._current_streamed_assistant_text = ""
    a._has_content_after_think_block.return_value = False
    a._strip_think_blocks.side_effect = lambda t: t or ""
    a._last_content_with_tools = None
    a._thinking_prefill_retries = 2
    a._fallback_chain = None
    a._empty_content_retries = 0
    a._emit_diagnostic_status = MagicMock()
    a._buffer_diagnostic_status = MagicMock()
    a._flush_status_buffer = MagicMock()
    a._build_assistant_message = MagicMock(return_value={"role": "assistant", "content": ""})
    # MagicMock.getattr(foo, "bar", False) returns a truthy child mock unless
    # the attribute is explicitly set to a falsy value.  The real AIAgent
    # object never auto-creates these attributes.
    a._post_tool_empty_retried = False
    a._response_was_previewed = False
    return a


def _assistant_msg():
    return SimpleNamespace(reasoning=None, reasoning_content=None, reasoning_details=None)


def _empty_response():
    choice = SimpleNamespace(message=_assistant_msg(), finish_reason="stop")
    choice.message.content = ""
    choice.message.tool_calls = None
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


@pytest.fixture
def loop_agent(monkeypatch):
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = lambda **_kwargs: _empty_response()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    monkeypatch.setattr("model_tools.handle_function_call", MagicMock(return_value="unused"))
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("agent.turn_empty_response.interruptible_backoff_sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent, "_persist_session", MagicMock())
    monkeypatch.setattr(agent, "_save_trajectory", MagicMock())
    monkeypatch.setattr(agent, "_cleanup_task_resources", MagicMock())
    return agent


def _recover(agent, messages, **kwargs):
    return ter.recover_empty_response(
        agent,
        _assistant_msg(),
        None,
        "stop",
        final_response="",
        messages=messages,
        api_messages=[],
        conversation_history=[],
        active_system_prompt="sys",
        api_call_count=1,
        turn_exit_reason=None,
        preflight_compression_blocked=False,
        **kwargs,
    )


def _completed_tool_round():
    return [
        {"role": "user", "content": "Run the background task."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "Process exited successfully.", "tool_call_id": "call_1"},
    ]


class TestNotificationTurnLoop:
    def test_process_complete_empty_reply_exits_without_nudge(self, loop_agent):
        result = loop_agent.run_conversation(
            "[process completed] Background task finished.",
            conversation_history=_completed_tool_round(),
            persist_user_display_kind="process_complete",
        )

        assert loop_agent.client.chat.completions.create.call_count == 1
        assert result["final_response"] == ""
        assert result["turn_exit_reason"] == "notification_no_reply"
        provider_messages = loop_agent.client.chat.completions.create.call_args.kwargs["messages"]
        assert all(
            "You just executed tool calls but returned an empty response."
            not in str(message.get("content") or "")
            for message in provider_messages
        )

    def test_human_turn_empty_reply_still_retries(self, loop_agent):
        loop_agent.run_conversation(
            "What happened?",
            conversation_history=_completed_tool_round(),
        )

        assert loop_agent.client.chat.completions.create.call_count > 1


class TestNotificationTurnDetection:
    def test_user_row_with_process_complete_no_tool_is_notification_turn(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi!"},
            {"role": "user", "content": "[process completed] task done", "display_kind": "process_complete"},
        ]
        assert ter._notification_turn_without_tool_round(messages) is True

    def test_user_row_with_async_delegation_complete_no_tool_is_notification_turn(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi!"},
            {"role": "user", "content": "[delegate finished]", "display_kind": "async_delegation_complete"},
        ]
        assert ter._notification_turn_without_tool_round(messages) is True

    def test_notification_turn_with_tool_is_not_detected(self):
        messages = [
            {"role": "user", "content": "[process completed]", "display_kind": "process_complete"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ]
        assert ter._notification_turn_without_tool_round(messages) is False

    def test_ordinary_user_turn_is_not_detected(self):
        messages = [
            {"role": "user", "content": "ordinary message"},
        ]
        assert ter._notification_turn_without_tool_round(messages) is False

    def test_ordinary_user_turn_with_tools_is_not_detected(self):
        messages = [
            {"role": "user", "content": "run task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ]
        assert ter._notification_turn_without_tool_round(messages) is False

    def test_empty_messages_is_not_detected(self):
        assert ter._notification_turn_without_tool_round([]) is False

    def test_notification_with_nudge_row_as_user_injection_is_not_detected(self):
        """The model replied with text, then the user sent a follow-up message:
        that follow-up is the message the next turn answers, not a notification."""
        messages = [
            {"role": "user", "content": "[process completed]", "display_kind": "process_complete"},
            {"role": "assistant", "content": "got it"},
            {"role": "user", "content": "now do X"},
        ]
        assert ter._notification_turn_without_tool_round(messages) is False


class TestNotificationQuietExit:
    def test_process_complete_empty_reply_exits_quietly(self):
        agent = _agent()
        messages = [
            {"role": "user", "content": "[process completed]", "display_kind": "process_complete"},
        ]
        verdict = _recover(agent, messages)
        assert verdict.action == "break"
        assert verdict.final_response == ""
        assert verdict.turn_exit_reason == ter.NOTIFICATION_NO_REPLY_EXIT

    def test_async_delegation_complete_empty_reply_exits_quietly(self):
        agent = _agent()
        messages = [
            {"role": "user", "content": "delegate X returned", "display_kind": "async_delegation_complete"},
        ]
        verdict = _recover(agent, messages)
        assert verdict.action == "break"
        assert verdict.final_response == ""
        assert verdict.turn_exit_reason == ter.NOTIFICATION_NO_REPLY_EXIT

    def test_notification_turn_with_tool_mid_task_not_quiet_exit(self):
        """A model that acted on a notification with a tool call is mid-task;
        _notification_turn_without_tool_round is False, so the quiet exit path
        is skipped. The post-tool nudge fires instead (verdict is 'continue'
        with two scaffolding rows added to messages)."""
        agent = _agent()
        agent._last_content_with_tools = "mid-task reply"
        agent._last_content_tools_all_housekeeping = False
        messages = [
            {"role": "user", "content": "[process completed]", "display_kind": "process_complete"},
            {
                "role": "assistant",
                "content": "mid-task reply",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ]
        verdict = _recover(agent, messages)
        # Should not exit via notification_no_reply.
        assert verdict.turn_exit_reason != ter.NOTIFICATION_NO_REPLY_EXIT
        # Post-tool nudge fired: verdict is "continue" and scaffolding was added.
        assert verdict.action == "continue"
        # Two scaffolding rows: (empty) assistant + nudge user.
        assert len(messages) == 5
        assert messages[3].get("content") == "(empty)"
        assert messages[3].get("_empty_recovery_synthetic") is True
        assert messages[4].get("role") == "user"
        assert messages[4].get("_empty_recovery_synthetic") is True


class TestNotificationNoReplyExitConstant:
    def test_constant_is_importable_and_stable(self):
        assert ter.NOTIFICATION_NO_REPLY_EXIT == "notification_no_reply"


class TestResponseFiltersNotificationNoReply:
    def test_notification_no_reply_with_empty_text_is_intentional_silence(self):
        from gateway.response_filters import is_intentional_silence_agent_result

        agent_result = {"failed": False, "turn_exit_reason": ter.NOTIFICATION_NO_REPLY_EXIT}
        assert is_intentional_silence_agent_result(agent_result, "") is True
        assert is_intentional_silence_agent_result(agent_result, " ") is True

    def test_notification_no_reply_with_non_empty_text_is_not_silence(self):
        from gateway.response_filters import is_intentional_silence_agent_result

        agent_result = {"failed": False, "turn_exit_reason": ter.NOTIFICATION_NO_REPLY_EXIT}
        assert is_intentional_silence_agent_result(agent_result, "got it") is False

    def test_notification_no_reply_failed_turn_is_not_silence(self):
        from gateway.response_filters import is_intentional_silence_agent_result

        agent_result = {"failed": True, "turn_exit_reason": ter.NOTIFICATION_NO_REPLY_EXIT}
        assert is_intentional_silence_agent_result(agent_result, "") is False

    def test_ordinary_empty_response_exhausted_is_not_notification_silence(self):
        """An empty_response_exhausted turn with no text and no silence
        marker is not intentional silence -- it falls through to normal
        delivery."""
        from gateway.response_filters import is_intentional_silence_agent_result

        agent_result = {"failed": False, "turn_exit_reason": "empty_response_exhausted"}
        # "(empty)" is not a LIVE_GATEWAY_SILENT_MARKERS entry; it reaches
        # delivery through the empty-response path, not silence.
        assert is_intentional_silence_agent_result(agent_result, "(empty)") is False
        # But with the [SILENT] marker it IS silence (unrelated to notification path).
        assert is_intentional_silence_agent_result(agent_result, "[SILENT]") is True

    def test_notification_no_reply_alongside_real_silence_marker(self):
        from gateway.response_filters import is_intentional_silence_agent_result

        agent_result = {"failed": False, "turn_exit_reason": ter.NOTIFICATION_NO_REPLY_EXIT}
        assert is_intentional_silence_agent_result(agent_result, "[SILENT]") is True

    def test_notification_no_reply_none_result_is_not_silence(self):
        from gateway.response_filters import is_intentional_silence_agent_result

        assert is_intentional_silence_agent_result(None, "") is False
