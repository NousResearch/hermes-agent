"""Machinery turns get a quiet empty-response exit, not a paid retry budget.

An injected internal_notification turn (``display_kind=internal_notification``,
#82888 machinery lane) is not owed a reply. When the model answers it with an
empty completion, the quiet exit ends the turn with no nudge, no paid retries,
no provider switch, and no ``(empty)`` sentinel: #135816 logged notification
turns burning twelve requests where every paid retry came back empty too. Only
machinery turns collapse; a human turn keeps the full ladder.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent import turn_empty_response as ter
from gateway import response_filters as gateway_filters
from gateway.response_filters import MACHINERY_DISPLAY_KINDS as gateway_machinery_kinds


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


class TestMachineryEmptyRetryBudget:
    def test_internal_notification_turn_skips_paid_retries(self):
        agent = _agent()
        messages = [
            {"role": "user", "content": "[kanban] board updated", "display_kind": "internal_notification"},
        ]
        verdict = _recover(agent, messages)
        assert verdict.action == "break"
        assert verdict.final_response == ""
        assert verdict.turn_exit_reason == ter.NOTIFICATION_NO_REPLY_EXIT
        assert agent._empty_content_retries == 0  # no retry was scheduled
        agent._emit_diagnostic_status.assert_not_called()  # no "retrying (1/3)" chatter

    def test_human_turn_keeps_full_budget_and_retries(self):
        agent = _agent()
        messages = [{"role": "user", "content": "What happened?"}]
        verdict = _recover(agent, messages)
        assert verdict.action == "continue"
        assert agent._empty_content_retries == 1

    def test_only_machinery_kinds_collapse(self):
        agent = _agent()
        messages = [
            {"role": "user", "content": "background result", "display_kind": "background_result"},
        ]
        verdict = _recover(agent, messages)
        assert verdict.action == "continue"
        assert agent._empty_content_retries == 1


class TestMachineryKindsMirrorGateway:
    def test_agent_mirror_matches_gateway_machinery_kinds(self):
        # Cross-layer drift guard: the agent-side mirror
        # (agent/turn_empty_response.py _MACHINERY_DISPLAY_KINDS) must stay equal to the
        # set the gateway actually derives its machinery kinds from
        # (gateway/response_filters.py MACHINERY_DISPLAY_KINDS). If the gateway ever grows
        # a new machinery kind, this mirror must follow, otherwise the quiet exit silently
        # keeps the new machinery turn on the paid human retry ladder.
        assert ter._MACHINERY_DISPLAY_KINDS == gateway_machinery_kinds
        assert "internal_notification" in gateway_machinery_kinds

    def test_quiet_exit_reason_and_row_match_the_gateway(self):
        # The gateway recognizes the exit by this string and the persisted row by this
        # marker; both are mirrored on the agent side because agent/ cannot import gateway/.
        assert ter.NOTIFICATION_NO_REPLY_EXIT == gateway_filters.NOTIFICATION_NO_REPLY_EXIT
        assert ter.MACHINERY_NO_REPLY_ROW in gateway_filters.LIVE_GATEWAY_SILENT_MARKERS
