"""A machinery turn the model leaves unanswered stays silent through the gateway's turn runner (#135816).

The agent ends a background-completion / internal-notification turn quietly when the model adds
nothing (``turn_exit_reason == "notification_no_reply"``). The TurnRunner's result projection used
to fill the empty reply with the "Processing completed but no response" notice before the delivery
seam's silence check ran, so every late completion still reached the chat as an error.
"""
from types import SimpleNamespace

import pytest

from hermes_agent.gateway.config import Platform
from hermes_agent.gateway.response_filters import NOTIFICATION_NO_REPLY_EXIT


def _runner_reply(result):
    from hermes_agent.gateway.run_turn_runner import TurnRunner
    from hermes_agent.gateway.session import SessionSource
    from hermes_agent.gateway.turn_context import TurnContext
    from tests.gateway.test_local_viewer_events_submit import _gateway_runner, _RecordingAgent

    class _Agent(_RecordingAgent):
        def run_conversation(self, message, **kwargs):
            return {"messages": [], "provider": "custom", "model": "m", **result}

    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="sid", user_id="u"),
        message="[IMPORTANT: Background process proc_1 completed normally]", history=[], session_id="sid",
        session_key="key", user_config={}, AIAgent=_Agent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    return TurnRunner(_gateway_runner(), ctx).run_sync()["final_response"]


def test_quiet_machinery_exit_reaches_delivery_empty():
    assert _runner_reply({"final_response": "", "api_calls": 1, "completed": True,
                          "turn_exit_reason": NOTIFICATION_NO_REPLY_EXIT}) == ""


@pytest.mark.parametrize("result", [
    # Work done, no text, any other exit: the user still learns nothing came back.
    {"final_response": "", "api_calls": 1, "completed": True, "turn_exit_reason": "text_response(finish_reason=stop)"},
    # A failed turn carrying the quiet reason is still a failure.
    {"final_response": "", "api_calls": 1, "failed": True, "completed": False,
     "turn_exit_reason": NOTIFICATION_NO_REPLY_EXIT, "error": "boom"},
])
def test_other_empty_results_still_get_a_notice(result):
    assert _runner_reply(result).strip()
