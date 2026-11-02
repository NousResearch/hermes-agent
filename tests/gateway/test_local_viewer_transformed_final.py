"""A ``transform_llm_output`` rewrite reaches a local viewer (Ink TUI, Desktop) on ``message.complete``.

The viewer painted the raw reply from ``message.delta`` frames; the transform ran after the stream.
A streamed local turn is delivered by the event stream itself, so ``_handle_message`` answers
``None`` and the completion frame was the only carrier left, with ``text: null``: the raw reply
stayed on screen while the session stored (and replays) the rewrite.
"""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.session_results import execution_result
from gateway.session_settlement_recovery import completion_payload
from gateway.turn_context import TurnContext
from tests.gateway.test_local_viewer_events_submit import _gateway_runner, _RecordingAgent

RAW = "Your api key is SECRET-123."
REWRITE = "Your api key is [REDACTED]."


class _StreamedAgent(_RecordingAgent):
    transformed = True

    def run_conversation(self, message, **kwargs):
        if self.transformed:
            return {"final_response": REWRITE, "messages": [], "response_transformed": True,
                    "pre_transform_response": RAW}
        return {"final_response": RAW, "messages": []}


def _settle(platform, transformed):
    _StreamedAgent.transformed = transformed
    ctx = TurnContext(
        source=SessionSource(platform=platform, chat_id="sid", user_id="u"),
        message="show me the config", history=[], session_id="sid", session_key="key", user_config={},
        AIAgent=_StreamedAgent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    captured = {}
    token = execution_result.set(captured)
    try:
        turn = TurnRunner(_gateway_runner(), ctx)
        turn._approval_owner = (SimpleNamespace(sessions={"sid": SimpleNamespace(tool_progress_mode=None)},
                                                profile_id="p", adopt_agent=lambda *a: None), "sid", 1)
        turn.run_sync()
    finally:
        execution_result.reset(token)
    # The streamed turn was delivered by the event stream: the handler's reply is None.
    return completion_payload({"admission_id": "a1", "request_id": "r1"}, {"outcome": "completed"}, None, captured)


def test_a_streamed_local_turn_completes_with_the_transformed_final():
    from tui_gateway.contracts.registry import EVENTS
    complete = _settle(Platform.LOCAL, transformed=True)
    assert complete["text"] == REWRITE and complete["response_transformed"] is True
    EVENTS["message.complete"].payload.model_validate(
        {k: v for k, v in complete.items() if k not in ("content", "admission_id", "outcome")})


@pytest.mark.parametrize("platform, transformed", [(Platform.LOCAL, False), (Platform.TELEGRAM, True)])
def test_the_stream_stays_the_delivery_without_a_local_rewrite(platform, transformed):
    # An unrewritten streamed reply is already on screen; a messaging turn's chat holds its own copy.
    assert _settle(platform, transformed)["text"] is None
