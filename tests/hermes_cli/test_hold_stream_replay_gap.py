"""The classic CLI viewer of a held turn paints and speaks only the rewrite, also when a replay gap
makes it recover the reply from the settled receipt (built from a real held agent result)."""
import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.fakes.redactor_plugin import RAW, REWRITE, redactor_home
from tests.hermes_cli.test_gateway_agent import SID, FakeClient, connect_with, frame


def _held_receipt(home: Path) -> dict:
    """The settled CLI receipt of a real held turn (``take_cli_result_fields`` over its result)."""
    from gateway.session_cli_turn import take_cli_result_fields
    from hermes_cli import plugins
    from run_agent import AIAgent

    with FakeLLMServer([Text(RAW)]) as server:
        redactor_home(home, server.base_url, hold=True)
        plugins._reset_plugin_managers_for_tests()
        try:
            core = AIAgent(base_url=server.base_url, api_key="sk-fake", provider="custom", model="fake-model",
                           quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["todo"])
            early = []
            core.stream_delta_callback = early.append
            result = core.run_conversation("show config")
        finally:
            plugins._reset_plugin_managers_for_tests()
    assert result["final_response"] == REWRITE and not [t for t in early if t]
    return {**take_cli_result_fields(result), "completed": True, "api_calls": 1}


@pytest.mark.parametrize("gap", [False, True], ids=["streamed", "replay_gap"])
def test_held_reply_stays_rewritten_across_cli_replay_gap(gap):
    from hermes_cli.gateway_agent import GatewayAgent
    from hermes_cli.gateway_connection import GatewayConnection

    receipt = _held_receipt(Path(os.environ["HERMES_HOME"]))
    terminal = frame("message.complete", {"text": REWRITE, "outcome": "completed", "status": "complete"})
    events = [frame("session.replay_gap", {}, admission=None)] if gap else [terminal]
    conn = GatewayConnection(connect=connect_with(FakeClient(events, result=receipt)))
    conn.start(5)
    shown, spoken = [], []
    try:
        conn.state(SID)
        client = GatewayAgent(connection=conn, session_id=SID, stream_delta_callback=shown.append)
        out = client.run_conversation("show config", stream_callback=spoken.append)
    finally:
        conn.close()
    assert out["final_response"] == REWRITE
    assert RAW not in shown + spoken, {"display": shown, "tts": spoken}
