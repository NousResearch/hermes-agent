"""TUI, Desktop and the classic CLI (local viewers of the canonical gateway) show a held reply only
as its rewrite.

Real ``GatewayRunner._run_agent`` on a ``LOCAL`` source with the ``LocalSessionAdapter`` and a
recording session authority (the viewers' event stream), a real ``AIAgent`` and a loopback model.
With a ``hold_stream`` redactor loaded, no ``message.delta`` / ``message.interim`` frame carries
text, and the turn's ``message.complete`` carries the rewritten final (the handler reply is not
swallowed as "already streamed"). A plain transform keeps streaming the raw reply (control).
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.session import SessionSource
from gateway.session_local import LocalSessionAdapter
from gateway.session_results import execution_result
from gateway.session_settlement_recovery import completion_payload
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.redactor_plugin import REWRITE, raw_leaks, redactor_home, redactor_turn
from tests.gateway.test_run_progress_topics import _make_runner


async def _local_turn(*, hold, monkeypatch):
    import gateway.run as gateway_run
    from hermes_cli import plugins

    home = Path(os.environ["HERMES_HOME"])
    authority = MagicMock()
    authority.sessions = {"sid": MagicMock()}
    authority.db.get_session.return_value = {"runtime_generation": 1}
    authority.publish_execution.return_value = True  # a viewer is attached and takes each frame
    source = SessionSource(platform=Platform.LOCAL, chat_id="sid", user_id="u", chat_type="dm")
    with FakeLLMServer(redactor_turn(delay_per_chunk=0.01)) as srv:
        redactor_home(home, srv.base_url, hold=hold)
        plugins._reset_plugin_managers_for_tests()
        monkeypatch.setattr(gateway_run, "_hermes_home", home)
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
            "api_key": "sk-fake-e2e", "base_url": srv.base_url, "provider": "custom", "api_mode": "chat_completions"})
        local = LocalSessionAdapter(authority)
        runner = _make_runner(local)
        runner.session_authority = authority
        runner.config.streaming = StreamingConfig(enabled=True, edit_interval=0.01, buffer_threshold=1, cursor="")
        captured = {}
        token = execution_result.set(captured)
        try:
            result = await runner._run_agent(message="show me the config", context_prompt="", history=[],
                                             source=source, session_id="sid", session_key="agent:main:local:sid")
        finally:
            execution_result.reset(token)
            plugins._reset_plugin_managers_for_tests()
    reply = None if result.get("already_sent") else result["final_response"]
    complete = completion_payload({"admission_id": "a1", "request_id": "r1"}, {"outcome": "completed"},
                                  reply, captured)
    frames = [(c.args[2], c.args[3]) for c in authority.publish_execution.call_args_list]
    return frames, complete


def _texts(frames, *kinds):
    return [payload.get("text") for kind, payload in frames if kind in kinds and isinstance(payload, dict)]


@pytest.mark.asyncio
async def test_a_held_local_turn_completes_with_only_the_rewrite(monkeypatch):
    frames, complete = await _local_turn(hold=True, monkeypatch=monkeypatch)

    assert [t for t in _texts(frames, "message.delta", "message.interim") if t] == []
    assert raw_leaks(str(payload) for _, payload in frames) == []
    assert complete["text"] == REWRITE


@pytest.mark.asyncio
async def test_a_plain_transform_still_streams_to_local_viewers(monkeypatch):
    frames, _ = await _local_turn(hold=False, monkeypatch=monkeypatch)

    assert raw_leaks(_texts(frames, "message.delta"))
