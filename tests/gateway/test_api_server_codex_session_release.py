"""#137451: an API-server turn must not outlive its Codex app-server child.

Each request builds a fresh agent and drops it after the turn. The ``codex app-server``
child is not garbage-collected with it, so unless the turn closes it the process leaks for
the gateway's life and keeps the Codex thread's writer lock, so the next request cannot
resume the thread. Session tool state (terminal processes, memory provider, the
background-review fork in ``_active_children``) must survive, so neither ``close()`` nor
``release_clients()`` may be what ends the turn.
"""
import asyncio
from functools import partial
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_mod
from gateway.platforms import api_server_runs as runs_mod
from gateway.platforms.api_server import APIServerAdapter
from run_agent import AIAgent


class _CodexSession:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def _agent_with_live_codex_child():
    agent = MagicMock()
    agent.session_id = "session"
    agent._last_compaction_in_place = False
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    agent.run_conversation.return_value = {"final_response": "ok", "messages": []}
    session = _CodexSession()
    agent._codex_session = session
    agent._close_codex_session = partial(AIAgent._close_codex_session, agent)
    return agent, session


def _assert_child_closed_and_session_state_kept(agent, session):
    assert session.closed == 1
    assert agent._codex_session is None
    agent.close.assert_not_called()
    agent.release_clients.assert_not_called()


@pytest.mark.asyncio
async def test_session_chat_turn_closes_the_codex_child(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agent, session = _agent_with_live_codex_child()

    with patch.object(adapter, "_create_agent", return_value=agent):
        result, _usage = await adapter._run_agent("hi", [], session_id="session")

    assert result["final_response"] == "ok"
    _assert_child_closed_and_session_state_kept(agent, session)


@pytest.mark.asyncio
async def test_failed_session_chat_turn_still_closes_the_codex_child(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agent, session = _agent_with_live_codex_child()
    agent.run_conversation.side_effect = RuntimeError("turn blew up")

    with patch.object(adapter, "_create_agent", return_value=agent), pytest.raises(RuntimeError):
        await adapter._run_agent("hi", [], session_id="session")

    _assert_child_closed_and_session_state_kept(agent, session)


def test_v1_run_turn_closes_the_codex_child(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agent, session = _agent_with_live_codex_child()
    launch = runs_mod._RunLaunch(
        adapter, "run-1", asyncio.Queue(), "session", None, False, "hi", [], False,
        agent_kwargs={"room_dispatch": None}, request_profile=None,
        browser_control_principal=None, browser_control_transport_family=None)

    result, _usage, _runtime = runs_mod._run_agent_sync(
        adapter, launch, agent, lambda _event: None, _api_server=api_server_mod)

    assert result["final_response"] == "ok"
    _assert_child_closed_and_session_state_kept(agent, session)
