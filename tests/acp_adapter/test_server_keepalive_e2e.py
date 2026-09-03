"""Real prompt-path tests: HermesACPAgent.prompt() ↔ keepalive wiring.

Addresses PR #75124 review: prior test_server_keepalive.py only constructs
TurnKeepalive directly and never exercises `HermesACPAgent.prompt()`, so it
does not validate the actual start/mark_activity/finally-stop wiring landed
in acp_adapter/server.py:1681-1875. These tests use the same acp_agent
fixture pattern as tests/acp/test_mcp_e2e.py and assert:

  1. Success path: prompt() calls keepalive.start() and keepalive.stop().
  2. Executor-failure path: keepalive.stop() STILL runs (the `finally` block).
  3. Disabled path: when make_turn_keepalive returns None, prompt() does not
     crash on a missing keepalive.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import PromptResponse, TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


@pytest.fixture()
def mock_manager():
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def acp_agent(mock_manager):
    return HermesACPAgent(session_manager=mock_manager)


def _wire_mock_conn(acp_agent):
    mock_conn = MagicMock(spec=acp.Client)
    mock_conn.session_update = AsyncMock()
    mock_conn.request_permission = AsyncMock()
    acp_agent._conn = mock_conn
    return mock_conn


def _make_spy_keepalive():
    spy = MagicMock()
    spy.start = MagicMock()
    spy.stop = MagicMock()
    spy.mark_activity = MagicMock()
    return spy


@pytest.mark.asyncio
async def test_prompt_starts_and_stops_keepalive_on_success(acp_agent, mock_manager):
    """Happy path: prompt() → keepalive.start() at turn commit, .stop() in finally."""
    resp = await acp_agent.new_session(cwd="/tmp")
    session_id = resp.session_id
    state = mock_manager.get_session(session_id)
    _wire_mock_conn(acp_agent)

    spy = _make_spy_keepalive()

    def _run_ok(user_message, conversation_history=None, task_id=None, **kwargs):
        return {
            "final_response": "hi",
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "hi"},
            ],
        }
    state.agent.run_conversation = _run_ok

    prompt = [TextContentBlock(type="text", text="hello")]
    with patch("acp_adapter.server.make_turn_keepalive", return_value=spy):
        r = await acp_agent.prompt(prompt=prompt, session_id=session_id)

    assert isinstance(r, PromptResponse)
    spy.start.assert_called_once()
    spy.stop.assert_called_once()


@pytest.mark.asyncio
async def test_prompt_stops_keepalive_on_executor_failure(acp_agent, mock_manager):
    """Failure path: run_conversation raises → keepalive.stop() STILL fires (finally)."""
    resp = await acp_agent.new_session(cwd="/tmp")
    session_id = resp.session_id
    state = mock_manager.get_session(session_id)
    _wire_mock_conn(acp_agent)

    spy = _make_spy_keepalive()

    def _boom(user_message, conversation_history=None, task_id=None, **kwargs):
        raise RuntimeError("executor exploded")
    state.agent.run_conversation = _boom

    prompt = [TextContentBlock(type="text", text="hello")]
    with patch("acp_adapter.server.make_turn_keepalive", return_value=spy):
        r = await acp_agent.prompt(prompt=prompt, session_id=session_id)

    # The server catches the executor exception and returns end_turn — the
    # important assertion is that keepalive was started AND then cleaned up
    # even though the turn body raised.
    assert isinstance(r, PromptResponse)
    spy.start.assert_called_once()
    spy.stop.assert_called_once()

    # Executor-failure cleanup we promised the sweeper is also intact:
    assert state.is_running is False
    assert state.current_prompt_text == ""


@pytest.mark.asyncio
async def test_prompt_when_keepalive_disabled_via_config(acp_agent, mock_manager):
    """Disabled path: make_turn_keepalive returns None → prompt() must not crash."""
    resp = await acp_agent.new_session(cwd="/tmp")
    session_id = resp.session_id
    state = mock_manager.get_session(session_id)
    _wire_mock_conn(acp_agent)

    def _run_ok(user_message, conversation_history=None, task_id=None, **kwargs):
        return {
            "final_response": "hi",
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "hi"},
            ],
        }
    state.agent.run_conversation = _run_ok

    prompt = [TextContentBlock(type="text", text="hello")]
    with patch("acp_adapter.server.make_turn_keepalive", return_value=None):
        r = await acp_agent.prompt(prompt=prompt, session_id=session_id)

    assert isinstance(r, PromptResponse)
