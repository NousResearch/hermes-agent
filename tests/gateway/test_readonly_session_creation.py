"""Regression tests for read-only gateway commands creating ghost sessions."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner(session_key: str):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._queue_depth = lambda _session_key, adapter=None: 0
    runner._session_key_for_source = MagicMock(return_value=session_key)
    runner.session_store = MagicMock()
    runner.session_store.get_session.return_value = None
    runner.session_store.get_or_create_session.side_effect = AssertionError(
        "read-only command should not create a session"
    )
    runner._session_db = None
    return runner


@pytest.mark.asyncio
async def test_status_without_session_does_not_create_stub_session():
    source = _make_source()
    runner = _make_runner("agent:main:telegram:dm:c1")
    event = SimpleNamespace(source=source)

    result = await runner._handle_status_command(event)

    assert "No active session yet." in result
    runner.session_store.get_session.assert_called_once_with(source)


@pytest.mark.asyncio
async def test_usage_without_session_does_not_create_stub_session():
    source = _make_source()
    runner = _make_runner("agent:main:telegram:dm:c1")
    event = SimpleNamespace(source=source)

    result = await runner._handle_usage_command(event)

    assert "No usage data available" in result
    runner.session_store.get_session.assert_called_once_with(source)
