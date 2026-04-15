import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _bind_internal_status_methods(runner: GatewayRunner) -> None:
    for name in (
        "_internal_status_enabled",
        "_build_busy_ack_message",
        "_build_internal_status_callback",
        "_build_background_review_callback",
    ):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner, GatewayRunner))


@pytest.mark.asyncio
async def test_busy_ack_for_end_users_is_neutral(monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"display": {}})
    runner, adapter = make_restart_runner()
    _bind_internal_status_methods(runner)
    runner._busy_ack_ts = {}

    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "api_call_count": 7,
        "max_iterations": 60,
        "current_tool": "search_files",
    }
    runner._running_agents["session"] = running_agent
    runner._running_agents_ts["session"] = 0

    event = MagicMock()
    event.text = "new message"
    event.message_id = "42"
    event.source = make_restart_source()

    await runner._handle_active_session_busy_message(event, "session")

    assert adapter.sent == ["One sec, finishing the previous reply and then I'll handle this."]
    assert "Interrupting current task" not in adapter.sent[0]
    assert "iteration" not in adapter.sent[0]
    assert "running:" not in adapter.sent[0]


def test_detailed_busy_ack_requires_explicit_internal_status_config(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"display": {"internal_status": True}},
    )
    runner, _adapter = make_restart_runner()
    _bind_internal_status_methods(runner)

    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "api_call_count": 7,
        "max_iterations": 60,
        "current_tool": "search_files",
    }
    runner._running_agents_ts["session"] = 0

    message = runner._build_busy_ack_message(
        source=make_restart_source(),
        running_agent=running_agent,
        session_key="session",
        now=120.0,
    )

    assert "Interrupting current task" in message
    assert "iteration 7/60" in message
    assert "running: search_files" in message


@pytest.mark.asyncio
async def test_internal_callbacks_are_disabled_by_default(monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"display": {}})
    runner, adapter = make_restart_runner()
    _bind_internal_status_methods(runner)
    loop = asyncio.get_running_loop()

    status_cb = runner._build_internal_status_callback(
        source=make_restart_source(),
        adapter=adapter,
        chat_id="123456",
        metadata=None,
        loop=loop,
    )
    review_cb = runner._build_background_review_callback(
        source=make_restart_source(),
        adapter=adapter,
        chat_id="123456",
        metadata=None,
        loop=loop,
    )

    assert status_cb is None
    assert review_cb is None


@pytest.mark.asyncio
async def test_internal_callbacks_send_when_explicitly_enabled(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"display": {"internal_status": True}},
    )
    runner, adapter = make_restart_runner()
    _bind_internal_status_methods(runner)
    loop = asyncio.get_running_loop()

    status_cb = runner._build_internal_status_callback(
        source=make_restart_source(),
        adapter=adapter,
        chat_id="123456",
        metadata=None,
        loop=loop,
    )
    review_cb = runner._build_background_review_callback(
        source=make_restart_source(),
        adapter=adapter,
        chat_id="123456",
        metadata=None,
        loop=loop,
    )

    assert status_cb is not None
    assert review_cb is not None

    status_cb("lifecycle", "⚠️ Connection to provider dropped")
    review_cb("💾 User profile updated")
    await asyncio.sleep(0.05)

    assert adapter.sent == [
        "⚠️ Connection to provider dropped",
        "💾 User profile updated",
    ]
