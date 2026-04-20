"""Tests that on_session_finalize and on_session_reset plugin hooks fire in the gateway."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-old",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    new_session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-new",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = new_session_entry
    runner.session_store.reset_session.return_value = new_session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    return runner


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_reset_fires_finalize_hook(mock_invoke_hook):
    """/new must fire on_session_finalize with the OLD session id."""
    runner = _make_runner()

    await runner._handle_reset_command(_make_event("/new"))

    mock_invoke_hook.assert_any_call(
        "on_session_finalize", session_id="sess-old", platform="telegram"
    )


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_reset_fires_reset_hook(mock_invoke_hook):
    """/new must fire on_session_reset with the NEW session id."""
    runner = _make_runner()

    await runner._handle_reset_command(_make_event("/new"))

    mock_invoke_hook.assert_any_call(
        "on_session_reset", session_id="sess-new", platform="telegram"
    )


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_finalize_before_reset(mock_invoke_hook):
    """on_session_finalize must fire before on_session_reset."""
    runner = _make_runner()

    await runner._handle_reset_command(_make_event("/new"))

    calls = [c for c in mock_invoke_hook.call_args_list
             if c[0][0] in ("on_session_finalize", "on_session_reset")]
    hook_names = [c[0][0] for c in calls]
    assert hook_names == ["on_session_finalize", "on_session_reset"]


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_shutdown_fires_finalize_for_active_agents(mock_invoke_hook):
    """Gateway stop() must fire on_session_finalize for each active agent."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._background_tasks = set()
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._shutdown_event = MagicMock()
    runner.adapters = {}
    runner._exit_reason = "test"
    runner._exit_code = None
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._running_agents_ts = {}
    runner._update_runtime_status = MagicMock()

    agent1 = MagicMock()
    agent1.session_id = "sess-a"
    agent2 = MagicMock()
    agent2.session_id = "sess-b"
    runner._running_agents = {"key-a": agent1, "key-b": agent2}

    with patch("gateway.status.remove_pid_file"), \
         patch("gateway.status.write_runtime_status"):
        await runner.stop()

    finalize_calls = [
        c for c in mock_invoke_hook.call_args_list
        if c[0][0] == "on_session_finalize"
    ]
    session_ids = {c[1]["session_id"] for c in finalize_calls}
    assert session_ids == {"sess-a", "sess-b"}


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook", side_effect=Exception("boom"))
async def test_hook_error_does_not_break_reset(mock_invoke_hook):
    """Plugin hook errors must not prevent /new from completing."""
    runner = _make_runner()

    result = await runner._handle_reset_command(_make_event("/new"))

    # Should still return a success message despite hook errors
    assert "Session reset" in result or "New session" in result
