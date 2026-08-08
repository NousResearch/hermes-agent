"""Session-expiry watcher must skip sessions whose agent turn is still running."""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry
from gateway.session_state import SessionState


def _make_runner(session_key: str, session_id: str):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner._running = True
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._last_session_store_prune_ts = 0.0

    entry = SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=datetime.now() - timedelta(hours=2),
        updated_at=datetime.now() - timedelta(hours=2),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    entry.expiry_finalized = False

    runner.session_store = MagicMock()
    runner.session_store._ensure_loaded = MagicMock()
    runner.session_store._entries = {session_key: entry}
    runner.session_store._is_session_expired = MagicMock(return_value=True)
    runner.session_store._lock = MagicMock()
    runner.session_store._lock.__enter__ = MagicMock(return_value=None)
    runner.session_store._lock.__exit__ = MagicMock(return_value=None)
    runner.session_store._save = MagicMock()

    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()
    runner._clear_conversation_scope = MagicMock()
    runner._sweep_idle_cached_agents = MagicMock(return_value=0)
    return runner, entry


async def _run_once(runner, interval=1):
    """Drive the watcher through one pass, then stop."""
    _orig = asyncio.sleep
    calls = {"n": 0}

    async def _fast_sleep(_):
        calls["n"] += 1
        if calls["n"] > 1:  # initial startup + one tail sleep => one pass done
            runner._running = False
        await _orig(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await runner._session_expiry_watcher(interval)


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_expiry_watcher_skips_running_session(mock_invoke_hook):
    """A session whose agent turn is running is not finalized by expiry."""
    key = "agent:main:telegram:dm:42"
    runner, _ = _make_runner(key, "sess-running")
    runner._sessions = {key: SessionState()}
    runner._sessions[key].turn.agent = MagicMock()  # turn is mid-flight

    await _run_once(runner)

    assert mock_invoke_hook.call_args_list == [], (
        "running session must not be finalized on_session_finalize"
    )
    runner._evict_cached_agent.assert_not_called()
    runner._cleanup_agent_resources.assert_not_called()
    runner._clear_conversation_scope.assert_not_called()


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_expiry_watcher_still_finalizes_idle_session(mock_invoke_hook):
    """A truly idle session must still be finalized (guard is not a blanket)."""
    key = "agent:main:telegram:dm:43"
    runner, _ = _make_runner(key, "sess-idle")
    runner._sessions = {}  # no running turn

    def _hook_and_stop(*a, **kw):
        runner._running = False
        return None

    mock_invoke_hook.side_effect = _hook_and_stop

    await _run_once(runner)

    finalize = [
        c for c in mock_invoke_hook.call_args_list
        if c.args and c.args[0] == "on_session_finalize"
    ]
    assert any(c[1].get("session_id") == "sess-idle" for c in finalize), (
        "idle session should still be finalized by expiry"
    )
