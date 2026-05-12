from unittest.mock import MagicMock

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner
from tests.gateway.test_stale_code_replay import _telegram_event


@pytest.mark.asyncio
async def test_stale_code_monitor_restarts_when_idle():
    runner, _adapter = make_restart_runner()
    runner._detect_stale_code = MagicMock(return_value=True)
    runner._trigger_stale_code_restart = MagicMock(return_value=True)

    result = await runner._stale_code_monitor_tick()

    assert result == "idle_restart"
    runner._trigger_stale_code_restart.assert_called_once_with()


@pytest.mark.asyncio
async def test_stale_code_monitor_defers_while_active():
    runner, _adapter = make_restart_runner()
    runner._running_agents["telegram:42:u-1"] = object()
    runner._detect_stale_code = MagicMock(return_value=True)
    runner._trigger_stale_code_restart = MagicMock(return_value=True)

    result = await runner._stale_code_monitor_tick()

    assert result == "deferred_active"
    assert runner._stale_code_restart_deferred is True
    runner._trigger_stale_code_restart.assert_not_called()

    runner._running_agents.clear()
    result = await runner._stale_code_monitor_tick()

    assert result == "idle_restart"
    assert runner._stale_code_restart_deferred is False
    runner._trigger_stale_code_restart.assert_called_once_with()


@pytest.mark.asyncio
async def test_hot_path_stale_code_fallback_queues_message(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, _adapter = make_restart_runner()
    runner._detect_stale_code = MagicMock(return_value=True)
    runner._trigger_stale_code_restart = MagicMock(return_value=True)

    response = await runner._handle_message(_telegram_event(text="hot path fallback"))

    assert "restarting" in response.lower()
    assert list((tmp_path / "gateway" / "replay_queue").glob("*.json"))
    runner._trigger_stale_code_restart.assert_called_once_with()


def test_detect_stale_code_when_sentinel_mtime_increases(tmp_path):
    runner, _adapter = make_restart_runner()
    sentinel = tmp_path / "gateway-run.py"
    sentinel.write_text("old", encoding="utf-8")
    runner._stale_code_sentinel_paths = lambda: (sentinel,)
    runner._stale_code_baseline = runner._capture_stale_code_snapshot()

    old_ns = sentinel.stat().st_mtime_ns
    sentinel.write_text("new", encoding="utf-8")
    # Force a strictly newer mtime on filesystems with coarse timestamp resolution.
    sentinel.touch()
    if sentinel.stat().st_mtime_ns <= old_ns:
        import os

        os.utime(sentinel, ns=(old_ns + 1_000_000_000, old_ns + 1_000_000_000))

    assert runner._detect_stale_code() is True
