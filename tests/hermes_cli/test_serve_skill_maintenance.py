"""Exercise the real periodic maintenance loop with isolated on-disk state."""
import asyncio
import json
from contextlib import suppress

import pytest


@pytest.mark.asyncio
async def test_removed_sessions_keep_profile_idle_watermark(tmp_path, monkeypatch):
    import time
    from pathlib import Path

    import tui_gateway.server as gateway
    from agent.curator import load_state, save_state
    from hermes_cli.web_server_sessions import _auto_archive_ticker_loop, _skill_maintenance_idle_for

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "skills").mkdir()
    (tmp_path / "config.yaml").write_text(
        "curator:\n  enabled: true\n  interval_hours: 168\n"
        "  min_idle_hours: 0.002\n  prune_builtins: false\n", encoding="utf-8")
    save_state({"last_run_at": "2020-01-01T00:00:00+00:00", "run_count": 0})
    # Let the actual timer age past its idle threshold before recent activity.
    task = asyncio.create_task(_auto_archive_ticker_loop(interval_s=.05, initial_delay_s=8))
    try:
        await asyncio.sleep(7.5)
        for reason in ("tui_close", "ws_orphan_reap"):
            recent = time.time()
            with gateway._sessions_lock:
                gateway._sessions["watermark"] = {
                    "last_active": recent, "running": False, "profile_home": str(tmp_path)}
            assert gateway._close_session_by_id("watermark", end_reason=reason)
            assert _skill_maintenance_idle_for(recent - 3600) < 2
        # An active sibling profile must not hold this profile's idle clock.
        with gateway._sessions_lock:
            gateway._sessions["other-profile"] = {
                "running": True, "profile_home": str(tmp_path / "other")}
        assert _skill_maintenance_idle_for(recent - 3600) is not None
        await asyncio.sleep(2)
        assert load_state()["run_count"] == 0
        async with asyncio.timeout(12):
            while "consolidation off" not in (load_state().get("last_run_summary") or ""):
                await asyncio.sleep(.05)
        assert "consolidation off" in load_state()["last_run_summary"]
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        with gateway._sessions_lock:
            gateway._sessions.pop("other-profile", None)
            gateway._sessions.pop("watermark", None)


@pytest.mark.asyncio
async def test_serve_timer_runs_due_curator_once_and_honors_pause(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "skills").mkdir()
    (tmp_path / "config.yaml").write_text(
        "curator:\n  enabled: true\n  consolidate: false\n  interval_hours: 168\n"
        "  min_idle_hours: 0\n  prune_builtins: false\n", encoding="utf-8")
    from agent.curator import load_state, save_state, set_paused
    from hermes_cli.web_server_sessions import _auto_archive_ticker_loop

    save_state({"last_run_at": "2020-01-01T00:00:00+00:00", "run_count": 0, "paused": True})
    task = asyncio.create_task(_auto_archive_ticker_loop(interval_s=.02, initial_delay_s=0))
    try:
        await asyncio.sleep(.15)
        assert load_state()["run_count"] == 0
        set_paused(False)
        # Active turns must suppress maintenance even with a zero idle threshold.
        import tui_gateway.server as gateway
        with gateway._sessions_lock:
            gateway._sessions['maintenance-test'] = {"running": True}
        try:
            await asyncio.sleep(.15)
            assert load_state()["run_count"] == 0
        finally:
            with gateway._sessions_lock:
                gateway._sessions.pop('maintenance-test', None)
        async with asyncio.timeout(8):
            while load_state()["run_count"] == 0:
                await asyncio.sleep(.02)
        await asyncio.sleep(.15)
        state = json.loads((tmp_path / "skills" / ".curator_state").read_text())
        assert state["run_count"] == 1
        assert "consolidation off" in state["last_run_summary"]
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
