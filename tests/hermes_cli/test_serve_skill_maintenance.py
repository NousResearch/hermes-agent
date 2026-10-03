"""Exercise the real periodic maintenance loop with isolated on-disk state."""
import asyncio
import json
from contextlib import suppress
from queue import Queue

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
    import hermes_cli.web_server_sessions as web_server_sessions

    ticks = Queue()
    original_maintenance = web_server_sessions._maybe_run_skill_maintenance

    def observed_maintenance(started_at):
        try:
            return original_maintenance(started_at)
        finally:
            ticks.put(None)

    monkeypatch.setattr(web_server_sessions, "_maybe_run_skill_maintenance", observed_maintenance)

    save_state({"last_run_at": "2020-01-01T00:00:00+00:00", "run_count": 0, "paused": True})
    task = asyncio.create_task(web_server_sessions._auto_archive_ticker_loop(
        interval_s=.02, initial_delay_s=0))
    try:
        await asyncio.to_thread(ticks.get, True, 8)
        assert load_state()["run_count"] == 0
        # Active turns must suppress maintenance even with a zero idle threshold.
        import tui_gateway.server as gateway
        with gateway._sessions_lock:
            gateway._sessions['maintenance-test'] = {"running": True}
        set_paused(False)
        try:
            await asyncio.to_thread(ticks.get, True, 8)
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


def test_serve_maintenance_tick_fires_once_per_hosted_profile_in_its_scope(tmp_path, monkeypatch):
    """Multi-profile ``hermes serve``: the ``on_maintenance_tick('serve')`` hook reaches EVERY hosted
    profile's own plugins once, each inside that profile's runtime scope, so a fail-closed
    ``get_secret`` resolves the profile's own ``.env`` (never UnscopedSecretError, never another's)."""
    from pathlib import Path

    import agent.curator as curator
    import hermes_cli.web_server_sessions as web_server_sessions
    from agent.secret_scope import set_multiplex_active
    from hermes_cli.plugins import _reset_plugin_managers_for_tests

    fake_home = tmp_path / "home"
    launch = fake_home / ".hermes"
    other = launch / "profiles" / "s5probe-b"
    log = tmp_path / "ticks.jsonl"
    for home, value in ((launch, "launch-secret"), (other, "b-secret")):
        plugin = home / "plugins" / "tick_probe"
        plugin.mkdir(parents=True)
        (home / ".env").write_text(f"S5_PROBE_KEY={value}\n", encoding="utf-8")
        (home / "config.yaml").write_text("plugins:\n  enabled: [tick_probe]\n", encoding="utf-8")
        (plugin / "plugin.yaml").write_text("name: tick_probe\nversion: 0.1.0\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(
            "import json\n"
            "def _tick(surface, **_):\n"
            "    from agent.secret_scope import get_secret\n"
            "    from hermes_constants import get_hermes_home\n"
            "    try:\n"
            "        secret = get_secret('S5_PROBE_KEY')\n"
            "    except Exception as exc:\n"
            "        secret = type(exc).__name__\n"
            f"    with open({str(log)!r}, 'a', encoding='utf-8') as f:\n"
            "        f.write(json.dumps([get_hermes_home().name, surface, secret]) + '\\n')\n"
            "def register(ctx):\n"
            "    ctx.register_hook('on_maintenance_tick', _tick)\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("S5_PROBE_KEY", raising=False)
    monkeypatch.setattr(curator, "maybe_run_curator", lambda **_: None)
    _reset_plugin_managers_for_tests()
    set_multiplex_active(True)
    try:
        web_server_sessions._maybe_run_skill_maintenance(0.0)
    finally:
        set_multiplex_active(False)
        _reset_plugin_managers_for_tests()

    ticks = sorted(tuple(json.loads(line)) for line in log.read_text(encoding="utf-8").splitlines())
    assert ticks == [(".hermes", "serve", "launch-secret"), ("s5probe-b", "serve", "b-secret")]
