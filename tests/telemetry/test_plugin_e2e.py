"""End-to-end telemetry wiring test.

Unlike test_plugin_hooks.py (which calls the plugin's ``_on_*`` callbacks directly),
this drives the REAL dispatch chain that core uses at runtime:

    discover_plugins()  ->  plugin registers hooks  ->  invoke_hook(name, **kwargs)
                        ->  registered callback      ->  emitter  ->  tel_* tables

If the bundled plugin stops auto-loading, stops registering a hook, or the hook name
drifts from what core fires, the hand-written hook tests still pass but real runs go
dark. This test is the guard against that — it only touches public entry points
(``discover_plugins`` / ``invoke_hook``), exactly as core does.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

import hermes_state


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """A clean HERMES_HOME with state.db, a fresh plugin manager, and a fresh emitter."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"
    hermes_state.SessionDB(db_path=db)

    # Reset the global plugin-manager singleton so discovery re-runs in this HERMES_HOME.
    import hermes_cli.plugins as plugins_mod
    monkeypatch.setattr(plugins_mod, "_plugin_manager", None, raising=False)

    # Reset the emitter singleton so it binds to this state.db (and tear it down after).
    from agent.telemetry import emitter as emitter_mod
    emitter_mod.reset_emitter_for_tests(None)
    # Clear the plugin's per-run accumulators between tests.
    import plugins.telemetry as plug
    plug._runs.clear()

    yield db, plugins_mod, emitter_mod, plug

    try:
        emitter_mod.get_emitter().flush()
    except Exception:
        pass
    emitter_mod.reset_emitter_for_tests(None)
    monkeypatch.setattr(plugins_mod, "_plugin_manager", None, raising=False)


def _fire_one_turn(invoke_hook):
    """Fire the hook sequence of a single completed turn, as core does."""
    invoke_hook("on_session_start", session_id="s1",
                model="anthropic/claude-opus-4", platform="cli")
    invoke_hook("post_api_request", session_id="s1", platform="cli",
                provider="anthropic", base_url=None, model="claude-opus-4",
                api_duration=0.9,
                usage={"input_tokens": 1000, "output_tokens": 120,
                       "cache_read_tokens": 0, "cache_write_tokens": 0,
                       "reasoning_tokens": 0})
    invoke_hook("post_tool_call", session_id="s1", platform="cli",
                function_name="web_search", duration_ms=210, result='{"data": "ok"}')
    invoke_hook("on_session_finalize", session_id="s1", platform="cli",
                reason="shutdown")


def test_real_dispatch_writes_tel_rows(runtime):
    """The bundled plugin, loaded via discover_plugins, captures a turn end to end."""
    db, plugins_mod, emitter_mod, _plug = runtime

    plugins_mod.discover_plugins(force=True)

    # The plugin must have registered the lifecycle hooks core fires.
    mgr = plugins_mod.get_plugin_manager()
    registered = {k for k, v in getattr(mgr, "_hooks", {}).items() if v}
    for hook in ("on_session_start", "post_api_request", "post_tool_call",
                 "on_session_finalize"):
        assert hook in registered, f"core hook {hook!r} not registered by the plugin"

    _fire_one_turn(plugins_mod.invoke_hook)

    time.sleep(0.5)  # let the background writer drain
    emitter_mod.get_emitter().flush()

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    assert conn.execute("SELECT COUNT(*) c FROM tel_runs").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM tel_model_calls").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM tel_tool_calls").fetchone()["c"] == 1

    # Real values, not buckets.
    mc = conn.execute("SELECT provider, model FROM tel_model_calls").fetchone()
    assert mc["provider"] == "anthropic"
    assert mc["model"] == "claude-opus-4"
    tc = conn.execute("SELECT tool_name FROM tel_tool_calls").fetchone()
    assert tc["tool_name"] == "web_search"
    run = conn.execute("SELECT end_reason, model_call_count, tool_call_count "
                       "FROM tel_runs").fetchone()
    assert run["end_reason"] == "completed"
    assert run["model_call_count"] == 1
    assert run["tool_call_count"] == 1
    conn.close()


def test_local_disabled_writes_nothing(runtime, monkeypatch):
    """telemetry.local=false: the plugin does not auto-load, so no rows are written."""
    db, plugins_mod, emitter_mod, _plug = runtime
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"telemetry": {"local": False}},
        raising=False,
    )

    plugins_mod.discover_plugins(force=True)
    _fire_one_turn(plugins_mod.invoke_hook)
    time.sleep(0.3)
    try:
        emitter_mod.get_emitter().flush()
    except Exception:
        pass

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM tel_runs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM tel_model_calls").fetchone()[0] == 0
    conn.close()
