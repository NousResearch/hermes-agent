"""G3 correction — boolean-coercion parity for max_spawn_per_tick.

Direct dispatch (``dispatch_once``) rejects ``bool`` via an explicit
``isinstance(..., bool)`` guard in the ``_per_tick_cap`` computation. But
the CLI and gateway config-load paths coerce with ``int(value)``:

    int(True)  == 1  →  cap of 1  (wrong — should be None)
    int(False) == 0  →  None      (accidentally correct, but for the wrong reason)

This is a configuration-validity inconsistency: the same YAML value produces
different behaviour depending on which code path reads it. These tests pin
the required parity — every bool → None on every path — and were written
RED before the implementation fix (strict TDD).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# shared fixtures — mirror the existing watcher test module's setup
# (NO module deletion: the kb imported above must stay the same object the
#  watcher resolves at runtime, or monkeypatch targets won't take effect)
# ---------------------------------------------------------------------------
@pytest.fixture()
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialised default kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# CLI config-path: _coerce_positive_int must reject bool
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [True, False])
def test_cli_config_rejects_bool_max_spawn_per_tick(monkeypatch, bad):
    """CLI config load must treat bool max_spawn_per_tick as None (no cap),
    matching direct dispatch's ``isinstance(..., bool)`` guard."""
    test_home = tempfile.mkdtemp(prefix="kanban_per_tick_bool_cli_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    # Rebind Path.home so config resolution finds the temp home.
    monkeypatch.setattr(Path, "home", lambda: Path(test_home).parent)

    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch

    fake_config = {"kanban": {"max_in_progress": 5, "max_spawn_per_tick": bad}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}
    # Patch the REAL entry point: ``kanban_db.dispatch_once`` is a deprecated
    # compat shim re-exported only for external plugins, and ``_cmd_dispatch``
    # resolves the symbol through ``kanban_db_dispatch`` at call time — so
    # patching the old alias leaves the real call unpatched.
    monkeypatch.setattr(
        kanban_db_dispatch, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )
    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)
    assert captured.get("max_spawn_per_tick") is None, (
        f"CLI config path must reject bool max_spawn_per_tick={bad!r} → None, "
        f"got {captured.get('max_spawn_per_tick')!r}"
    )


# ---------------------------------------------------------------------------
# Gateway config-path: int(raw) must reject bool
# ---------------------------------------------------------------------------
async def _drive_watcher(runner, *, target_ticks: int):
    """Run ``_kanban_dispatcher_watcher`` until ``target_ticks`` ticks occur."""
    import gateway.kanban_watchers as kw

    from hermes_cli import kanban_db_dispatch as _kbd_mod

    _orig_sleep = asyncio.sleep
    state = {"ticks": 0}
    _real_reap = _kbd_mod.reap_worker_zombies

    def _counting_reap():
        state["ticks"] += 1
        if state["ticks"] >= target_ticks:
            runner._running = False
        return _real_reap()

    async def _fast_sleep(_d):
        await _orig_sleep(0)

    with patch.object(_kbd_mod, "reap_worker_zombies", _counting_reap), \
         patch.object(kw.asyncio, "sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_dispatcher_watcher(),
            timeout=30.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [True, False])
async def test_gateway_config_rejects_bool_max_spawn_per_tick(kanban_home, monkeypatch, bad):
    """Gateway config load must treat bool max_spawn_per_tick as None (no cap),
    matching direct dispatch's ``isinstance(..., bool)`` guard."""
    import gateway.kanban_watchers as kw

    fake_config = {"kanban": {"dispatch_in_gateway": True, "max_spawn_per_tick": bad}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}

    def _spy_dispatch_once(conn, **kw):
        captured.update(kw)
        return kb.DispatchResult()

    # The watcher resolves these through ``kanban_db_dispatch`` at call time;
    # ``kb`` is a deprecated compat shim (removal 2026-09-14) and patching it
    # alone is a silent no-op.
    from hermes_cli import kanban_db_dispatch as _kbd_mod

    monkeypatch.setattr(_kbd_mod, "dispatch_once", _spy_dispatch_once)
    monkeypatch.setattr(_kbd_mod, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(_kbd_mod, "has_spawnable_review", lambda conn: False)

    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_dispatcher_lock_handle = None

    await _drive_watcher(runner, target_ticks=2)

    assert "max_spawn_per_tick" in captured, "dispatch_once was never called"
    assert captured.get("max_spawn_per_tick") is None, (
        f"Gateway config path must reject bool max_spawn_per_tick={bad!r} → None, "
        f"got {captured.get('max_spawn_per_tick')!r}"
    )


# ---------------------------------------------------------------------------
# Regression: valid positive int still works on both config paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("good", [1, 3, 10])
def test_cli_config_accepts_positive_int_max_spawn_per_tick(monkeypatch, good):
    """Regression: a genuine positive int must still pass through the CLI
    config path unchanged after the bool-rejection fix."""
    test_home = tempfile.mkdtemp(prefix="kanban_per_tick_int_cli_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    monkeypatch.setattr(Path, "home", lambda: Path(test_home).parent)

    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch

    fake_config = {"kanban": {"max_in_progress": 5, "max_spawn_per_tick": good}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}
    # Same compat-shim hazard as the bool test above.
    monkeypatch.setattr(
        kanban_db_dispatch, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )
    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)
    assert captured.get("max_spawn_per_tick") == good


@pytest.mark.asyncio
async def test_gateway_config_accepts_positive_int_max_spawn_per_tick(kanban_home, monkeypatch):
    """Regression: a genuine positive int must still pass through the gateway
    config path unchanged after the bool-rejection fix."""
    import gateway.kanban_watchers as kw

    fake_config = {"kanban": {"dispatch_in_gateway": True, "max_spawn_per_tick": 3}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}

    def _spy_dispatch_once(conn, **kw):
        captured.update(kw)
        return kb.DispatchResult()

    # The watcher resolves these through ``kanban_db_dispatch`` at call time;
    # ``kb`` is a deprecated compat shim (removal 2026-09-14) and patching it
    # alone is a silent no-op.
    from hermes_cli import kanban_db_dispatch as _kbd_mod

    monkeypatch.setattr(_kbd_mod, "dispatch_once", _spy_dispatch_once)
    monkeypatch.setattr(_kbd_mod, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(_kbd_mod, "has_spawnable_review", lambda conn: False)

    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_dispatcher_lock_handle = None

    await _drive_watcher(runner, target_ticks=2)

    assert captured.get("max_spawn_per_tick") == 3
