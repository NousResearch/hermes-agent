"""Tests for the ``kanban_dispatch_tick`` plugin hook.

Verifies that ``kanban_db.dispatch_once`` fires the hook after each tick
(including idle ticks and dry runs), that ``outcome`` classifies the tick
correctly, and that a misbehaving subscriber never breaks the dispatcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_ticks(monkeypatch):
    """Register a capturing callback for the dispatch tick hook."""
    mgr = get_plugin_manager()
    events: list[dict] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("kanban_dispatch_tick", []).append(
        lambda **kw: events.append(kw)
    )
    try:
        yield events
    finally:
        mgr._hooks = saved


def test_dispatch_tick_hook_is_registered_as_valid():
    """The dispatch tick hook name is part of VALID_HOOKS."""
    assert "kanban_dispatch_tick" in VALID_HOOKS


def test_idle_tick_fires_hook_with_outcome_idle(kanban_home, captured_ticks):
    """An empty board produces one hook fire with outcome='idle'."""
    conn = kb.connect()
    try:
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 12345)
    finally:
        conn.close()
    assert len(captured_ticks) == 1
    kw = captured_ticks[0]
    assert kw["outcome"] == "idle"
    assert kw["dry_run"] is False
    assert isinstance(kw["result"], kb.DispatchResult)


def test_active_tick_fires_hook_with_outcome_ok(kanban_home, captured_ticks):
    """A tick that spawns a worker fires the hook with outcome='ok'."""
    conn = kb.connect()
    try:
        # 'default' is guaranteed to exist as a profile, so the dispatcher
        # actually spawns it instead of skipping as non-spawnable.
        tid = kb.create_task(conn, title="t", assignee="default")
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    finally:
        conn.close()
    ok_events = [kw for kw in captured_ticks if kw["outcome"] == "ok"]
    assert ok_events, (
        "expected an ok tick, got "
        f"{[kw['outcome'] for kw in captured_ticks]}"
    )
    kw = ok_events[-1]
    result = kw["result"]
    assert any(row[0] == tid for row in result.spawned)


def test_dry_run_tick_carries_dry_run_flag(kanban_home, captured_ticks):
    """dry_run=True is propagated to the hook payload."""
    conn = kb.connect()
    try:
        kb.dispatch_once(conn, dry_run=True, spawn_fn=lambda *a, **k: None)
    finally:
        conn.close()
    assert captured_ticks
    assert all(kw["dry_run"] is True for kw in captured_ticks)


def test_misbehaving_subscriber_does_not_break_dispatcher(kanban_home, monkeypatch):
    """A hook callback that raises must not break the dispatch tick."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("subscriber exploded")

    mgr._hooks.setdefault("kanban_dispatch_tick", []).append(_boom)
    try:
        conn = kb.connect()
        try:
            # Idle tick — the dispatcher must return a DispatchResult cleanly
            # despite the raising subscriber.
            result = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
            assert isinstance(result, kb.DispatchResult)
        finally:
            conn.close()
    finally:
        mgr._hooks = saved
