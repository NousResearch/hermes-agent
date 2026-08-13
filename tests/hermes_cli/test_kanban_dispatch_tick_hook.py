"""Tests for the ``on_kanban_dispatch_tick`` observer hook.

Re-port of PR #56066 per the #64231 batch disposition: renamed to the
taxonomy form and fired by ``kanban_db.dispatch_once`` strictly AFTER the
board's single-writer dispatch lock has been released — the original fired
inside ``_dispatch_tick_lock``, so a slow subscriber could extend the
critical section and stall a sibling dispatcher. Verifies the post-lock
contract directly, the ``outcome`` classification (including idle and
contended ticks), ``dry_run`` propagation, and that a misbehaving
subscriber never breaks the dispatcher.
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
    mgr._hooks.setdefault("on_kanban_dispatch_tick", []).append(
        lambda **kw: events.append(kw)
    )
    try:
        yield events
    finally:
        mgr._hooks = saved


def test_dispatch_tick_hook_is_registered_as_valid():
    assert "on_kanban_dispatch_tick" in VALID_HOOKS


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
    assert "board" in kw
    assert "profile_name" in kw


def test_active_tick_fires_hook_with_outcome_ok(
    kanban_home, all_assignees_spawnable, captured_ticks,
):
    """A tick that spawns a worker fires the hook with outcome='ok'."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    finally:
        conn.close()
    ok_events = [kw for kw in captured_ticks if kw["outcome"] == "ok"]
    assert ok_events, (
        f"expected an ok tick, got {[kw['outcome'] for kw in captured_ticks]}"
    )
    result = ok_events[-1]["result"]
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


def test_contended_tick_fires_with_outcome_skipped_locked(
    kanban_home, captured_ticks,
):
    """A losing dispatcher still reports its (empty) tick to observers."""
    db_path = kb.kanban_db_path()
    conn = kb.connect()
    try:
        with kb._dispatch_tick_lock(db_path) as held:
            assert held is True
            result = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
        assert result.skipped_locked is True
    finally:
        conn.close()
    assert len(captured_ticks) == 1
    assert captured_ticks[0]["outcome"] == "skipped_locked"


def test_tick_hook_fires_after_dispatch_lock_released(kanban_home):
    """The #56066 sweeper finding, as a contract: subscribers run OUTSIDE
    the single-writer critical section. From the callback, acquiring the
    board's dispatch lock must succeed — flock conflicts across file
    descriptors within one process, so this fails if the hook still fires
    while ``dispatch_once`` holds the lock."""
    db_path = kb.kanban_db_path()
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    acquired: list[bool] = []

    def _probe_lock(**kw):
        with kb._dispatch_tick_lock(db_path) as held:
            acquired.append(held)

    mgr._hooks.setdefault("on_kanban_dispatch_tick", []).append(_probe_lock)
    try:
        conn = kb.connect()
        try:
            kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
        finally:
            conn.close()
    finally:
        mgr._hooks = saved
    assert acquired == [True]


def test_misbehaving_subscriber_does_not_break_dispatcher(kanban_home):
    """A hook callback that raises must not break the dispatch tick."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("subscriber exploded")

    mgr._hooks.setdefault("on_kanban_dispatch_tick", []).append(_boom)
    try:
        conn = kb.connect()
        try:
            result = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
            assert isinstance(result, kb.DispatchResult)
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_no_subscriber_short_circuits_tick_hook(kanban_home, monkeypatch):
    """With nothing registered, the tick observer is never invoked."""
    from hermes_cli import lifecycle

    invoked: list[str] = []
    real_invoke = lifecycle.invoke_hook

    def _spy(hook_name, **kw):
        invoked.append(hook_name)
        return real_invoke(hook_name, **kw)

    monkeypatch.setattr(lifecycle, "invoke_hook", _spy)
    conn = kb.connect()
    try:
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
    finally:
        conn.close()
    assert "on_kanban_dispatch_tick" not in invoked
