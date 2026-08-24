"""Tests for the ``on_kanban_worker_*`` observer hooks (RFC #58548).

Verifies the worker-lifecycle observers accepted in the #64231 batch
disposition: ``on_kanban_worker_spawned`` fires after ``spawn_fn`` returns
and the worker PID is durably persisted, ``on_kanban_worker_exited`` is
tick-derived from ``detect_crashed_workers`` and fires after every reclaim
transaction has committed, and ``on_kanban_worker_stale_claim`` fires when
``release_stale_claims`` reclaims a TTL-expired claim. All three are
observer-only, short-circuit on ``has_hook``, and can never break the
dispatcher.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager

WORKER_HOOKS = (
    "on_kanban_worker_spawned",
    "on_kanban_worker_exited",
    "on_kanban_worker_stale_claim",
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Crash detection acts immediately in these tests (no launch grace).
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_hooks(monkeypatch):
    """Register capturing callbacks for the worker-lifecycle hooks."""
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in WORKER_HOOKS:
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved

def test_dispatch_spawn_fires_worker_spawned(
    kanban_home, all_assignees_spawnable, captured_hooks,
):
    """A dispatched spawn fires the hook AFTER the PID is durably persisted."""
    pid_at_fire_time: list = []

    def _read_pid(**kw):
        # Read through a FRESH connection: proves the PID write was
        # committed before the hook fired (the RFC timing contract).
        c2 = sqlite3.connect(kb.kanban_db_path())
        try:
            row = c2.execute(
                "SELECT worker_pid FROM tasks WHERE id = ?", (kw["task_id"],)
            ).fetchone()
            pid_at_fire_time.append(row[0] if row else None)
        finally:
            c2.close()

    mgr = get_plugin_manager()
    mgr._hooks.setdefault("on_kanban_worker_spawned", []).append(_read_pid)

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="alice")
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
        assert any(row[0] == tid for row in result.spawned)
    finally:
        conn.close()

    fired = [e for e in captured_hooks if e[0] == "on_kanban_worker_spawned"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "alice"
    assert kw["worker_pid"] == 4242
    assert kw["workspace_path"]
    assert kw["run_id"] is not None
    assert "profile_name" in kw
    assert "board" in kw
    assert pid_at_fire_time == [4242]

def test_crash_reclaim_fires_worker_exited(kanban_home, captured_hooks, monkeypatch):
    """A dead-PID reclaim fires the exit observer with the exit facts."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, tid)
        kbd._set_worker_pid(conn, tid, 98765)
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        assert kbd.detect_crashed_workers(conn) == [tid]
    finally:
        conn.close()

    fired = [e for e in captured_hooks if e[0] == "on_kanban_worker_exited"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert kw["worker_pid"] == 98765
    assert kw["exit_kind"] == "unknown"
    assert kw["exit_code"] is None
    assert kw["outcome"] == "crashed"
    assert kw["retry_status"] == "ready"
    assert kw["run_id"] is not None
    assert "profile_name" in kw
    assert "board" in kw

def test_stale_claim_reclaim_fires_hook(kanban_home, captured_hooks):
    """A TTL-expired reclaim fires the stale-claim observer post-commit."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, tid)
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 100, tid),
        )
        conn.commit()
        assert kb.release_stale_claims(conn) == 1
    finally:
        conn.close()

    fired = [e for e in captured_hooks if e[0] == "on_kanban_worker_stale_claim"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert kw["worker_pid"] is None
    assert kw["heartbeat_stale"] is False
    assert kw["retry_status"] == "ready"
    assert kw["run_id"] is not None
    assert "profile_name" in kw
    assert "board" in kw

def test_raising_callbacks_never_break_worker_lifecycle(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Raising subscribers must not break spawn, crash reclaim, or stale reclaim."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    for hook in WORKER_HOOKS:
        mgr._hooks.setdefault(hook, []).append(_boom)
    try:
        conn = kbc.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="alice")
            result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 111)
            assert any(row[0] == tid for row in result.spawned)

            monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
            assert kbd.detect_crashed_workers(conn) == [tid]

            kb.claim_task(conn, tid)
            conn.execute(
                "UPDATE tasks SET claim_expires = ?, worker_pid = NULL "
                "WHERE id = ?",
                (int(time.time()) - 100, tid),
            )
            conn.commit()
            assert kb.release_stale_claims(conn) == 1
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_no_subscriber_short_circuits_worker_hooks(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """With nothing registered, the new observers are never invoked at all."""
    from hermes_cli import lifecycle

    invoked: list[str] = []
    real_invoke = lifecycle.invoke_hook

    def _spy(hook_name, **kw):
        invoked.append(hook_name)
        return real_invoke(hook_name, **kw)

    monkeypatch.setattr(lifecycle, "invoke_hook", _spy)
    conn = kbc.connect()
    try:
        kb.create_task(conn, title="t", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 222)
    finally:
        conn.close()
    assert "on_kanban_worker_spawned" not in invoked
    # The shipped claimed hook has no short-circuit and still fires.
    assert "kanban_task_claimed" in invoked
