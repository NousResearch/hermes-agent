"""E2E regression for the global kanban worker-concurrency cap (#33488).

The standalone daemon forwards ``kanban.max_in_progress`` into
``dispatch_once``. These tests pin that the REAL ``dispatch_once`` (not a
mocked stub) both accepts the kwarg AND enforces it as a live concurrency
ceiling — i.e. it counts tasks already ``running`` plus this tick's spawns
against the cap, so a backlog can never burst-spawn an unbounded burst.

This addresses the reviewgate on the propagation PR: the existing
propagation test injects a fake ``dispatch_once`` that only captures
kwargs, which can't distinguish \"enforced\" from \"accepted-but-ignored\".
These tests exercise the real dispatcher against a real SQLite board.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh HERMES_HOME with a kanban DB and an alpha profile."""
    test_home = tempfile.mkdtemp(prefix="kanban_worker_limit_test_")
    for prof in ("alpha", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def test_real_dispatch_once_enforces_global_worker_limit(isolated_kanban_home):
    """With max_in_progress=2, one tick must dispatch at most 2 tasks —
    the rest stay ready for a later tick, never exceeding the ceiling."""
    kb = isolated_kanban_home
    total = 5
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(total):
            kb.create_task(conn, title=f"t{i}", assignee="alpha")

    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=False,
            max_in_progress=2,
        )

    assert len(res.spawned) == 2
    # The remaining tasks were NOT spawned: the dispatcher broke out of the
    # ready loop at the cap rather than bursting past it.
    with kb.connect_closing() as conn:
        still_ready = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'ready'"
        ).fetchone()[0]
        running = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()[0]
    assert running == 2
    assert still_ready == total - 2


def test_limit_frees_headroom_after_running_tasks_complete(isolated_kanban_home):
    """The cap is a LIVE ceiling: once a running task completes, the next
    tick dispatches one more, so total in-flight never exceeds max_in_progress."""
    kb = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        ids = [
            kb.create_task(conn, title=f"t{i}", assignee="alpha")
            for i in range(3)
        ]

    with kb.connect_closing() as conn:
        res1 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False, max_in_progress=2,
        )
    assert len(res1.spawned) == 2

    # Complete one running task -> headroom opens back up.
    done_id = res1.spawned[0][0]
    with kb.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?",
                (done_id,),
            )

    with kb.connect_closing() as conn:
        res2 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False, max_in_progress=2,
        )
    assert len(res2.spawned) == 1

    free = 0
    with kb.connect_closing() as conn:
        free = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'ready'"
        ).fetchone()[0]
    assert free == 0
    assert done_id not in [s[0] for s in res2.spawned]