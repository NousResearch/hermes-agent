"""Dispatch gates and the claim must agree on the assignee (#147534).

The dispatcher gates (``kanban.dispatch_profiles`` allowlist, profile existence,
per-profile cap) run against the assignee read at tick start. A reassignment
landing between those gates and the claim must not spawn a worker as the new,
never-vetted profile: the claim loses and the next tick gates the new assignee.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text(
        "kanban:\n  dispatch_profiles: [worker-a]\n", encoding="utf-8",
    )
    kb.init_db()
    return home


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_reassign_inside_claim_window_does_not_spawn_unvetted_profile(
    kanban_home, all_assignees_spawnable, monkeypatch, lane,
):
    spawned_as: list[str] = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="card", assignee="worker-a")
        if lane == "review":
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        real_guard = kbd.check_respawn_guard

        def guard_then_reassign(c, task_id, **kw):
            # Last gate before the claim: an operator reassigns the card right now.
            reason = real_guard(c, task_id, **kw)
            with kbc.connect() as other:
                kb.assign_task(other, task_id, "worker-b")
            return reason

        monkeypatch.setattr(kbd, "check_respawn_guard", guard_then_reassign)
        res = kbd.dispatch_once(
            conn, spawn_fn=lambda task, _ws: spawned_as.append(task.assignee) or 4242,
        )
        task = kb.get_task(conn, tid)
        monkeypatch.setattr(kbd, "check_respawn_guard", real_guard)
        nxt = kbd.dispatch_once(conn, spawn_fn=lambda task, _ws: spawned_as.append(task.assignee) or 4242)

    assert task is not None
    assert spawned_as == []
    assert res.spawned == []
    assert (task.status, task.assignee, task.claim_lock) == (lane, "worker-b", None)
    # The next tick gates the card against its real assignee: not on the allowlist.
    assert nxt.spawned == []
    assert nxt.skipped_nonspawnable == [tid]
