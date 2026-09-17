"""Kanban stale-claim fencing for run-scoped comments.

Regression for #99283 (remaining atom on current main): ``complete_task``/
``block_task``/``heartbeat_worker`` already fence a reclaimed zombie's writes
via ``expected_run_id`` CAS — verified on main by the spike behind this fix.
The hole this pins shut is ``add_comment``: it carried no run identity, so a
zombie worker's late comment landed on the thread, and
``build_worker_context`` injects that thread into the successor's system
prompt — context poisoning, not a harmless log line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _zombie_then_successor(conn):
    """Claim as an unkillable (foreign-host) worker, expire, reclaim, re-claim
    locally. Returns (task_id, zombie_run_id, successor_run_id)."""
    tid = kb.create_task(conn, title="fence", assignee="w")
    zombie = kb.claim_task(conn, tid, claimer="10.9.9.9:9:1")
    assert zombie is not None
    zombie_run = zombie.current_run_id
    conn.execute("UPDATE tasks SET claim_expires = 1 WHERE id = ?", (tid,))
    conn.commit()
    assert kb.release_stale_claims(conn, signal_fn=None) == 1
    successor = kb.claim_task(conn, tid, claimer=kb._claimer_id())
    assert successor is not None
    return tid, zombie_run, successor.current_run_id


def test_zombie_comment_is_fenced_off_successor_thread(conn):
    """A worker whose run was reclaimed must not leave a comment on the task
    the successor now owns: ``build_worker_context`` injects comments into the
    next worker's system prompt, so a zombie comment is context poisoning."""
    tid, zombie_run, _successor_run = _zombie_then_successor(conn)

    with pytest.raises(RuntimeError, match="no longer owns"):
        kb.add_comment(
            conn, tid, author="zombie", body="stale write",
            expected_run_id=zombie_run,
        )

    assert kb.list_comments(conn, tid) == []
    ctx = kb.build_worker_context(conn, tid)
    assert "stale write" not in ctx

    # The successor (and humans via the CLI) can still comment freely.
    kb.add_comment(conn, tid, author="human", body="legitimate note")
    assert len(kb.list_comments(conn, tid)) == 1
