"""Tamper detection for kanban.db writes that bypass the kernel (#110080).

The completion gate lives in the tool/CLI path only; SQLite accepts writes from
any process, so a refused worker can just run
``UPDATE tasks SET status='done' ...`` itself. These tests pin the read-side
detection: the kernel's own flow stays silent, while a raw status flip — or a
raw event INSERT/UPDATE/DELETE — surfaces as a diagnostic.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd

TAMPER_KINDS = {"out_of_band_transition", "event_chain_tampered"}


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kbc.connect()
    try:
        yield c
    finally:
        c.close()


def _events(conn, task_id):
    return kb.list_events(conn, task_id)


def _diags(conn, task_id):
    return kd.compute_task_diagnostics(
        kb.get_task(conn, task_id), _events(conn, task_id), kb.list_runs(conn, task_id),
    )


def _tamper_kinds(conn, task_id) -> set[str]:
    return {d.kind for d in _diags(conn, task_id)} & TAMPER_KINDS


def _claimed_task(conn) -> str:
    tid = kb.create_task(conn, title="ship it", assignee="worker")
    assert kb.claim_task(conn, tid, claimer="worker:1") is not None
    return tid


def _direct_done(conn, tid: str) -> None:
    """The incident: raw SQL once the completion gate refused the transition."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), tid),
        )


def test_kernel_completion_reports_nothing(conn):
    tid = _claimed_task(conn)
    assert kb.complete_task(conn, tid, result="shipped", summary="shipped")
    assert kb.get_task(conn, tid).status == "done"
    assert kb.verify_event_chain(tid, _events(conn, tid)) == []
    assert _tamper_kinds(conn, tid) == set()


def test_completion_gate_refusal_then_raw_update_is_flagged(conn):
    parent = kb.create_task(conn, title="parent", assignee="worker")
    assert kb.claim_task(conn, parent, claimer="worker:0") is not None
    tid = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
    assert kb.claim_task(conn, tid, claimer="worker:1") is None  # parent not done
    assert kb.complete_task(conn, tid, result="done!") is False  # the gate refuses
    _direct_done(conn, tid)

    flagged = [d for d in _diags(conn, tid) if d.kind == "out_of_band_transition"]
    assert len(flagged) == 1
    assert flagged[0].severity == "critical"
    # 3 events: created, dependency_wait (create_task records the open parent), claim_rejected.
    assert flagged[0].data == {"status": "done", "expected_event": "completed", "event_count": 3}
    # The tasks row was forged; the event log itself is untouched.
    assert kb.verify_event_chain(tid, _events(conn, tid)) == []

    # Same verdict on the sqlite3.Row objects the dashboard/CLI fleet path passes.
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    rows = list(conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall())
    assert any(d.kind == "out_of_band_transition" for d in kd.compute_task_diagnostics(row, rows, []))


def test_forged_completed_event_is_caught_by_the_chain(conn):
    """Faking the event too (raw INSERT) satisfies the status/event pairing and
    is caught by the hash chain instead."""
    tid = _claimed_task(conn)
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'completed', NULL, ?)",
            (tid, int(time.time())),
        )
    _direct_done(conn, tid)

    assert "out_of_band_transition" not in _tamper_kinds(conn, tid)
    assert [f["kind"] for f in kb.verify_event_chain(tid, _events(conn, tid))] == ["unsigned_event"]
    assert "event_chain_tampered" in _tamper_kinds(conn, tid)


def test_edited_event_payload_is_flagged(conn):
    tid = _claimed_task(conn)
    assert kb.complete_task(conn, tid, result="ok")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET payload = '{\"forged\": true}' WHERE id = ("
            " SELECT id FROM task_events WHERE task_id = ? AND kind = 'completed'"
            " ORDER BY id DESC LIMIT 1)",
            (tid,),
        )

    assert [f["kind"] for f in kb.verify_event_chain(tid, _events(conn, tid))] == ["hash_mismatch"]
    assert "event_chain_tampered" in _tamper_kinds(conn, tid)


def test_deleted_event_is_flagged(conn):
    tid = kb.create_task(conn, title="audit log", assignee="worker")
    for kind in ("one", "two", "three"):
        kb._append_event(conn, tid, kind)
    with kb.write_txn(conn):
        conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'two'", (tid,))

    assert [f["kind"] for f in kb.verify_event_chain(tid, _events(conn, tid))] == ["chain_broken"]


def test_legacy_and_gc_pruned_rows_are_not_false_positives(conn):
    """Rows written before the chain (NULL hash) and a gc-pruned prefix must
    stay silent — detection may not fire on boards that predate the feature."""
    tid = _claimed_task(conn)
    assert kb.complete_task(conn, tid, result="ok")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET prev_hash = NULL, event_hash = NULL WHERE task_id = ?", (tid,),
        )
    assert kb.verify_event_chain(tid, _events(conn, tid)) == []
    # First chained row after the legacy block: still no findings, no diagnostic.
    kb.add_comment(conn, tid, "human", "LGTM")
    assert kb.verify_event_chain(tid, _events(conn, tid)) == []
    assert _tamper_kinds(conn, tid) == set()

    # gc prunes the prefix but keeps the terminal event (its audit anchor).
    assert kb.gc_events(conn, older_than_seconds=1) >= 0
    assert kb.get_task(conn, tid).status == "done"
    assert kb.verify_event_chain(tid, _events(conn, tid)) == []
    assert _tamper_kinds(conn, tid) == set()


def test_existing_board_gains_chain_columns(kanban_home):
    db_path = kb.kanban_db_path()
    with kbc.connect_closing() as c:
        c.execute("ALTER TABLE task_events DROP COLUMN event_hash")
        c.execute("ALTER TABLE task_events DROP COLUMN prev_hash")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kbc.connect_closing() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(task_events)")}
        assert {"prev_hash", "event_hash"} <= cols

    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="post-migration", assignee="worker")
        assert any(e.event_hash for e in _events(c, tid))
