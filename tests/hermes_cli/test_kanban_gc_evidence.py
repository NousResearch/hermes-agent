"""Every ``kanban gc`` event sweep must leave first-party evidence in the same
event stream it prunes (#133284): a board-level ``gc_pruned`` event carrying
the cutoff, deleted count, pruned id window, and a sha256 fingerprint of the
deleted ids, so an auditor can tell a recorded retention prune from an
unexplained deletion."""
import argparse
import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_ops


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    return tmp_path


def _done_task_with_old_event(conn):
    tid = kb.create_task(conn, title="finished")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))
        conn.execute("UPDATE task_events SET created_at=0 WHERE task_id=?", (tid,))
    return tid


def _gc_evidence_rows(conn):
    return conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id=? AND kind='gc_pruned' "
        "ORDER BY id ASC",
        (kb.BOARD_TASK_ID,),
    ).fetchall()


def test_gc_events_prune_leaves_matching_evidence(board):
    with kbc.connect_closing() as conn:
        tid = _done_task_with_old_event(conn)
        max_id_before = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM task_events"
        ).fetchone()[0]
        # Auditor's baseline: the id list before the sweep (what an external
        # export / hash chain would hold). The fingerprint must reconcile
        # against it, not against the post-DELETE visible space.
        pre_ids = [r[0] for r in conn.execute(
            "SELECT id FROM task_events ORDER BY id ASC"
        ).fetchall()]
        deleted = kb.gc_events(conn, older_than_seconds=1)
        assert deleted > 0
        assert deleted == 1, "the seed task's lone 'created' event is its only event and is prunable"
        rows = _gc_evidence_rows(conn)
        assert len(rows) == 1, "prune without a gc_pruned evidence row is an unexplained deletion"
        payload = json.loads(rows[0]["payload"])
        assert payload["deleted_count"] == deleted
        assert payload["retention_seconds"] == 1
        assert payload["run_at"] >= payload["cutoff"] > 0
        # Deleted ids must fall inside the recorded window, the window must
        # predate the evidence row itself, and the fingerprint must match an
        # independent recomputation over the surviving-visible id space.
        assert payload["min_event_id"] is not None
        assert 1 <= payload["min_event_id"] <= payload["max_event_id"] < rows[0]["id"]
        assert payload["max_event_id"] <= max_id_before
        expected = hashlib.sha256(
            ",".join(str(i) for i in pre_ids).encode("utf-8")
        ).hexdigest()
        assert payload["deleted_ids_sha256"] == expected
        # A second sweep appends its own evidence; earlier evidence survives gc
        # (board sentinel is not a done/archived task, so it is never pruned).
        kb.gc_events(conn, older_than_seconds=1)
        assert len(_gc_evidence_rows(conn)) == 2


def test_gc_events_zero_row_sweep_still_records_evidence(board):
    with kbc.connect_closing() as conn:
        kb.create_task(conn, title="active, nothing to prune")
        assert kb.gc_events(conn, older_than_seconds=1) == 0
        rows = _gc_evidence_rows(conn)
        assert len(rows) == 1, "a no-op sweep must still leave a dated gc record"
        payload = json.loads(rows[0]["payload"])
        assert payload["deleted_count"] == 0
        assert payload["min_event_id"] is None and payload["max_event_id"] is None


def test_cmd_gc_retention_leaves_evidence(board):
    """The real ``kanban gc`` command path records the evidence too."""
    args = argparse.Namespace(event_retention_days=30, log_retention_days=30)
    with kbc.connect_closing() as conn:
        _done_task_with_old_event(conn)
    assert kanban_ops._cmd_gc(args) == 0
    with kbc.connect_closing() as conn:
        rows = _gc_evidence_rows(conn)
        assert len(rows) == 1
        assert json.loads(rows[0]["payload"])["deleted_count"] > 0
