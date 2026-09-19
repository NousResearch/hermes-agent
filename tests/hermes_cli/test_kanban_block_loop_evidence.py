"""Tests for block-loop self-heal evidence records.

Regression for #111112: when ``block_task`` trips ``block_loop_detected``, it
must also append a ``block_loop_evidence`` event via ``_append_event`` (no new
table) carrying ``task_id``, ``block_sequence`` (the block kinds + timestamps
that led to the trip), a nullable ``resolution``, and ``recurrence_count`` —
how many prior trips on this task already match this trip's sequence shape.
Triage routing and the watcher ping are untouched; this only augments the
``block_loop_detected`` event that already fires them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


def _trip_loop(conn, tid, kind="capability"):
    """Block/unblock/re-block with the same ``kind`` until block_loop_detected fires."""
    kb.block_task(conn, tid, reason="x", kind=kind)
    kb.unblock_task(conn, tid)
    _make_running_again(conn, tid)
    kb.block_task(conn, tid, reason="x", kind=kind)


def _evidence_events(conn, tid):
    return [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_evidence"]


def test_block_loop_evidence_recorded_with_sequence_and_null_resolution(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        _trip_loop(conn, tid, kind="capability")

        evidence = _evidence_events(conn, tid)
        assert len(evidence) == 1, "block_loop_detected must write exactly one evidence event"
        payload = evidence[0].payload
        assert payload["task_id"] == tid
        assert payload["resolution"] is None
        assert [item["kind"] for item in payload["block_sequence"]] == ["capability", "capability"]
        assert all(isinstance(item["at"], int) for item in payload["block_sequence"])
        assert payload["recurrence_count"] == 0

        # Routing/ping are untouched: the trip still lands in triage.
        assert kb.get_task(conn, tid).status == "triage"


def test_block_loop_evidence_recurrence_count_matches_prior_same_shape_trips(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        _trip_loop(conn, tid, kind="capability")
        assert _evidence_events(conn, tid)[-1].payload["recurrence_count"] == 0

        # A human resolves the first trip and the task restarts clean; the same
        # capability/capability failure pattern then recurs.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='ready', block_kind=NULL, block_recurrences=0 WHERE id=?",
                (tid,),
            )
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        _trip_loop(conn, tid, kind="capability")

        evidence = _evidence_events(conn, tid)
        assert len(evidence) == 2
        second = evidence[-1].payload
        assert [item["kind"] for item in second["block_sequence"]] == ["capability", "capability"]
        assert second["recurrence_count"] == 1, "must count the earlier identical-shape trip"
