"""Handoff metadata must survive completion, review, and a lost run write.

Regression tests for the kanban handoff losing its ``task_runs.metadata``:

* the implementer hands off with ``request_review(metadata=...)``, the reviewer
  then closes the card with its own run — the handoff must still be readable
  from the card (``latest_handoff_run``) and from the child worker context
  (``## Parent task results``);
* a completion whose closing run was already ended (reclaim / timeout /
  restored DB) used to drop ``summary``+``metadata`` silently while
  ``kanban_complete`` answered ``ok: true`` — the write is reported honestly
  and the payload is synthesized onto a run instead of vanishing;
* when the payload really cannot be persisted, the tool says so and backfills
  what it can instead of reporting a false success.

Assertions are on the actual handoff values (branch, PR text, session id) read
back from the rows, not on "metadata is a non-empty object" — a completion
stamps ``worker_session_id`` onto every run, so a non-empty object proves
nothing.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

HANDOFF = {
    "report": "ready",
    "branch": "card-test-handoff-metadata",
    "changed_files": ["hermes_cli/kanban_db.py", "tools/kanban_tools.py"],
    "tests_run": 7,
    "pr_text": "kanban: handoff metadata survives review + closed run",
    "session_id": "local-abc123",
}


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Isolated board: its own kanban.db, no production paths touched."""
    home = tmp_path / "kanban-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_SESSION_ID", "local-test-session")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    conn = kbc.connect(home / "kanban.db")
    try:
        yield _Board(conn, monkeypatch)
    finally:
        conn.close()


class _Board:
    def __init__(self, conn, monkeypatch):
        self.conn = conn
        self.mp = monkeypatch

    # --- setup helpers -------------------------------------------------
    def claimed_task(self, title: str, assignee: str = "developer") -> tuple[str, int]:
        tid = kb.create_task(self.conn, title=title, body="handoff", assignee=assignee,
                             workspace_kind="scratch")
        kb.claim_task(self.conn, tid, claimer="dispatcher-claim")
        return tid, kb.get_task(self.conn, tid).current_run_id

    def child_of(self, parent_id: str) -> str:
        return kb.create_task(self.conn, title="child", body="child", assignee="worker",
                              workspace_kind="scratch", parents=[parent_id])

    def worker_context(self, task_id: str) -> str:
        return kb.build_worker_context(self.conn, task_id)

    # --- acting as the card's worker -----------------------------------
    def as_worker(self, tid: str, run_id: int | None) -> None:
        self.mp.setenv("HERMES_KANBAN_TASK", tid)
        if run_id is None:
            self.mp.delenv("HERMES_KANBAN_RUN_ID", raising=False)
        else:
            self.mp.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    def complete(self, tid: str, run_id: int | None, **kwargs) -> dict:
        """Call the tool handler the model actually calls; returns the payload."""
        from tools import kanban_tools as kt

        self.as_worker(tid, run_id)
        try:
            return json.loads(kt._handle_complete(kwargs))
        finally:
            self.mp.delenv("HERMES_KANBAN_TASK", raising=False)
            self.mp.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    # --- reading back ---------------------------------------------------
    def run(self, run_id: int):
        return self.conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()

    def runs_with_handoff(self, tid: str, key: str = "branch", value: str = HANDOFF["branch"]):
        found = []
        for run in kb.list_runs(self.conn, tid):
            if isinstance(run.metadata, dict) and run.metadata.get(key) == value:
                found.append(run)
        return found


def test_direct_completion_persists_handoff(board):
    tid, run_id = board.claimed_task("direct complete")
    child = board.child_of(tid)

    resp = board.complete(tid, run_id, summary="handoff: done", metadata=dict(HANDOFF))

    assert resp["ok"] is True
    closing = kb.latest_run(board.conn, tid)
    assert kb.get_task(board.conn, tid).status == "done"
    assert isinstance(closing.metadata, dict)
    for key, value in HANDOFF.items():
        assert closing.metadata[key] == value
    assert closing.summary == "handoff: done"
    ctx = board.worker_context(child)
    assert HANDOFF["branch"] in ctx
    assert HANDOFF["pr_text"] in ctx


def test_handoff_survives_the_reviewers_closing_run(board):
    tid, impl_run = board.claimed_task("review flow")
    child = board.child_of(tid)
    board.as_worker(tid, impl_run)
    assert kb.request_review(board.conn, tid, summary="готово к ревью", metadata=dict(HANDOFF),
                             reviewer="reviewer", expected_run_id=impl_run) is True
    board.mp.delenv("HERMES_KANBAN_TASK", raising=False)
    board.mp.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    kb.claim_review_task(board.conn, tid, claimer="reviewer-claim")
    reviewer_run = kb.get_task(board.conn, tid).current_run_id

    resp = board.complete(tid, reviewer_run, summary="approved: выглядит верно")

    assert resp["ok"] is True
    completed = [r for r in kb.list_runs(board.conn, tid) if r.outcome == "completed"]
    closing = completed[-1]
    # The failure mode under test: the closing run is the reviewer's and carries
    # at most the stamped worker session id, not the implementer's handoff.
    assert closing.summary.startswith("approved")
    assert HANDOFF["branch"] not in (closing.metadata or {})

    handoff = kb.latest_handoff_run(board.conn, tid)
    assert handoff is not None
    assert {k: handoff.metadata[k] for k in HANDOFF} == HANDOFF
    ctx = board.worker_context(child)
    assert HANDOFF["branch"] in ctx
    assert HANDOFF["pr_text"] in ctx


def test_completion_after_an_already_closed_run_keeps_the_handoff(board):
    tid, run_id = board.claimed_task("run closed by a sweep")
    with kb.write_txn(board.conn):
        # A sweep already ended the run; the card's pointer still names it
        # (restored DB / manual repair / interleaving writer).
        kb._end_run(board.conn, tid, outcome="timed_out", status="timed_out",
                    error="worker exceeded max runtime")
        board.conn.execute(
            "UPDATE tasks SET status = 'running', current_run_id = ? WHERE id = ?", (run_id, tid))

    resp = board.complete(tid, run_id, summary="handoff после таймаута",
                         metadata=dict(HANDOFF))

    landed = board.runs_with_handoff(tid)
    assert landed, "handoff metadata disappeared when the closing run was already ended"
    assert {k: landed[0].metadata[k] for k in HANDOFF} == HANDOFF
    # The tool's answer must match what is actually stored.
    assert resp["ok"] is True
    assert resp["run_id"] == landed[0].id


def test_string_metadata_is_an_error_not_a_silent_loss(board):
    tid, run_id = board.claimed_task("metadata as a JSON string")

    resp = board.complete(tid, run_id, summary="handoff", metadata=json.dumps(HANDOFF))

    assert resp.get("ok") is not True
    assert "error" in resp
    assert kb.get_task(board.conn, tid).status != "done"


def test_end_run_reports_a_write_that_did_not_land(board):
    tid, run_id = board.claimed_task("double close")
    with kb.write_txn(board.conn):
        first = kb._end_run(board.conn, tid, outcome="crashed", status="crashed", summary="first")
        board.conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, tid))
        second = kb._end_run(board.conn, tid, outcome="completed", status="done",
                            summary="second", metadata=dict(HANDOFF))

    assert first == run_id
    assert second is None, "_end_run claimed a close the row never took"
    row = board.run(run_id)
    assert row["outcome"] == "crashed"
    assert not row["summary"] == "second"


def test_completion_with_unwritable_handoff_is_not_reported_ok(board, monkeypatch):
    tid, run_id = board.claimed_task("storage drops the payload")
    real_end = kb._end_run

    def _end_without_payload(conn, task_id, *, outcome, summary=None, error=None,
                             metadata=None, status=None):
        # Storage that closes the run but does not keep summary/metadata.
        return real_end(conn, task_id, outcome=outcome, summary=None, error=error,
                        metadata=None, status=status)

    monkeypatch.setattr(kb, "_end_run", _end_without_payload)
    monkeypatch.setattr(kb, "_synthesize_ended_run", lambda *a, **k: 0)
    monkeypatch.setattr(kb, "edit_task", lambda *a, **k: False)

    resp = board.complete(tid, run_id, summary="handoff на сбойном хранилище",
                         metadata=dict(HANDOFF))

    assert resp.get("ok") is not True, resp
    assert "metadata" in json.dumps(resp, ensure_ascii=False)
    assert not board.runs_with_handoff(tid)


def test_lost_handoff_is_backfilled_and_flagged(board, monkeypatch):
    tid, run_id = board.claimed_task("payload dropped on the closing run")
    real_end = kb._end_run

    def _end_dropping_payload(conn, task_id, *, outcome, summary=None, error=None,
                              metadata=None, status=None):
        # The run closes for real, but neither summary nor metadata reach the row.
        return real_end(conn, task_id, outcome=outcome, summary=None, error=error,
                        metadata=None, status=status)

    monkeypatch.setattr(kb, "_end_run", _end_dropping_payload)

    resp = board.complete(tid, run_id, summary="handoff для бэкфилла", metadata=dict(HANDOFF))

    assert resp["ok"] is True
    assert resp.get("handoff_repaired") is True
    landed = board.runs_with_handoff(tid)
    assert landed, "backfill did not restore the handoff"
    assert {k: landed[0].metadata[k] for k in HANDOFF} == HANDOFF
    assert "handoff для бэкфилла" in board.worker_context(board.child_of(tid))
