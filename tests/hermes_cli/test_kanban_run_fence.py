"""A delayed failure/PID write from a stale worker cannot mutate a successor run."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    out = [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else None)
        for r in rows
    ]
    if kind is not None:
        out = [e for e in out if e[0] == kind]
    return out


def test_budget_failure_owned_run_is_recorded_once(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="owned budget failure", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.current_run_id is not None
        run_id = claimed.current_run_id

        assert not kbd._record_task_failure(
            conn, task_id, "budget", outcome="timed_out", failure_limit=2,
            release_claim=True, end_run=True, expected_run_id=run_id,
        )
        first = conn.execute(
            "SELECT status, current_run_id, consecutive_failures FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(first) == ("ready", None, 1)
        events_before = len(_events(conn, task_id))

        assert not kbd._record_task_failure(
            conn, task_id, "duplicate", outcome="timed_out", failure_limit=2,
            release_claim=True, end_run=True, expected_run_id=run_id,
        )
        second = conn.execute(
            "SELECT status, current_run_id, consecutive_failures FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(second) == tuple(first)
        assert len(_events(conn, task_id)) == events_before


def test_budget_failure_stale_run_cannot_mutate_successor(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="successor fence", assignee="worker")
        first = kb.claim_task(conn, task_id)
        assert first is not None and first.current_run_id is not None
        old_run_id = first.current_run_id
        assert not kbd._record_task_failure(
            conn, task_id, "first failure", outcome="timed_out", failure_limit=3,
            release_claim=True, end_run=True, expected_run_id=old_run_id,
        )
        successor = kb.claim_task(conn, task_id)
        assert successor is not None and successor.current_run_id != old_run_id
        before = conn.execute(
            "SELECT status, current_run_id, claim_lock, consecutive_failures, last_failure_error "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        events_before = len(_events(conn, task_id))

        assert not kbd._record_task_failure(
            conn, task_id, "late old finalizer", outcome="timed_out", failure_limit=1,
            release_claim=True, end_run=True, expected_run_id=old_run_id,
        )
        after = conn.execute(
            "SELECT status, current_run_id, claim_lock, consecutive_failures, last_failure_error "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        assert tuple(after) == tuple(before)
        assert len(_events(conn, task_id)) == events_before


def test_delayed_worker_pid_is_fenced_from_successor(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="pid fence", assignee="worker")
        first = kb.claim_task(conn, task_id)
        assert first is not None and first.current_run_id is not None
        old_run_id = first.current_run_id
        assert not kbd._record_task_failure(
            conn, task_id, "first failed", outcome="spawn_failed", failure_limit=3,
            release_claim=True, end_run=True, expected_run_id=old_run_id,
        )
        successor = kb.claim_task(conn, task_id)
        assert successor is not None and successor.current_run_id != old_run_id
        before = conn.execute(
            "SELECT current_run_id, worker_pid, worker_started_at FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        events_before = len(_events(conn, task_id, kind="spawned"))

        assert not kbd._set_worker_pid(
            conn, task_id, os.getpid(), expected_run_id=old_run_id,
        )
        after = conn.execute(
            "SELECT current_run_id, worker_pid, worker_started_at FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(after) == tuple(before)
        assert len(_events(conn, task_id, kind="spawned")) == events_before

