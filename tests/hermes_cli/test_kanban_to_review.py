"""Behavior contract for the Hub-direct running -> review transition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_to_review_task_ends_owned_run_and_preserves_pr_handoff(kanban_home):
    """A Hub-owned running task atomically becomes a review handoff."""
    pr_url = "https://github.com/acme/widgets/pull/42"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Ship widget", assignee="code-supervisor")
        claimed = kb.claim_task(conn, task_id, claimer="hub-direct")
        assert claimed is not None
        run_id = claimed.current_run_id
        kb._set_worker_pid(conn, task_id, 4242)

        reviewed = kb.to_review_task(
            conn,
            task_id,
            pr_url=pr_url,
            claimer="hub-direct",
        )

        assert reviewed is not None
        assert reviewed.status == "review"
        assert reviewed.result == pr_url
        assert reviewed.claim_lock is None
        assert reviewed.claim_expires is None
        assert reviewed.current_run_id is None
        assert reviewed.worker_pid is None
        run = conn.execute(
            "SELECT status, outcome, summary, ended_at, claim_lock, worker_pid "
            "FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert run["status"] == "review"
        assert run["outcome"] == "review"
        assert run["summary"] == pr_url
        assert run["ended_at"] is not None
        assert run["claim_lock"] is None
        assert run["worker_pid"] is None
        event = conn.execute(
            "SELECT kind, payload, run_id FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert event["kind"] == "reviewed"
        assert json.loads(event["payload"]) == {"pr_url": pr_url}
        assert event["run_id"] == run_id


def test_to_review_task_rejects_a_different_claimer_without_mutating(kanban_home):
    """Only the worker that owns a running claim may hand it to review."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Ship widget", assignee="code-supervisor")
        claimed = kb.claim_task(conn, task_id, claimer="hub-direct")
        assert claimed is not None
        run_id = claimed.current_run_id
        events_before = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0]

        assert kb.to_review_task(
            conn,
            task_id,
            pr_url="https://github.com/acme/widgets/pull/42",
            claimer="other-dispatcher",
        ) is None

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.claim_lock == "hub-direct"
        assert task.result is None
        run = conn.execute(
            "SELECT status, outcome, ended_at, claim_lock FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert tuple(run) == ("running", None, None, "hub-direct")
        events_after = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        assert events_after == events_before


def test_to_review_task_replays_only_the_same_pr_handoff(kanban_home):
    """A retry is safe, but cannot replace a review task's PR evidence."""
    first_pr = "https://github.com/acme/widgets/pull/42"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Ship widget", assignee="code-supervisor")
        assert kb.claim_task(conn, task_id, claimer="hub-direct") is not None
        assert kb.to_review_task(conn, task_id, pr_url=first_pr, claimer="hub-direct") is not None
        events_before = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        run_before = conn.execute(
            "SELECT status, outcome, summary, ended_at FROM task_runs WHERE task_id = ?",
            (task_id,),
        ).fetchone()

        replay = kb.to_review_task(conn, task_id, pr_url=first_pr, claimer="hub-direct")
        conflict = kb.to_review_task(
            conn,
            task_id,
            pr_url="https://github.com/acme/widgets/pull/99",
            claimer="hub-direct",
        )

        assert replay is not None
        assert replay.status == "review"
        assert conflict is None
        assert kb.get_task(conn, task_id).result == first_pr
        events_after = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        run_after = conn.execute(
            "SELECT status, outcome, summary, ended_at FROM task_runs WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        assert events_after == events_before
        assert tuple(run_after) == tuple(run_before)
