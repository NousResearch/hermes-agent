from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_submit_task_for_review_closes_worker_run_and_preserves_handoff(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="review me", assignee="test-worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 1, last_failure_error = ? WHERE id = ?",
            ("prior implementation crash", task_id),
        )
        conn.commit()

        assert kb.submit_task_for_review(
            conn,
            task_id,
            summary="PR ready for human review",
            metadata={"pr_url": "https://example.test/pr/1", "head_sha": "abc123"},
            expected_run_id=run_id,
        ) is True

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "review"
        assert task.current_run_id is None
        assert task.claim_lock is None
        assert task.claim_expires is None
        assert task.worker_pid is None
        assert task.completed_at is None
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None

        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "submitted_for_review"
        assert run.status == "review"
        assert run.summary == "PR ready for human review"
        assert run.metadata == {
            "pr_url": "https://example.test/pr/1",
            "head_sha": "abc123",
        }

        events = kb.list_events(conn, task_id)
        assert events[-1].kind == "submitted_for_review"
        assert events[-1].run_id == run_id
    finally:
        conn.close()


def test_human_can_complete_task_from_review(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="approve me", assignee="test-worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.submit_task_for_review(
            conn,
            task_id,
            summary="ready",
            expected_run_id=claimed.current_run_id,
        )

        assert kb.complete_task(
            conn,
            task_id,
            summary="human review approved",
            metadata={"approved_by": "Nick"},
        ) is True

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.completed_at is not None
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "completed"
        assert run.summary == "human review approved"
    finally:
        conn.close()


def test_human_review_mode_does_not_dispatch_review_agent(kanban_home, monkeypatch):
    conn = kb.connect()
    spawned: list[str] = []
    try:
        (kanban_home / "config.yaml").write_text(
            "kanban:\n  review_mode: human\n",
            encoding="utf-8",
        )
        task_id = kb.create_task(conn, title="wait for Nick", assignee="test-worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.submit_task_for_review(
            conn,
            task_id,
            summary="ready for Nick",
            expected_run_id=claimed.current_run_id,
        )

        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _profile: True)
        assert kb.has_spawnable_review(conn) is False
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, _workspace: spawned.append(task.id),
            max_spawn=1,
        )

        assert result.spawned == []
        assert spawned == []
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "review"
        assert task.current_run_id is None
    finally:
        conn.close()


def test_invalid_explicit_review_mode_fails_closed_to_human(kanban_home):
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  review_mode: humam\n",
        encoding="utf-8",
    )

    assert getattr(kb, "_resolve_review_mode")() == "human"
