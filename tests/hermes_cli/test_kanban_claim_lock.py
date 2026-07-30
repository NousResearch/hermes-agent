"""Regression tests for claim-owned kanban completion."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_complete_task_rejects_matching_claim_lock_in_triage(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="owned work", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker:current")
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET status = 'triage' WHERE id = ?",
            (task_id,),
        )
        conn.commit()

        assert not kb.complete_task(
            conn,
            task_id,
            result="finished",
            expected_claim_lock="worker:current",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "triage"


def test_complete_task_matching_claim_lock_preserves_non_completable_state(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="human-routed work", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker:current")
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET status = 'todo' WHERE id = ?",
            (task_id,),
        )
        conn.commit()

        # A matching lock must not bypass the set of completable task states.
        assert not kb.complete_task(
            conn,
            task_id,
            result="finished",
            expected_claim_lock="worker:current",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "todo"
        assert task.result is None
        assert task.claim_lock == "worker:current"


def test_complete_task_rejects_stale_claim_lock(kanban_home: Path) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="reassigned work", assignee="worker")
        first = kb.claim_task(conn, task_id, claimer="worker:first")
        assert first is not None
        assert kb.reclaim_task(conn, task_id, signal_fn=lambda *_args: None)
        second = kb.claim_task(conn, task_id, claimer="worker:second")
        assert second is not None

        assert not kb.complete_task(
            conn,
            task_id,
            result="stale result",
            expected_claim_lock="worker:first",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.result is None
        assert task.claim_lock == "worker:second"
        assert task.current_run_id == second.current_run_id


def test_complete_task_matching_claim_lock_retry_is_idempotent(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="retry completion", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker:retry")
        assert claimed is not None

        assert kb.complete_task(
            conn,
            task_id,
            result="original",
            summary="original summary",
            expected_run_id=claimed.current_run_id,
            expected_claim_lock="worker:retry",
        )
        assert kb.complete_task(
            conn,
            task_id,
            result="replacement",
            summary="replacement summary",
            expected_run_id=claimed.current_run_id,
            expected_claim_lock="worker:retry",
        )

        task = kb.get_task(conn, task_id)
        completed = [
            event
            for event in kb.list_events(conn, task_id)
            if event.kind == "completed"
        ]
        assert task is not None
        assert task.result == "original"
        assert len(completed) == 1


def test_complete_task_without_claim_lock_preserves_legacy_behavior(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="manual completion")

        assert kb.complete_task(conn, task_id, result="done")

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.result == "done"


@pytest.mark.parametrize("status", ["running", "ready", "blocked"])
def test_complete_task_accepts_matching_claim_lock_on_completable_status(
    kanban_home: Path, status: str
) -> None:
    """The happy path of the fenced branch: right lock, every completable status.

    The rejection tests only pin what the fence rejects. Parametrized over the
    full IN list because a single positive case (say ``running``) would stay
    green if the predicate silently dropped ``ready`` or ``blocked``.
    """
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="owned work", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker:current")
        assert claimed is not None
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        conn.commit()

        assert kb.complete_task(
            conn,
            task_id,
            result="finished",
            expected_claim_lock="worker:current",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.result == "finished"
        assert task.claim_lock is None
