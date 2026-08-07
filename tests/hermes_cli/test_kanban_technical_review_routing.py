"""Technical review handoffs stay inside the reviewer lane.

The worker guidance uses ``review-required:`` as a machine-readable handoff
convention.  It must not enter the human ``blocked``/``triage`` loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "reviewer").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_review_required_handoff_routes_to_spawnable_reviewer(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="implementation", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="worker")
        assert claimed is not None

        assert kb.block_task(
            conn,
            task_id,
            reason="review-required: tests and diff are ready",
            kind="needs_input",
            expected_run_id=claimed.current_run_id,
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert task.block_kind is None
        assert task.block_recurrences == 0

        events = kb.list_events(conn, task_id)
        assert any(event.kind == "review_requested" for event in events)
        assert not any(
            event.kind in {"blocked", "block_loop_detected"} for event in events
        )

        dispatch = kb.dispatch_once(conn, dry_run=True, max_spawn=1)
        assert dispatch.spawned == [(task_id, "reviewer", "")]


def test_review_required_handoff_falls_back_without_reviewer_profile(
    kanban_home: Path,
) -> None:
    (kanban_home / "profiles" / "reviewer").rmdir()

    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="portable implementation", assignee="default"
        )
        claimed = kb.claim_task(conn, task_id, claimer="default")
        assert claimed is not None

        assert kb.block_task(
            conn,
            task_id,
            reason="review-required: portable diff is ready",
            expected_run_id=claimed.current_run_id,
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "review"
        assert task.assignee == "default"

        dispatch = kb.dispatch_once(conn, dry_run=True, max_spawn=1)
        assert dispatch.spawned == [(task_id, "default", "")]
