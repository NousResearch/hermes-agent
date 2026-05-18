"""Lifecycle tests for lane-driven reviewer auto-spawn."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_and_complete_coder_task(
    conn,
    *,
    title: str,
    body: str,
    summary: str,
    metadata: dict | None = None,
) -> tuple[str, int]:
    source = kb.create_task(conn, title=title, assignee="coder", body=body)
    kb.claim_task(conn, source)
    run = kb.active_run(conn, source)
    assert run is not None
    assert kb.complete_task(
        conn,
        source,
        summary=summary,
        metadata=metadata or {},
        expected_run_id=run.id,
    )
    return source, run.id


def _reviewer_children_for_source(conn, source_task_id: str) -> list[kb.Task]:
    reviewers = [task for task in kb.list_tasks(conn, assignee="reviewer")]
    return [task for task in reviewers if source_task_id in kb.parent_ids(conn, task.id)]


def test_standard_coder_completion_spawns_reviewer_b(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    monkeypatch.setattr(kb, "_validate_task_extra_skills", lambda _skills: [])
    spawned: list[str] = []

    def fake_spawn(task, _workspace):
        spawned.append(task.id)
        return 0

    body = """
review_lane: STANDARD_REVIEW
"""

    with kb.connect() as conn:
        source, parent_run_id = _create_and_complete_coder_task(
            conn,
            title="Implement lane routing",
            body=body,
            summary="lane routing implemented",
            metadata={"changed_files": ["hermes_cli/kanban_db.py"]},
        )

        kb.dispatch_once(conn, spawn_fn=fake_spawn)

        reviewer_children = _reviewer_children_for_source(conn, source)
        assert len(reviewer_children) == 1
        reviewer = reviewer_children[0]

        assert reviewer.assignee == "reviewer"
        assert reviewer.skills == ["kanban-reviewer"]
        assert reviewer.max_runtime_seconds == 12 * 60
        assert reviewer.max_retries == 1
        assert reviewer.title.startswith("Review ")
        assert len(reviewer.title) <= 80

        reviewer_body = reviewer.body or ""
        assert f"parent_task: {source}" in reviewer_body
        assert f"parent_run: {parent_run_id}" in reviewer_body
        assert "review_lane: STANDARD_REVIEW" in reviewer_body
        assert "review_stage: reviewer_b" in reviewer_body
        assert "completion_summary: lane routing implemented" in reviewer_body

        idempotency = conn.execute(
            "SELECT idempotency_key FROM tasks WHERE id = ?",
            (reviewer.id,),
        ).fetchone()["idempotency_key"]
        assert idempotency == f"auto-reviewer:{source}:{parent_run_id}"
        assert spawned == [reviewer.id]


def test_standard_coder_completion_with_manual_reviewer_child_suppresses_auto_reviewer_b(
    kanban_home, all_assignees_spawnable
):
    spawned: list[str] = []

    def fake_spawn(task, _workspace):
        spawned.append(task.id)
        return 0

    with kb.connect() as conn:
        source, parent_run_id = _create_and_complete_coder_task(
            conn,
            title="Implement manual review pipeline",
            body="review_lane: STANDARD_REVIEW\n",
            summary="manual pipeline implementation done",
            metadata={"changed_files": ["hermes_cli/kanban_db.py"]},
        )
        manual_reviewer = kb.create_task(
            conn,
            title="Manual Reviewer-B for implementation",
            assignee="reviewer",
            parents=[source],
        )

        kb.dispatch_once(conn, spawn_fn=fake_spawn)

        reviewer_children = _reviewer_children_for_source(conn, source)
        assert [child.id for child in reviewer_children] == [manual_reviewer]
        assert spawned == [manual_reviewer]

        suppression_events = [
            event
            for event in kb.list_events(conn, source)
            if event.kind == "dispatch_auto_reviewer_child_suppressed"
        ]
        assert len(suppression_events) == 1
        assert suppression_events[0].run_id == parent_run_id
        suppression_payload = suppression_events[0].payload or {}
        assert suppression_payload["reason"] == "manual_reviewer_child_present"
        assert suppression_payload["manual_reviewer_children"] == [manual_reviewer]

        auto_events = [
            event
            for event in kb.list_events(conn, source)
            if event.kind == "dispatch_auto_reviewer_child_created"
        ]
        assert auto_events == []


def test_standard_coder_completion_with_manual_review_opt_out_suppresses_auto_reviewer_b(
    kanban_home, all_assignees_spawnable
):
    body = """
review_lane: STANDARD_REVIEW
review_pipeline: manual
auto_reviewer_b: false
"""

    with kb.connect() as conn:
        source, parent_run_id = _create_and_complete_coder_task(
            conn,
            title="Implement documented manual review pipeline",
            body=body,
            summary="manual opt-out implementation done",
            metadata={"changed_files": ["hermes_cli/kanban_db.py"]},
        )

        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)

        assert _reviewer_children_for_source(conn, source) == []
        assert res.spawned == []
        suppression_events = [
            event
            for event in kb.list_events(conn, source)
            if event.kind == "dispatch_auto_reviewer_child_suppressed"
        ]
        assert len(suppression_events) == 1
        assert suppression_events[0].run_id == parent_run_id
        suppression_payload = suppression_events[0].payload or {}
        assert suppression_payload["reason"] == "manual_review_pipeline_opt_out"
        assert suppression_payload["manual_reviewer_children"] == []


def test_coordinator_finalization_metadata_suppresses_duplicate_auto_reviewer_b(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Implement reviewed lifecycle fix",
            assignee="coder",
            body="review_lane: STANDARD_REVIEW\n",
        )
        reviewer = kb.create_task(
            conn,
            title="Independent approved reviewer",
            assignee="reviewer",
        )
        kb.claim_task(conn, source)
        run = kb.active_run(conn, source)
        assert run is not None

        assert kb.complete_task(
            conn,
            source,
            summary="Coordinator/Admin finalization after independent Reviewer APPROVED",
            metadata={
                "changed_files": ["hermes_cli/kanban_db.py"],
                "review_finalized_by_coordinator": True,
                "approved_reviewer_task": reviewer,
                "suppress_auto_reviewer_b": True,
                "reviewer_redispatch_forbidden": True,
                "lifecycle_finalization": "coordinator_admin_finalization",
            },
            expected_run_id=run.id,
        )

        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)

        spawned_ids = [item[0] if isinstance(item, tuple) else item for item in res.spawned]
        assert spawned_ids == [reviewer]
        assert _reviewer_children_for_source(conn, source) == []
        suppression_events = [
            event
            for event in kb.list_events(conn, source)
            if event.kind == "dispatch_auto_reviewer_child_suppressed"
        ]
        assert len(suppression_events) == 1
        assert suppression_events[0].run_id == run.id
        payload = suppression_events[0].payload or {}
        assert payload["reason"] == "coordinator_finalization_existing_approved_reviewer"
        assert payload["approved_reviewer_task"] == reviewer


def test_coordinator_finalization_terminalizes_legacy_parent_gated_reviewer_before_ready(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Finalize reviewed blocked source",
            assignee="coder",
            body="review_lane: STANDARD_REVIEW\n",
        )
        legacy_reviewer = kb.create_task(
            conn,
            title="Legacy parent-gated reviewer",
            assignee="reviewer",
            parents=[source],
        )
        approved_reviewer = kb.create_task(
            conn,
            title="Independent approved reviewer",
            assignee="reviewer",
        )
        initial_legacy_task = kb.get_task(conn, legacy_reviewer)
        assert initial_legacy_task is not None
        assert initial_legacy_task.status == "todo"

        kb.claim_task(conn, source)
        run = kb.active_run(conn, source)
        assert run is not None
        assert kb.complete_task(
            conn,
            source,
            summary="Coordinator/Admin finalization after independent Reviewer APPROVED",
            metadata={
                "changed_files": ["hermes_cli/kanban_db.py"],
                "review_finalized_by_coordinator": True,
                "approved_reviewer_task": approved_reviewer,
                "suppress_auto_reviewer_b": True,
                "reviewer_redispatch_forbidden": True,
                "lifecycle_finalization": "coordinator_admin_finalization",
            },
            expected_run_id=run.id,
        )

        legacy_task = kb.get_task(conn, legacy_reviewer)
        assert legacy_task is not None
        assert legacy_task.status == "done"
        assert legacy_task.result is not None
        assert legacy_task.result.startswith("superseded/noop:")
        latest_legacy_run = kb.latest_run(conn, legacy_reviewer)
        assert latest_legacy_run is not None
        assert latest_legacy_run.outcome == "completed"
        assert latest_legacy_run.metadata is not None
        assert latest_legacy_run.metadata["lifecycle_outcome"] == "superseded_noop"
        assert latest_legacy_run.metadata["superseded_by"] == approved_reviewer
        noop_events = [
            event
            for event in kb.list_events(conn, legacy_reviewer)
            if event.kind == "superseded_noop_terminalized"
        ]
        assert len(noop_events) == 1
        assert noop_events[0].payload["previous_status"] == "todo"


def test_coordinator_finalization_prose_only_does_not_suppress_auto_reviewer_b(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    monkeypatch.setattr(kb, "_validate_task_extra_skills", lambda _skills: [])
    with kb.connect() as conn:
        source = kb.create_task(
            conn,
            title="Implement prose-only reviewed lifecycle fix",
            assignee="coder",
            body="review_lane: STANDARD_REVIEW\n",
        )
        kb.claim_task(conn, source)
        run = kb.active_run(conn, source)
        assert run is not None
        assert kb.complete_task(
            conn,
            source,
            summary=(
                "Coordinator finalized after approved reviewer t_12345678; "
                "do not redispatch reviewer"
            ),
            metadata={"changed_files": ["hermes_cli/kanban_db.py"]},
            expected_run_id=run.id,
        )

        kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)

        reviewer_children = _reviewer_children_for_source(conn, source)
        assert len(reviewer_children) == 1
        assert [
            event
            for event in kb.list_events(conn, source)
            if event.kind == "dispatch_auto_reviewer_child_suppressed"
        ] == []


def test_fastlane_coder_completion_no_reviewer_spawn(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        source, _run_id = _create_and_complete_coder_task(
            conn,
            title="Fix typo in docs",
            body="review_lane: FASTLANE_KANBAN\n",
            summary="typo fixed",
            metadata={"changed_files": ["website/docs/faq.md"]},
        )

        res = kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)

        assert _reviewer_children_for_source(conn, source) == []
        assert res.spawned == []


def test_reviewer_idempotent_on_re_dispatch(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    monkeypatch.setattr(kb, "_validate_task_extra_skills", lambda _skills: [])
    with kb.connect() as conn:
        source, _run_id = _create_and_complete_coder_task(
            conn,
            title="Lifecycle semantics update",
            body="review_lane: STANDARD_REVIEW\n",
            summary="lifecycle update complete",
            metadata={"changed_files": ["hermes_cli/kanban_db.py"]},
        )

        kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)
        first_children = _reviewer_children_for_source(conn, source)
        assert len(first_children) == 1

        kb.dispatch_once(conn, spawn_fn=lambda *_args: 0)
        second_children = _reviewer_children_for_source(conn, source)
        assert [child.id for child in second_children] == [child.id for child in first_children]
