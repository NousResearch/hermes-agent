"""Kanban task prerequisites fail before a worker can be claimed."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import projects_db as pdb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        yield home, conn


def _write_skill(home: Path, name: str) -> Path:
    skill = home / "skills" / name / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        f"---\nname: {name}\ndescription: Test fixture.\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return skill


def _task_in_review_with_forced_skill(home: Path, conn, skill_name: str):
    builder_skill = _write_skill(home / "profiles" / "builder", skill_name)
    _write_skill(home / "profiles" / "reviewer", skill_name)
    task_id = kb.create_task(
        conn,
        title="review return prerequisite",
        assignee="builder",
        skills=[skill_name],
    )
    implementation = kb.claim_task(conn, task_id, claimer="builder:test")
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    return task_id, builder_skill


def test_worktree_creation_requires_an_absolute_anchor(board, tmp_path):
    _home, conn = board

    with pytest.raises(ValueError, match="worktree.*(?:absolute|default_workdir)"):
        kb.create_task(conn, title="unanchored", workspace_kind="worktree")
    assert kb.list_tasks(conn) == []

    explicit_repo = tmp_path / "explicit-repo"
    explicit_id = kb.create_task(
        conn,
        title="explicit",
        workspace_kind="worktree",
        workspace_path=str(explicit_repo),
    )
    explicit_task = kb.get_task(conn, explicit_id)
    assert explicit_task is not None
    assert explicit_task.workspace_path == str(explicit_repo)

    project_repo = tmp_path / "project-repo"
    with pdb.connect_closing() as project_conn:
        project_id = pdb.create_project(
            project_conn, name="Project", folders=[str(project_repo)]
        )
        project = pdb.get_project(project_conn, project_id)
    assert project is not None
    project_task_id = kb.create_task(conn, title="project", project_id=project.id)
    project_task = kb.get_task(conn, project_task_id)
    assert project_task is not None
    assert project_task.workspace_kind == "worktree"
    assert project_task.workspace_path == str(
        project_repo / ".worktrees" / project_task_id
    )

    default_repo = tmp_path / "default-repo"
    kb.write_board_metadata("default", default_workdir=str(default_repo))
    default_id = kb.create_task(conn, title="default", workspace_kind="worktree")
    default_task = kb.get_task(conn, default_id)
    assert default_task is not None
    assert default_task.workspace_path == str(default_repo)


def test_forced_skills_are_atomic_and_assignee_scoped(board, monkeypatch):
    home, conn = board
    _write_skill(home, "available-skill")
    from tools import skills_tool

    monkeypatch.setattr(
        skills_tool,
        "load_env",
        lambda: (_ for _ in ()).throw(AssertionError("preflight read profile secrets")),
    )

    with pytest.raises(ValueError, match="forced skills require an assignee"):
        kb.create_task(conn, title="unassigned", skills=["available-skill"])

    with pytest.raises(ValueError) as exc_info:
        kb.create_task(
            conn,
            title="missing",
            assignee="default",
            skills=["missing-z", "missing-a"],
        )
    message = str(exc_info.value)
    assert "missing-a" in message and "missing-z" in message
    assert kb.list_tasks(conn) == []

    task_id = kb.create_task(
        conn,
        title="valid",
        assignee="default",
        skills=["available-skill"],
    )

    compatible = home / "profiles" / "compatible"
    compatible.mkdir(parents=True)
    _write_skill(compatible, "profile-skill")
    profile_task_id = kb.create_task(
        conn,
        title="profile valid",
        assignee="compatible",
        skills=["profile-skill"],
    )
    profile_task = kb.get_task(conn, profile_task_id)
    assert profile_task is not None
    assert profile_task.skills == ["profile-skill"]

    incompatible = home / "profiles" / "incompatible"
    incompatible.mkdir(parents=True)
    with pytest.raises(ValueError, match="available-skill"):
        kb.assign_task(conn, task_id, "incompatible")
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.assignee == "default"

    assert kb.block_task(conn, task_id, reason="wait") is True
    (home / "skills" / "available-skill" / "SKILL.md").unlink()
    with pytest.raises(ValueError, match="available-skill"):
        kb.unblock_task(conn, task_id)
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "blocked"

    _write_skill(home, "dependency-skill")
    parent_id = kb.create_task(conn, title="parent")
    child_id = kb.create_task(
        conn,
        title="child",
        assignee="default",
        skills=["dependency-skill"],
        parents=[parent_id],
    )
    (home / "skills" / "dependency-skill" / "SKILL.md").unlink()
    assert kb.complete_task(conn, parent_id) is True
    kb.recompute_ready(conn)
    child = kb.get_task(conn, child_id)
    assert child is not None
    assert child.status == "blocked"
    assert [
        event.kind
        for event in kb.list_events(conn, child_id)
        if event.kind == "prerequisite_blocked"
    ] == ["prerequisite_blocked"]


def test_request_changes_refuses_incompatible_restored_implementer(board):
    home, conn = board
    task_id, builder_skill = _task_in_review_with_forced_skill(
        home, conn, "implementation-review"
    )
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:test")
    assert review is not None
    builder_skill.unlink()

    ok, detail = kb.request_changes(
        conn,
        task_id,
        reason="Please revise.",
        expected_run_id=review.current_run_id,
    )

    assert ok is False
    assert "implementation-review" in (detail or "")
    unchanged = kb.get_task(conn, task_id)
    assert unchanged is not None
    assert unchanged.status == "running"
    assert unchanged.assignee == "reviewer"
    assert unchanged.current_run_id == review.current_run_id
    assert not any(
        event.kind == "changes_requested" for event in kb.list_events(conn, task_id)
    )


def test_dependency_promotion_preserves_a_legacy_board_default_worktree(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    board_name = "legacy-board"
    kb.create_board(board_name)

    from hermes_cli import kanban_prerequisites as prerequisites

    with kbc.connect_closing(board=board_name) as conn:
        parent_id = kb.create_task(conn, title="parent", board=board_name)
        with monkeypatch.context() as creation_patch:
            creation_patch.setattr(
                prerequisites, "validate_worktree_anchor", lambda *args, **kwargs: None
            )
            task_id = kb.create_task(
                conn,
                title="legacy worktree",
                assignee="default",
                workspace_kind="worktree",
                parents=[parent_id],
                board=board_name,
            )
        repository = tmp_path / "repository"
        kb.write_board_metadata(board_name, default_workdir=str(repository))
        assert kb.complete_task(conn, parent_id) is True
        promoted = kb.get_task(conn, task_id)
        assert promoted is not None
        assert promoted.status == "ready"

        from hermes_cli.kanban_prerequisites import validate_task

        validate_task(promoted, board=board_name)

        kb.write_board_metadata(board_name, default_workdir="")
        with monkeypatch.context() as creation_patch:
            creation_patch.setattr(
                prerequisites, "validate_worktree_anchor", lambda *args, **kwargs: None
            )
            blocked_id = kb.create_task(
                conn,
                title="blocked legacy worktree",
                assignee="default",
                workspace_kind="worktree",
                board=board_name,
            )
            assert kb.block_task(conn, blocked_id, reason="wait") is True
            promote_id = kb.create_task(
                conn,
                title="manually promoted legacy worktree",
                assignee="default",
                workspace_kind="worktree",
                initial_status="blocked",
                board=board_name,
            )
        kb.write_board_metadata(board_name, default_workdir=str(repository))

        assert kb.unblock_task(conn, blocked_id) is True
        unblocked = kb.get_task(conn, blocked_id)
        assert unblocked is not None
        assert unblocked.status == "ready"

        promoted_ok, promoted_error = kb.promote_task(conn, promote_id, actor="test")
        assert promoted_ok is True
        assert promoted_error is None
        manually_promoted = kb.get_task(conn, promote_id)
        assert manually_promoted is not None
        assert manually_promoted.status == "ready"


def test_claim_boundary_blocks_incompatible_card_after_reclaim(board):
    """Vladamir's reproduced sequence: valid claim -> skill removal -> reclaim
    -> direct re-claim must NOT open a new run."""
    home, conn = board
    skill = _write_skill(home, "reclaim-guard-skill")
    task_id = kb.create_task(
        conn, title="reclaim guard", assignee="default", skills=["reclaim-guard-skill"]
    )

    first = kb.claim_task(conn, task_id, claimer="host:a")
    assert first is not None
    run_ids_before = [run.id for run in kb.list_runs(conn, task_id)]

    assert kb.reclaim_task(conn, task_id, reason="operator") is True
    ready = kb.get_task(conn, task_id)
    assert ready is not None
    assert ready.status == "ready"

    skill.unlink()
    refused = kb.claim_task(conn, task_id, claimer="host:b")
    assert refused is None

    blocked = kb.get_task(conn, task_id)
    assert blocked is not None
    assert blocked.status == "blocked"
    assert blocked.assignee == "default"
    assert blocked.claim_lock is None
    assert blocked.consecutive_failures == 0
    assert blocked.current_run_id is None
    runs_after = kb.list_runs(conn, task_id)
    assert [run.id for run in runs_after] == run_ids_before
    events = [
        event
        for event in kb.list_events(conn, task_id)
        if event.kind == "prerequisite_blocked"
    ]
    assert len(events) == 1
    assert events[0].payload is not None
    assert "reclaim-guard-skill" in events[0].payload["reason"]
    assert events[0].payload["source_status"] == "ready"


def test_review_claim_boundary_blocks_incompatible_reviewer(board):
    home, conn = board
    task_id, _builder_skill = _task_in_review_with_forced_skill(
        home, conn, "review-claim-guard"
    )

    valid_review = kb.claim_review_task(conn, task_id, claimer="reviewer:test")
    assert valid_review is not None
    assert kb.reclaim_task(conn, task_id, reason="operator") is True
    back_to_review = kb.get_task(conn, task_id)
    assert back_to_review is not None
    assert back_to_review.status == "review"

    reviewer_skill = (
        home / "profiles" / "reviewer" / "skills" / "review-claim-guard" / "SKILL.md"
    )
    reviewer_skill.unlink()

    refused = kb.claim_review_task(conn, task_id, claimer="reviewer:test")
    assert refused is None

    blocked = kb.get_task(conn, task_id)
    assert blocked is not None
    assert blocked.status == "blocked"
    assert blocked.consecutive_failures == 0
    events = [
        event
        for event in kb.list_events(conn, task_id)
        if event.kind == "prerequisite_blocked"
    ]
    assert len(events) == 1
    assert events[0].payload is not None
    assert "review-claim-guard" in events[0].payload["reason"]
    assert events[0].payload["source_status"] == "review"


def test_crash_requeue_refuses_incompatible_card_before_claim(board, monkeypatch):
    """Automatic requeue class: crash detection restores ``ready``, and the
    central claim boundary (not just the dispatcher) refuses the card whose
    forced skill vanished mid-run — no new run, no EXTRA retry increment
    beyond the crash's own accounting."""
    import subprocess

    home, conn = board
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    skill = _write_skill(home, "crash-route-skill")
    task_id = kb.create_task(
        conn, title="crash route", assignee="default", skills=["crash-route-skill"]
    )

    host = kb._claimer_id().split(":", 1)[0]
    first = kb.claim_task(conn, task_id, claimer=f"{host}:a")
    assert first is not None
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, task_id, dead.pid)

    skill.unlink()
    assert kbd.detect_crashed_workers(conn) == [task_id]

    requeued = kb.get_task(conn, task_id)
    assert requeued is not None
    assert requeued.status == "ready"
    failures_after_crash = requeued.consecutive_failures
    run_ids_after_crash = [run.id for run in kb.list_runs(conn, task_id)]

    refused = kb.claim_task(conn, task_id, claimer="host:b")
    assert refused is None
    blocked = kb.get_task(conn, task_id)
    assert blocked is not None
    assert blocked.status == "blocked"
    assert blocked.consecutive_failures == failures_after_crash
    assert [run.id for run in kb.list_runs(conn, task_id)] == run_ids_after_crash
    events = [
        event
        for event in kb.list_events(conn, task_id)
        if event.kind == "prerequisite_blocked"
    ]
    assert len(events) == 1
    assert events[0].payload is not None
    assert "crash-route-skill" in events[0].payload["reason"]


def test_dispatch_blocks_legacy_invalid_task_without_claim_or_retry(board):
    home, conn = board
    skill = _write_skill(home, "removed-skill")
    task_id = kb.create_task(
        conn,
        title="legacy",
        assignee="default",
        skills=["removed-skill"],
    )
    skill.unlink()
    spawns = []

    first = kbd.dispatch_once(
        conn, spawn_fn=lambda *args, **kwargs: spawns.append(args)
    )
    second = kbd.dispatch_once(
        conn, spawn_fn=lambda *args, **kwargs: spawns.append(args)
    )

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "blocked"
    assert task.consecutive_failures == 0
    assert task.current_run_id is None
    assert kb.list_runs(conn, task_id) == []
    events = [
        event
        for event in kb.list_events(conn, task_id)
        if event.kind == "prerequisite_blocked"
    ]
    assert len(events) == 1
    assert events[0].payload is not None
    assert "removed-skill" in events[0].payload["reason"]
    assert first.spawned == second.spawned == []
    assert spawns == []
