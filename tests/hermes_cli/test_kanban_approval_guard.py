"""Durable human-approval guard contracts for Kanban tasks."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kb.connect_closing() as connection:
        yield connection


def test_parent_completion_cannot_clear_initial_human_gate(conn) -> None:
    parent = kb.create_task(conn, title="prepare deployment")
    gated = kb.create_task(
        conn,
        title="deploy to production",
        parents=[parent],
        initial_status="blocked",
    )

    assert kb.complete_task(conn, parent, result="prepared")
    assert kb.recompute_ready(conn) == 0
    assert kb.get_task(conn, gated).status == "blocked"


def test_claim_denied_without_current_approval_and_audited(conn) -> None:
    gated = kb.create_task(
        conn,
        title="rotate production key",
        initial_status="blocked",
    )
    # Simulate a buggy/legacy writer that bypassed promotion and wrote ready.
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (gated,))

    assert kb.claim_task(conn, gated, claimer="worker") is None
    assert kb.get_task(conn, gated).status == "ready"
    denied = [
        event for event in kb.list_events(conn, gated)
        if event.kind == "approval_claim_denied"
    ]
    assert denied
    assert denied[-1].payload == {
        "source": "claim",
        "reason": "approval_pending",
        "task_revision": 1,
        "approved_revision": None,
    }


def test_valid_current_approval_permits_claim(conn) -> None:
    gated = kb.create_task(
        conn,
        title="publish release",
        approval_required=True,
    )

    assert kb.approve_task(conn, gated, actor="operator")
    approved = kb.get_task(conn, gated)
    assert kb.approval_state(approved) == "approved"
    assert approved.status == "ready"

    claimed = kb.claim_task(conn, gated, claimer="worker")
    assert claimed is not None
    assert claimed.status == "running"


def test_task_body_change_makes_existing_approval_stale(conn) -> None:
    gated = kb.create_task(
        conn,
        title="publish release",
        body="publish version one",
        approval_required=True,
    )
    assert kb.approve_task(conn, gated, actor="operator")

    assert kb.update_task_scope(
        conn,
        gated,
        body="publish version two",
        actor="dashboard",
    )

    changed = kb.get_task(conn, gated)
    assert changed.task_revision == 2
    assert changed.approved_revision == 1
    assert kb.approval_state(changed) == "stale"
    assert changed.status == "blocked"
    stale = [
        event for event in kb.list_events(conn, gated)
        if event.kind == "approval_stale"
    ]
    assert stale
    assert stale[-1].payload == {
        "actor": "dashboard",
        "previous_revision": 1,
        "task_revision": 2,
        "scope_hash": changed.approval_scope_hash,
        "changed_fields": ["body"],
    }


def test_claim_recomputes_scope_and_denies_unversioned_legacy_edit(conn) -> None:
    gated = kb.create_task(
        conn,
        title="publish release",
        body="approved body",
        approval_required=True,
    )
    assert kb.approve_task(conn, gated, actor="operator")
    # A legacy/custom writer may bypass update_task_scope. Claim enforcement
    # must compare the actual row scope, not trust two stored matching hashes.
    conn.execute(
        "UPDATE tasks SET body = 'changed behind the guard' WHERE id = ?",
        (gated,),
    )

    assert kb.claim_task(conn, gated, claimer="worker") is None
    denied = [
        event for event in kb.list_events(conn, gated)
        if event.kind == "approval_claim_denied"
    ]
    assert denied[-1].payload["reason"] == "approval_stale"


@pytest.mark.parametrize("scope_change", ["assignee", "dependency"])
def test_execution_scope_change_makes_existing_approval_stale(
    conn, scope_change: str,
) -> None:
    gated = kb.create_task(
        conn,
        title="publish release",
        assignee="worker-a",
        approval_required=True,
    )
    assert kb.approve_task(conn, gated, actor="operator")

    if scope_change == "assignee":
        assert kb.assign_task(conn, gated, "worker-b")
    else:
        parent = kb.create_task(conn, title="release prerequisite")
        kb.link_tasks(conn, parent, gated)

    changed = kb.get_task(conn, gated)
    assert changed.task_revision == 2
    assert changed.approved_revision == 1
    assert kb.approval_state(changed) == "stale"
    assert changed.status == "blocked"
    stale = [
        event for event in kb.list_events(conn, gated)
        if event.kind == "approval_stale"
    ]
    assert stale[-1].payload["changed_fields"] == [scope_change]


def test_workspace_change_makes_existing_approval_stale(conn, tmp_path) -> None:
    gated = kb.create_task(
        conn,
        title="deploy",
        workspace_kind="dir",
        workspace_path=str(tmp_path / "before"),
        approval_required=True,
    )
    assert kb.approve_task(conn, gated, actor="operator")
    approved = kb.get_task(conn, gated)

    kb.set_workspace_path(conn, gated, tmp_path / "after")

    changed = kb.get_task(conn, gated)
    assert changed.task_revision == approved.task_revision + 1
    assert kb.approval_state(changed) == "stale"
    assert changed.status == "blocked"
    stale = [
        event for event in kb.list_events(conn, gated)
        if event.kind == "approval_stale"
    ]
    assert stale[-1].payload["changed_fields"] == ["workspace_path"]


def test_revoke_clears_evidence_and_blocks_future_claim(conn) -> None:
    gated = kb.create_task(conn, title="deploy", approval_required=True)
    assert kb.approve_task(conn, gated, actor="operator")

    assert kb.revoke_approval(conn, gated, actor="operator")

    revoked = kb.get_task(conn, gated)
    assert kb.approval_state(revoked) == "pending"
    assert revoked.approved_task_id is None
    assert revoked.approved_revision is None
    assert revoked.approved_scope_hash is None
    assert revoked.approved_at is None
    assert revoked.status == "blocked"
    events = [event.kind for event in kb.list_events(conn, gated)]
    assert "approval_revoked" in events


def test_needs_input_requires_approval_and_unblock_is_not_approval(conn) -> None:
    task_id = kb.create_task(conn, title="choose deployment region")
    assert kb.claim_task(conn, task_id, claimer="worker") is not None
    assert kb.block_task(
        conn,
        task_id,
        reason="operator must choose a region",
        kind="needs_input",
    )

    gated = kb.get_task(conn, task_id)
    assert gated.approval_required is True
    assert kb.approval_state(gated) == "pending"
    assert "approval_required" in {
        event.kind for event in kb.list_events(conn, task_id)
    }

    assert kb.unblock_task(conn, task_id)
    unblocked = kb.get_task(conn, task_id)
    assert unblocked.status == "todo"
    assert kb.approval_state(unblocked) == "pending"
    assert kb.recompute_ready(conn) == 0
    assert kb.get_task(conn, task_id).status == "todo"


def test_ordinary_dependency_promotion_and_claim_are_unchanged(conn) -> None:
    parent = kb.create_task(conn, title="parent")
    child = kb.create_task(conn, title="child", parents=[parent])

    assert kb.complete_task(conn, parent, result="done")
    promoted = kb.get_task(conn, child)
    assert promoted.status == "ready"
    assert kb.approval_state(promoted) == "not_required"
    assert kb.claim_task(conn, child, claimer="worker") is not None


def test_legacy_tasks_migrate_as_non_gated_and_remain_claimable(tmp_path) -> None:
    db_path = tmp_path / "legacy-kanban.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    legacy.execute(
        """
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old task', 'ready', 1)"
    )
    legacy.commit()
    legacy.close()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect_closing(db_path=db_path) as migrated:
        task = kb.get_task(migrated, "legacy")
        assert task.task_revision == 1
        assert task.approval_required is False
        assert kb.approval_state(task) == "not_required"
        assert kb.claim_task(migrated, "legacy", claimer="worker") is not None


def test_force_promote_cannot_bypass_missing_approval(conn) -> None:
    gated = kb.create_task(
        conn,
        title="delete production dataset",
        approval_required=True,
    )

    promoted, error = kb.promote_task(
        conn,
        gated,
        actor="operator",
        force=True,
    )

    assert promoted is False
    assert error == "current human approval required (state=pending)"
    assert kb.get_task(conn, gated).status == "blocked"


def test_dispatcher_denies_unapproved_ready_task_before_claim(
    conn, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    gated = kb.create_task(
        conn,
        title="deploy",
        assignee="worker",
        approval_required=True,
    )
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (gated,))
    spawned: list[str] = []

    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda task, _workspace: spawned.append(task.id),
    )

    assert spawned == []
    assert result.approval_denied == [gated]
    denied = [
        event for event in kb.list_events(conn, gated)
        if event.kind == "approval_claim_denied"
    ]
    assert denied[-1].payload["source"] == "dispatcher"


def test_review_health_check_ignores_task_without_current_approval(
    conn, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    gated = kb.create_task(
        conn,
        title="review release",
        assignee="reviewer",
        approval_required=True,
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (gated,))

    assert kb.has_spawnable_review(conn) is False


def test_worker_start_revalidates_approval_and_claim_identity(conn) -> None:
    gated = kb.create_task(conn, title="deploy", approval_required=True)
    assert kb.approve_task(conn, gated, actor="operator")
    claimed = kb.claim_task(conn, gated, claimer="dispatcher")
    assert claimed is not None

    ok, reason = kb.validate_worker_claim(
        conn,
        gated,
        expected_run_id=claimed.current_run_id,
        expected_claim_lock=claimed.claim_lock,
        expected_revision=claimed.task_revision,
        expected_scope_hash=claimed.approval_scope_hash,
    )
    assert (ok, reason) == (True, None)

    assert kb.revoke_approval(conn, gated, actor="operator")
    ok, reason = kb.validate_worker_claim(
        conn,
        gated,
        expected_run_id=claimed.current_run_id,
        expected_claim_lock=claimed.claim_lock,
        expected_revision=claimed.task_revision,
        expected_scope_hash=claimed.approval_scope_hash,
    )
    assert (ok, reason) == (False, "approval_pending")
    denied = [
        event for event in kb.list_events(conn, gated)
        if event.kind == "approval_claim_denied"
    ]
    assert denied[-1].payload["source"] == "worker_start"


def test_review_claim_independently_denies_revoked_approval(conn) -> None:
    gated = kb.create_task(conn, title="review release", approval_required=True)
    assert kb.approve_task(conn, gated, actor="operator")
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (gated,))
    assert kb.revoke_approval(conn, gated, actor="operator")

    assert kb.claim_review_task(conn, gated, claimer="reviewer") is None
    denied = [
        event for event in kb.list_events(conn, gated)
        if event.kind == "approval_claim_denied"
    ]
    assert denied[-1].payload["source"] == "review_claim"


def test_worker_spawn_binds_revision_and_approval_scope(
    conn, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    gated = kb.create_task(
        conn,
        title="deploy",
        assignee="worker",
        approval_required=True,
    )
    assert kb.approve_task(conn, gated, actor="operator")
    claimed = kb.claim_task(conn, gated, claimer="dispatcher")
    assert claimed is not None
    captured: dict = {}

    class FakeProc:
        pid = 1234

    def fake_popen(_cmd, *args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert kb._default_spawn(claimed, str(workspace)) == 1234
    assert captured["env"]["HERMES_KANBAN_TASK_REVISION"] == "1"
    assert captured["env"]["HERMES_KANBAN_APPROVAL_SCOPE_HASH"] == (
        claimed.approval_scope_hash
    )


def test_dispatch_workspace_resolution_does_not_stale_approved_scope(
    conn, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    gated = kb.create_task(
        conn,
        title="deploy",
        assignee="worker",
        approval_required=True,
    )
    assert kb.approve_task(conn, gated, actor="operator")
    validations: list[tuple[bool, str | None]] = []

    def spawn(claimed, _workspace):
        validations.append(kb.validate_worker_claim(
            conn,
            claimed.id,
            expected_run_id=claimed.current_run_id,
            expected_claim_lock=claimed.claim_lock,
            expected_revision=claimed.task_revision,
            expected_scope_hash=claimed.approval_scope_hash,
        ))
        return 1234

    result = kb.dispatch_once(conn, spawn_fn=spawn)

    assert result.spawned and result.spawned[0][0] == gated
    assert validations == [(True, None)]


def test_dispatcher_pre_spawn_recheck_denies_post_claim_scope_race(
    conn, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    gated = kb.create_task(
        conn,
        title="deploy",
        body="approved body",
        assignee="worker",
        approval_required=True,
    )
    assert kb.approve_task(conn, gated, actor="operator")
    original_bind = kb._bind_resolved_workspace_to_approval

    def race_bind(connection, before, *, expected_parents):
        connection.execute(
            "UPDATE tasks SET body = 'changed after claim' WHERE id = ?",
            (before.id,),
        )
        return original_bind(
            connection,
            before,
            expected_parents=expected_parents,
        )

    monkeypatch.setattr(kb, "_bind_resolved_workspace_to_approval", race_bind)
    spawned: list[str] = []

    result = kb.dispatch_once(
        conn,
        spawn_fn=lambda task, _workspace: spawned.append(task.id),
    )

    assert spawned == []
    assert result.approval_denied == [gated]
    assert kb.get_task(conn, gated).status == "blocked"
    denied = [
        event for event in kb.list_events(conn, gated)
        if event.kind == "approval_claim_denied"
    ]
    assert denied[-1].payload["source"] == "dispatcher_pre_spawn"


def test_worker_environment_preflight_fails_closed_after_revoke(
    conn, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gated = kb.create_task(conn, title="deploy", approval_required=True)
    assert kb.approve_task(conn, gated, actor="operator")
    claimed = kb.claim_task(conn, gated, claimer="dispatcher")
    assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", gated)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claimed.claim_lock)
    monkeypatch.setenv(
        "HERMES_KANBAN_TASK_REVISION", str(claimed.task_revision),
    )
    monkeypatch.setenv(
        "HERMES_KANBAN_APPROVAL_SCOPE_HASH", claimed.approval_scope_hash,
    )

    assert kb.validate_worker_claim_from_env() == (True, None)
    assert kb.revoke_approval(conn, gated, actor="operator")
    assert kb.validate_worker_claim_from_env() == (False, "approval_pending")


def test_cli_parser_and_status_expose_safe_approval_contract(conn) -> None:
    from hermes_cli import kanban as kanban_cli

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    kanban_cli.build_parser(subparsers)
    create_args = parser.parse_args(
        ["kanban", "create", "deploy", "--approval-required"],
    )
    approve_args = parser.parse_args(
        ["kanban", "approve", "t_example", "--actor", "operator"],
    )
    revoke_args = parser.parse_args(
        ["kanban", "revoke-approval", "t_example", "--actor", "operator"],
    )
    assert create_args.approval_required is True
    assert approve_args.kanban_action == "approve"
    assert revoke_args.kanban_action == "revoke-approval"

    gated = kb.create_task(conn, title="deploy", approval_required=True)
    assert kb.approve_task(conn, gated, actor="operator")
    payload = kanban_cli._task_to_dict(kb.get_task(conn, gated))
    assert payload["approval"] == kb.approval_status(kb.get_task(conn, gated))
    assert "approved_by" not in payload
    assert "approved_scope_hash" not in payload


def test_cli_approve_and_revoke_commands_mutate_durable_state(
    conn, capsys: pytest.CaptureFixture[str],
) -> None:
    from argparse import Namespace
    from hermes_cli import kanban as kanban_cli

    gated = kb.create_task(conn, title="deploy", approval_required=True)
    assert kanban_cli._cmd_approve(
        Namespace(task_id=gated, actor="operator", json=False),
    ) == 0
    assert kb.approval_state(kb.get_task(conn, gated)) == "approved"

    assert kanban_cli._cmd_revoke_approval(
        Namespace(task_id=gated, actor="operator", json=False),
    ) == 0
    assert kb.approval_state(kb.get_task(conn, gated)) == "pending"
    output = capsys.readouterr().out
    assert "Approved" in output
    assert "Revoked approval" in output
