"""Focused tests for durable project runtime registration boundaries."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.project_finalization_contract import (
    acquire_finalization_lock,
    create_project_finalization,
    get_project_finalization,
    list_project_members,
    record_checker_verdict,
)
from hermes_cli.project_repair_router import (
    REGISTRATION_ALREADY_EXISTS,
    REGISTRATION_CREATED,
    REGISTRATION_STALE_SNAPSHOT,
    ProjectIdentity,
    ProjectVersionToken,
    RepairAction,
)
from hermes_cli.project_runtime_registration import (
    DESTINATION_FOUND,
    DESTINATION_MISSING,
    AtomicCheckerRegistration,
    CheckerRegistrationAction,
    checker_registration_identity,
    notification_route_identity,
    register_project_checker,
    register_project_repair,
    resolve_project_telegram_destination,
)


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def _contract() -> dict:
    return {
        "version": 1,
        "scope": "bounded project runtime task",
        "allowed_files": ["hermes_cli/project_finalizer.py"],
        "forbidden_files": [],
        "base_commit": "a" * 40,
        "required_evidence": ["focused tests"],
        "required_commands": ["pytest"],
        "allow_child_creation": False,
        "forbidden_git_actions": ["push"],
        "notification_verified": True,
    }


def _setup_project(conn: sqlite3.Connection):
    root = kb.create_task(conn, title="root", workspace_kind="dir", workspace_path="C:/repo")
    failed = kb.create_task(conn, title="failed", workspace_kind="dir", workspace_path="C:/repo")
    checker = kb.create_task(conn, title="placeholder checker", workspace_kind="dir", workspace_path="C:/repo")
    project = create_project_finalization(
        conn,
        board_id="board-a",
        root_task_id=root,
        final_checker_task_id=checker,
        repair_budget=1,
    )
    kb.add_notify_sub(
        conn,
        task_id=root,
        platform="telegram",
        chat_id="-100-secret",
        thread_id="42",
        user_id="private-user",
        notifier_profile="private-profile",
    )
    assert acquire_finalization_lock(
        conn,
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
        owner="runtime-lock",
        lease_seconds=100,
        now="100",
    )
    identity = ProjectIdentity(
        project_id="project-a",
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
    )
    token = ProjectVersionToken(
        snapshot_version="sha256:snapshot-a",
        project_version=project.version,
        lock_token="runtime-lock",
    )
    route = notification_route_identity("telegram", "-100-secret", "42")
    return identity, token, failed, checker, route


def _repair_action(project: ProjectIdentity, failed: str, route: str) -> RepairAction:
    return RepairAction(
        project=project,
        repair_identity="repair:sha256:" + "b" * 64,
        idempotency_key="repair:sha256:" + "b" * 64,
        failed_task_id=failed,
        failed_run_id=17,
        failure_fingerprint="c" * 64,
        repair_index=1,
        task_retry_index=1,
        worker_profile="builder-grok",
        task_contract=_contract(),
        notification_route_identities=(route,),
    )


def _checker_action(
    project: ProjectIdentity,
    route: str,
    *,
    snapshot: str = "sha256:candidate-a",
    candidate: str = "candidate-a",
) -> CheckerRegistrationAction:
    identity = checker_registration_identity(
        project,
        candidate_snapshot_version=snapshot,
        candidate_id=candidate,
    )
    return CheckerRegistrationAction(
        project=project,
        checker_identity=identity,
        idempotency_key=identity,
        candidate_snapshot_version=snapshot,
        candidate_id=candidate,
        worker_profile="checker-terra",
        task_contract=_contract(),
        notification_route_identities=(route,),
    )


def test_repair_registration_is_atomic_and_restart_idempotent(board):
    project, token, failed, _, route = _setup_project(board)
    action = _repair_action(project, failed, route)

    created = register_project_repair(board, action, token, now=101)
    replayed = register_project_repair(board, action, token, now=101)

    assert created.disposition == REGISTRATION_CREATED
    assert replayed.disposition == REGISTRATION_ALREADY_EXISTS
    assert replayed.repair_task_id == created.repair_task_id
    assert board.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (action.idempotency_key,)
    ).fetchone()[0] == 1
    members = list_project_members(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )
    assert [(member.task_id, member.membership_kind, member.required) for member in members] == [
        (created.repair_task_id, "repair", True)
    ]
    aggregate = get_project_finalization(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )
    assert aggregate.repair_generation == 1
    assert aggregate.state == "repairing"
    assert aggregate.version == token.project_version + 1
    event = board.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'project_repair_registered'",
        (created.repair_task_id,),
    ).fetchone()
    assert json.loads(event["payload"])["failure_fingerprint"] == action.failure_fingerprint
    assert kb.list_notify_subs(board, created.repair_task_id)[0]["chat_id"] == "-100-secret"


def test_route_inheritance_handles_threaded_and_unthreaded_destinations(board):
    project, token, failed, _, threaded_route = _setup_project(board)
    kb.add_notify_sub(
        board,
        task_id=project.root_task_id,
        platform="telegram",
        chat_id="-100-secret",
        thread_id=None,
    )
    unthreaded_route = notification_route_identity("telegram", "-100-secret", None)
    action = replace(
        _repair_action(project, failed, threaded_route),
        notification_route_identities=(threaded_route, unthreaded_route),
    )

    result = register_project_repair(board, action, token, now=101)

    assert {
        (row["chat_id"], row["thread_id"])
        for row in kb.list_notify_subs(board, result.repair_task_id)
    } == {("-100-secret", ""), ("-100-secret", "42")}


def test_runtime_registration_preserves_task_admission_gate(board):
    project, token, failed, _, route = _setup_project(board)
    action = _repair_action(project, failed, route)
    action = replace(action, task_contract={**action.task_contract, "base_commit": "not-a-sha"})

    result = register_project_repair(board, action, token, now=101)

    task = kb.get_task(board, result.repair_task_id)
    assert task.status == "todo"
    event = board.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'admission_rejected'",
        (result.repair_task_id,),
    ).fetchone()
    assert "invalid_base_commit" in json.loads(event["payload"])["reasons"]


@pytest.mark.parametrize(
    "change",
    [
        lambda action, token: (action, replace(token, project_version=token.project_version + 1)),
        lambda action, token: (action, replace(token, lock_token="wrong-lock")),
        lambda action, token: (
            replace(action, project=replace(action.project, generation=action.project.generation + 1)),
            token,
        ),
    ],
)
def test_repair_registration_refuses_stale_version_lock_or_generation_without_writes(board, change):
    project, token, failed, _, route = _setup_project(board)
    action, changed_token = change(_repair_action(project, failed, route), token)
    before = board.total_changes

    result = register_project_repair(board, action, changed_token, now=101)

    assert result.disposition == REGISTRATION_STALE_SNAPSHOT
    assert board.total_changes == before
    assert board.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (action.idempotency_key,)
    ).fetchone()[0] == 0


@pytest.mark.parametrize("failure_point", ["after_task", "after_membership", "after_project_update"])
def test_repair_registration_rolls_back_every_aggregate_write(board, failure_point):
    project, token, failed, _, route = _setup_project(board)
    action = _repair_action(project, failed, route)

    def fail(point: str) -> None:
        if point == failure_point:
            raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        register_project_repair(board, action, token, now=101, inject_failure=fail)

    assert board.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (action.idempotency_key,)
    ).fetchone()[0] == 0
    assert board.execute(
        "SELECT COUNT(*) FROM project_finalization_members WHERE membership_kind = 'repair'"
    ).fetchone()[0] == 0
    aggregate = get_project_finalization(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )
    assert aggregate.repair_generation == 0
    assert aggregate.version == token.project_version


def test_concurrent_repair_registration_returns_one_durable_identity(board):
    project, token, failed, _, route = _setup_project(board)
    action = _repair_action(project, failed, route)
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def register() -> None:
        conn = kb.connect()
        try:
            barrier.wait()
            results.append(register_project_repair(conn, action, token, now=101))
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            conn.close()

    threads = [threading.Thread(target=register) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(result.disposition for result in results) == [
        REGISTRATION_ALREADY_EXISTS,
        REGISTRATION_CREATED,
    ]
    assert len({result.repair_task_id for result in results}) == 1
    assert board.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (action.idempotency_key,)
    ).fetchone()[0] == 1


def test_checker_registration_is_atomic_authoritative_and_restart_idempotent(board):
    project, token, _, _, route = _setup_project(board)
    action = _checker_action(project, route)

    created = register_project_checker(board, action, token, now=101)
    replayed = register_project_checker(board, action, token, now=101)

    assert isinstance(created, AtomicCheckerRegistration)
    assert created.disposition == REGISTRATION_CREATED
    assert replayed.disposition == REGISTRATION_ALREADY_EXISTS
    assert replayed.checker_task_id == created.checker_task_id
    assert replayed.checker_identity == action.checker_identity
    aggregate = get_project_finalization(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )
    assert aggregate.final_checker_task_id == created.checker_task_id
    assert aggregate.checker_verdict is None
    assert aggregate.state == "evaluating"
    assert aggregate.version == token.project_version + 1
    members = list_project_members(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )
    assert [(member.task_id, member.membership_kind, member.required) for member in members] == [
        (created.checker_task_id, "checker", True)
    ]


def test_fresh_checker_replaces_stale_authority_without_deleting_history(board):
    project, token, _, _, route = _setup_project(board)
    first = register_project_checker(board, _checker_action(project, route), token, now=101)
    assert kb.complete_task(board, first.checker_task_id, result="checked")
    record_checker_verdict(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
        checker_task_id=first.checker_task_id,
        verdict="PASS",
    )
    next_token = replace(token, project_version=token.project_version + 1)
    fresh_action = _checker_action(
        project,
        route,
        snapshot="sha256:candidate-b",
        candidate="candidate-b",
    )

    fresh = register_project_checker(board, fresh_action, next_token, now=102)

    aggregate = get_project_finalization(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )
    assert fresh.disposition == REGISTRATION_CREATED
    assert fresh.checker_task_id != first.checker_task_id
    assert aggregate.final_checker_task_id == fresh.checker_task_id
    assert aggregate.checker_verdict is None
    members = list_project_members(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )
    assert [(member.task_id, member.membership_kind, member.required) for member in members] == [
        (first.checker_task_id, "support", False),
        (fresh.checker_task_id, "checker", True),
    ]
    assert board.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'project_checker_registered'",
        (first.checker_task_id,),
    ).fetchone()[0] == 1


@pytest.mark.parametrize("failure_point", ["after_task", "after_membership", "after_project_update"])
def test_checker_registration_rolls_back_every_aggregate_write(board, failure_point):
    project, token, _, placeholder, route = _setup_project(board)
    action = _checker_action(project, route)

    def fail(point: str) -> None:
        if point == failure_point:
            raise RuntimeError("injected checker failure")

    with pytest.raises(RuntimeError, match="injected checker failure"):
        register_project_checker(board, action, token, now=101, inject_failure=fail)

    assert board.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (action.idempotency_key,)
    ).fetchone()[0] == 0
    aggregate = get_project_finalization(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )
    assert aggregate.final_checker_task_id == placeholder
    assert aggregate.version == token.project_version


def test_concurrent_checker_registration_returns_one_durable_identity(board):
    project, token, _, _, route = _setup_project(board)
    action = _checker_action(project, route)
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def register() -> None:
        conn = kb.connect()
        try:
            barrier.wait()
            results.append(register_project_checker(conn, action, token, now=101))
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            conn.close()

    threads = [threading.Thread(target=register) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(result.disposition for result in results) == [
        REGISTRATION_ALREADY_EXISTS,
        REGISTRATION_CREATED,
    ]
    assert len({result.checker_task_id for result in results}) == 1


def test_checker_registration_refuses_wrong_lock_without_partial_writes(board):
    project, token, _, _, route = _setup_project(board)
    action = _checker_action(project, route)

    result = register_project_checker(
        board,
        action,
        replace(token, lock_token="wrong-lock"),
        now=101,
    )

    assert result.disposition == REGISTRATION_STALE_SNAPSHOT
    assert board.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (action.idempotency_key,)
    ).fetchone()[0] == 0


def test_destination_resolution_is_generation_scoped_deterministic_and_private(board):
    project, _, _, _, _ = _setup_project(board)
    kb.add_notify_sub(
        board,
        task_id=project.root_task_id,
        platform="telegram",
        chat_id="-200-secret",
        thread_id=None,
        user_id="other-private-user",
    )
    unrelated = kb.create_task(board, title="unrelated")
    kb.add_notify_sub(
        board,
        task_id=unrelated,
        platform="telegram",
        chat_id="-000-not-project",
    )

    first = resolve_project_telegram_destination(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )
    second = resolve_project_telegram_destination(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation,
    )

    assert first == second
    assert first.status == DESTINATION_FOUND
    assert first.platform == "telegram"
    assert first.chat_id == "-100-secret"
    assert first.thread_id == "42"
    assert first.route_identity == notification_route_identity("telegram", "-100-secret", "42")
    assert "-100-secret" not in repr(first)
    assert "private-user" not in repr(first)
    assert not hasattr(first, "user_id")
    assert not hasattr(first, "notifier_profile")


def test_destination_resolution_returns_explicit_missing_result(board):
    project, _, _, _, _ = _setup_project(board)

    result = resolve_project_telegram_destination(
        board,
        board_id=project.board_id,
        root_task_id=project.root_task_id,
        generation=project.generation + 1,
    )

    assert result.status == DESTINATION_MISSING
    assert result.reason == "project_generation_missing"
    assert result.chat_id is None
    assert result.route_identity is None
