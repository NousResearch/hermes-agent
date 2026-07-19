"""Admission and checker-verdict authority tests for G3.

These tests deliberately exercise the persistence boundary directly.  They do
not depend on the gateway loop or a live Hermes profile directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.project_finalization_contract import (
    acquire_finalization_lock,
    get_project_finalization,
    list_project_members,
    record_checker_verdict,
    record_terminal_outcome,
    reopen_project_finalization,
)
from hermes_cli.project_finalizer import evaluate_project
from hermes_cli.project_repair_router import ProjectIdentity, ProjectVersionToken
from hermes_cli.project_runtime_registration import (
    ADMISSION_ALREADY_ADMITTED,
    ADMISSION_CREATED,
    CheckerRegistrationAction,
    admit_existing_project,
    checker_registration_identity,
    notification_route_identity,
    register_project_checker,
    submit_project_checker_verdict,
)


def _contract() -> dict[str, object]:
    return {
        "version": 1,
        "scope": "bounded admitted project task",
        "allowed_files": ["hermes_cli/project_finalizer.py"],
        "forbidden_files": [],
        "base_commit": "a" * 40,
        "required_evidence": ["focused tests"],
        "required_commands": ["pytest"],
        "allow_child_creation": False,
        "forbidden_git_actions": ["push"],
        "notification_verified": True,
    }


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.project_runtime_registration.profile_exists",
        lambda profile: profile == "checker-profile",
    )
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def _task(conn, title: str, *, assignee: str = "builder-profile", contract: dict | None = None) -> str:
    return kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        workspace_kind="dir",
        workspace_path="C:/repo",
        contract=_contract() if contract is None else contract,
    )


def _admitted(board, *, extra_route: bool = False):
    root = _task(board, "root")
    implementation = _task(board, "implementation")
    kb.add_notify_sub(board, task_id=root, platform="telegram", chat_id="-100-private", thread_id="42")
    kb.add_notify_sub(board, task_id=implementation, platform="telegram", chat_id="-100-private", thread_id="42")
    kb.recompute_ready(board)
    if extra_route:
        kb.add_notify_sub(board, task_id=root, platform="telegram", chat_id="-100-other", thread_id="43")
    result = admit_existing_project(
        board,
        board_id="board-a",
        root_task_id=root,
        required_task_ids=(implementation,),
        checker_profile="checker-profile",
        now=100,
    )
    return root, implementation, result


def _checker_ready(board):
    root, implementation, admitted = _admitted(board)
    assert kb.complete_task(board, root, result="root complete")
    assert kb.complete_task(board, implementation, result="implementation complete")
    finalization = get_project_finalization(board, board_id="board-a", root_task_id=root)
    assert finalization is not None
    assert acquire_finalization_lock(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=1,
        owner="test-lock",
        lease_seconds=1000,
        now="100",
    )
    evaluation = evaluate_project(board, board_id="board-a", root_task_id=root, generation=1, evaluation_time=101)
    assert evaluation.failure_reason == "checker_required"
    project = ProjectIdentity("project-a", "board-a", root, 1)
    checker_identity = checker_registration_identity(
        project,
        candidate_snapshot_version=evaluation.candidate_snapshot_version,
        candidate_id=evaluation.candidate_snapshot_version,
    )
    action = CheckerRegistrationAction(
        project=project,
        checker_identity=checker_identity,
        idempotency_key=checker_identity,
        candidate_snapshot_version=evaluation.candidate_snapshot_version,
        candidate_id=evaluation.candidate_snapshot_version,
        worker_profile="checker-profile",
        task_contract=_contract(),
        notification_route_identities=(notification_route_identity("telegram", "-100-private", "42"),),
    )
    token = ProjectVersionToken(evaluation.snapshot_version, finalization.version, "test-lock")
    registered = register_project_checker(board, action, token, now=101)
    assert registered.checker_task_id
    claimed = kb.claim_task(board, registered.checker_task_id, claimer="checker")
    assert claimed is not None and claimed.current_run_id is not None
    return root, admitted, registered.checker_task_id, int(claimed.current_run_id)


def _evidence() -> list[dict[str, str]]:
    return [{"kind": "test", "reference": "tests/g3", "summary": "focused authority test passed"}]


def test_admission_is_atomic_private_and_has_no_checker_member(board):
    root, implementation, result = _admitted(board)

    assert result.disposition == ADMISSION_CREATED
    assert result.finalization.final_checker_task_id.startswith("pending-checker:sha256:")
    assert result.finalization.checker_profile == "checker-profile"
    assert result.finalization.notification_route_identity.startswith("subscription:sha256:")
    assert "-100-private" not in repr(result)
    members = list_project_members(board, board_id="board-a", root_task_id=root, generation=1)
    assert {(member.task_id, member.membership_kind) for member in members} == {
        (root, "required"), (implementation, "required")
    }
    event = board.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='project_admitted'", (root,)).fetchone()
    assert event is not None and "-100-private" not in event["payload"]


def test_admission_exact_replay_and_conflict_are_distinct(board):
    root, implementation, created = _admitted(board)
    replay = admit_existing_project(
        board, board_id="board-a", root_task_id=root, required_task_ids=(implementation,),
        checker_profile="checker-profile", now=101,
    )
    assert replay.disposition == ADMISSION_ALREADY_ADMITTED
    assert replay.admission_key == created.admission_key
    with pytest.raises(ValueError, match="conflicts"):
        admit_existing_project(
            board, board_id="board-a", root_task_id=root, required_task_ids=(implementation,),
            checker_profile="checker-profile", retention_days=4,
        )


def test_admitted_project_reopen_fails_closed_without_explicit_readmission(board):
    root, _, _ = _admitted(board)
    record_terminal_outcome(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=1,
        outcome="BLOCKED",
    )
    with pytest.raises(ValueError, match="explicit re-admission"):
        reopen_project_finalization(board, board_id="board-a", root_task_id=root)


@pytest.mark.parametrize("route_count", [0, 2])
def test_admission_requires_exactly_one_root_telegram_route(board, route_count):
    root = _task(board, "root")
    implementation = _task(board, "implementation")
    for index in range(route_count):
        kb.add_notify_sub(board, task_id=root, platform="telegram", chat_id=f"-100-{index}")
    with pytest.raises(ValueError, match="exactly one Telegram"):
        admit_existing_project(
            board, board_id="board-a", root_task_id=root, required_task_ids=(implementation,),
            checker_profile="checker-profile",
        )
    assert board.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 0


def test_admission_rejects_missing_contract_and_nonindependent_checker(board):
    root = _task(board, "root")
    missing_contract = _task(board, "missing", contract=None)
    # Explicit None means the helper's default contract; overwrite it to make a legacy card.
    board.execute("UPDATE tasks SET contract=NULL WHERE id=?", (missing_contract,))
    kb.add_notify_sub(board, task_id=root, platform="telegram", chat_id="-100-private")
    with pytest.raises(ValueError, match="not admissible"):
        admit_existing_project(
            board, board_id="board-a", root_task_id=root, required_task_ids=(missing_contract,),
            checker_profile="checker-profile",
        )

    invalid_contract = _task(board, "invalid")
    board.execute("UPDATE tasks SET contract=? WHERE id=?", ('{"version":"not-an-int"}', invalid_contract))
    with pytest.raises(ValueError, match="not admissible"):
        admit_existing_project(
            board, board_id="board-a", root_task_id=root, required_task_ids=(invalid_contract,),
            checker_profile="checker-profile",
        )

    independent = _task(board, "independent", assignee="checker-profile")
    with pytest.raises(ValueError, match="independent"):
        admit_existing_project(
            board, board_id="board-a", root_task_id=root, required_task_ids=(independent,),
            checker_profile="checker-profile",
        )


def test_admission_injected_failure_rolls_back_every_authority_row(board):
    root = _task(board, "root")
    implementation = _task(board, "implementation")
    kb.add_notify_sub(board, task_id=root, platform="telegram", chat_id="-100-private")
    with pytest.raises(RuntimeError, match="injected"):
        admit_existing_project(
            board, board_id="board-a", root_task_id=root, required_task_ids=(implementation,),
            checker_profile="checker-profile",
            inject_failure=lambda stage: (_ for _ in ()).throw(RuntimeError("injected")) if stage == "after_project" else None,
        )
    assert board.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0] == 0
    assert board.execute("SELECT COUNT(*) FROM project_finalization_members").fetchone()[0] == 0
    assert board.execute("SELECT COUNT(*) FROM task_events WHERE kind='project_admitted'").fetchone()[0] == 0


def test_checker_verdict_requires_structured_protocol_and_is_immutable(board):
    root, _, checker, run_id = _checker_ready(board)
    with pytest.raises(kb.ProjectCheckerVerdictRequiredError):
        kb.complete_task(board, checker, result="bypass")
    with pytest.raises(ValueError, match="structured runtime submission"):
        record_checker_verdict(
            board,
            board_id="board-a",
            root_task_id=root,
            generation=1,
            checker_task_id=checker,
            verdict="PASS",
        )

    submitted = submit_project_checker_verdict(
        board, board_id="board-a", task_id=checker, run_id=run_id,
        worker_profile="checker-profile", verdict="PASS", reason="all requirements met", evidence=_evidence(),
    )
    assert submitted.completed is True
    assert kb.get_task(board, checker).status == "done"
    with pytest.raises(ValueError, match="conflicts"):
        submit_project_checker_verdict(
            board, board_id="board-a", task_id=checker, run_id=run_id,
            worker_profile="checker-profile", verdict="PASS", reason="different reason", evidence=_evidence(),
        )
    assert board.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='project_checker_verdict_recorded'", (checker,)
    ).fetchone()[0] == 1
    finalization = get_project_finalization(board, board_id="board-a", root_task_id=root)
    assert finalization is not None and finalization.checker_verdict == "PASS"


def test_checker_verdict_rejects_wrong_profile_run_and_stale_candidate(board):
    root, _, checker, run_id = _checker_ready(board)
    kwargs = dict(board_id="board-a", task_id=checker, run_id=run_id, verdict="PASS", reason="ok", evidence=_evidence())
    with pytest.raises(ValueError, match="does not own"):
        submit_project_checker_verdict(board, worker_profile="other-profile", **kwargs)
    with pytest.raises(ValueError, match="not current"):
        submit_project_checker_verdict(board, worker_profile="checker-profile", run_id=run_id + 1000,
                                       board_id="board-a", task_id=checker, verdict="PASS", reason="ok", evidence=_evidence())

    # A new event on a required implementation task changes the candidate digest
    # while preserving the registered authority, so submission must fail stale.
    kb._append_event(board, root, "test_candidate_changed", {"reason": "new evidence"})
    with pytest.raises(ValueError, match="candidate is stale"):
        submit_project_checker_verdict(board, worker_profile="checker-profile", **kwargs)


def test_checker_verdict_crash_after_durable_commit_retries_after_reclaim(board):
    _, _, checker, run_id = _checker_ready(board)
    kwargs = dict(
        board_id="board-a", task_id=checker, run_id=run_id, worker_profile="checker-profile",
        verdict="PASS", reason="all requirements met", evidence=_evidence(),
    )
    with pytest.raises(RuntimeError, match="crash after verdict"):
        submit_project_checker_verdict(
            board, inject_failure=lambda stage: (_ for _ in ()).throw(RuntimeError("crash after verdict"))
            if stage == "after_verdict_commit" else None,
            **kwargs,
        )
    assert kb.get_task(board, checker).status == "running"
    assert kb.block_task(
        board,
        checker,
        reason="worker process ended after durable verdict commit",
        kind="transient",
        expected_run_id=run_id,
    )
    assert kb.unblock_task(
        board,
        checker,
        actor="restart-reconciler",
        reason="resume immutable checker verdict",
    )
    reclaimed = kb.claim_task(board, checker, claimer="replacement-checker")
    assert reclaimed is not None and reclaimed.current_run_id is not None
    replacement_run_id = int(reclaimed.current_run_id)
    assert replacement_run_id != run_id

    retry = submit_project_checker_verdict(
        board,
        **{**kwargs, "run_id": replacement_run_id},
    )
    assert retry.disposition == "already_recorded"
    assert retry.completed is True
    assert board.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='project_checker_verdict_recorded'", (checker,)
    ).fetchone()[0] == 1
    assert board.execute(
        "SELECT run_id FROM task_events WHERE task_id=? AND kind='project_checker_verdict_recorded'",
        (checker,),
    ).fetchone()["run_id"] == run_id
