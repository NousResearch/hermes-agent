"""Admission and checker-verdict authority tests for G3.

These tests deliberately exercise the persistence boundary directly.  They do
not depend on the gateway loop or a live Hermes profile directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.project_finalization_contract import (
    acquire_finalization_lock,
    get_project_finalization,
    list_project_members,
    record_checker_verdict,
    reopen_project_finalization,
)
from hermes_cli.project_finalizer import evaluate_project
from hermes_cli.project_repair_router import ProjectIdentity, ProjectVersionToken
from hermes_cli.project_runtime_registration import (
    ADMISSION_ALREADY_ADMITTED,
    ADMISSION_CREATED,
    DEFAULT_REPAIR_PROFILE,
    CheckerRegistrationAction,
    admit_existing_project,
    checker_registration_identity,
    notification_route_identity,
    record_project_checker_evidence,
    register_project_checker,
    submit_project_checker_verdict,
    validate_project_checker_evidence,
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
        lambda profile: profile in {"checker-profile", "builder-gptluna"},
    )
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def _task(
    conn,
    title: str,
    *,
    assignee: str = "builder-profile",
    contract: dict | None = None,
    workspace_path: str = "C:/repo",
) -> str:
    return kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        workspace_kind="dir",
        workspace_path=workspace_path,
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


def _sealed_checker_setup(board):
    workspace = Path.home() / "sealed-workspace"
    workspace.mkdir()
    root = _task(board, "sealed root", workspace_path=str(workspace))
    implementation = _task(board, "sealed implementation", workspace_path=str(workspace))
    kb.add_notify_sub(board, task_id=root, platform="telegram", chat_id="-100-private", thread_id="42")
    kb.recompute_ready(board)
    admitted = admit_existing_project(
        board,
        board_id="board-a",
        root_task_id=root,
        required_task_ids=(implementation,),
        checker_profile="checker-profile",
        repair_profile=DEFAULT_REPAIR_PROFILE,
        now=100,
    )
    assert admitted.finalization.sealed_evidence_required is True
    board.execute(
        "UPDATE tasks SET status='ready' WHERE id IN (?, ?)",
        (root, implementation),
    )
    assert kb.complete_task(board, root, result="root complete")
    kb.recompute_ready(board)
    assert kb.complete_task(board, implementation, result="implementation complete")
    root_path = workspace / "root.txt"
    implementation_path = workspace / "implementation.txt"
    root_path.write_text("root evidence", encoding="utf-8")
    implementation_path.write_text("implementation evidence", encoding="utf-8")
    kb.add_attachment(board, root, filename=root_path.name, stored_path=str(root_path), size=root_path.stat().st_size)
    kb.add_attachment(
        board,
        implementation,
        filename=implementation_path.name,
        stored_path=str(implementation_path),
        size=implementation_path.stat().st_size,
    )
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
    evaluation = evaluate_project(
        board, board_id="board-a", root_task_id=root, generation=1, evaluation_time=101
    )
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
        notification_route_identities=(finalization.notification_route_identity,),
    )
    token = ProjectVersionToken(evaluation.snapshot_version, finalization.version, "test-lock")
    return root, implementation, root_path, implementation_path, evaluation, action, token


def _sealed_evidence(root, implementation, root_path, implementation_path, candidate):
    return [
        {
            "kind": "file",
            "reference": root_path.name,
            "summary": "root artifact",
            "task_id": root,
            "path": root_path.name,
            "sha256": hashlib.sha256(root_path.read_bytes()).hexdigest(),
            "candidate_snapshot_version": candidate,
        },
        {
            "kind": "file",
            "reference": implementation_path.name,
            "summary": "implementation artifact",
            "task_id": implementation,
            "path": implementation_path.name,
            "sha256": hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
            "candidate_snapshot_version": candidate,
        },
    ]


def test_sealed_evidence_fails_closed_before_checker_creation_and_accepts_current_candidate(board):
    root, implementation, root_path, implementation_path, evaluation, action, token = _sealed_checker_setup(board)
    with pytest.raises(ValueError, match="evidence_missing"):
        register_project_checker(board, action, token, now=101)

    candidate = evaluation.candidate_snapshot_version
    valid_root_hash = hashlib.sha256(root_path.read_bytes()).hexdigest()
    invalid_cases = (
        (
            "wrong_task",
            {
                "kind": "file", "reference": "root.txt", "summary": "wrong task",
                "task_id": "not-a-member", "path": "root.txt", "sha256": valid_root_hash,
                "candidate_snapshot_version": candidate,
            },
        ),
        (
            "hash_invalid",
            {
                "kind": "file", "reference": "root.txt", "summary": "bad hash",
                "task_id": root, "path": "root.txt", "sha256": "0" * 63,
                "candidate_snapshot_version": candidate,
            },
        ),
        (
            "path_escape",
            {
                "kind": "file", "reference": "escape", "summary": "escaping path",
                "task_id": root, "path": "../escape", "sha256": valid_root_hash,
                "candidate_snapshot_version": candidate,
            },
        ),
    )
    for reason, item in invalid_cases:
        with pytest.raises(ValueError, match=reason):
            record_project_checker_evidence(
                board,
                board_id="board-a",
                root_task_id=root,
                generation=1,
                candidate_snapshot_version=candidate,
                evidence=[item],
                now=101,
            )

    assert board.execute("SELECT COUNT(*) FROM tasks WHERE idempotency_key=?", (action.idempotency_key,)).fetchone()[0] == 0
    evidence = _sealed_evidence(root, implementation, root_path, implementation_path, candidate)
    assert record_project_checker_evidence(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=1,
        candidate_snapshot_version=candidate,
        evidence=evidence,
        now=101,
    ) == 2
    validate_project_checker_evidence(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=1,
        candidate_snapshot_version=candidate,
    )
    registered = register_project_checker(board, action, token, now=101)
    assert registered.checker_task_id


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


def test_admission_persists_builder_repair_authority_and_replay_is_stable(board):
    root, implementation, created = _admitted(board)
    finalization = get_project_finalization(board, board_id="board-a", root_task_id=root)
    assert finalization is not None
    assert finalization.repair_worker_profile == DEFAULT_REPAIR_PROFILE

    replay = admit_existing_project(
        board,
        board_id="board-a",
        root_task_id=root,
        required_task_ids=(implementation,),
        checker_profile="checker-profile",
        now=101,
    )
    assert replay.disposition == ADMISSION_ALREADY_ADMITTED
    assert replay.admission_key == created.admission_key
    assert replay.finalization.repair_worker_profile == DEFAULT_REPAIR_PROFILE

    with pytest.raises(ValueError, match="repair worker authority"):
        admit_existing_project(
            board,
            board_id="board-a",
            root_task_id=root,
            required_task_ids=(implementation,),
            checker_profile="checker-profile",
            repair_profile="checker-profile",
        )


@pytest.mark.parametrize(
    ("persisted_profile", "error"),
    [(None, "no durable repair worker authority"), ("missing-profile", "unavailable")],
)
def test_admission_replay_rejects_missing_or_unavailable_repair_authority(
    board, persisted_profile, error
):
    root, implementation, _ = _admitted(board)
    board.execute(
        "UPDATE project_finalizations SET repair_worker_profile=? "
        "WHERE board_id='board-a' AND root_task_id=? AND generation=1",
        (persisted_profile, root),
    )

    with pytest.raises(ValueError, match=error):
        admit_existing_project(
            board,
            board_id="board-a",
            root_task_id=root,
            required_task_ids=(implementation,),
            checker_profile="checker-profile",
        )


def test_admission_exact_replay_uses_durable_route_after_root_notifier_gc(board):
    root, implementation, created = _admitted(board)
    route_before = board.execute(
        """
        SELECT route_identity FROM project_finalization_notification_routes
         WHERE board_id='board-a' AND root_task_id=? AND generation=1
        """,
        (root,),
    ).fetchall()
    counts_before = {
        "projects": board.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0],
        "members": board.execute("SELECT COUNT(*) FROM project_finalization_members").fetchone()[0],
        "routes": board.execute(
            "SELECT COUNT(*) FROM project_finalization_notification_routes"
        ).fetchone()[0],
    }
    assert len(route_before) == 1
    assert kb.remove_notify_sub(
        board,
        task_id=root,
        platform="telegram",
        chat_id="-100-private",
        thread_id="42",
    )

    replay = admit_existing_project(
        board,
        board_id="board-a",
        root_task_id=root,
        required_task_ids=(implementation,),
        checker_profile="checker-profile",
        now=101,
    )

    route_after = board.execute(
        """
        SELECT route_identity FROM project_finalization_notification_routes
         WHERE board_id='board-a' AND root_task_id=? AND generation=1
        """,
        (root,),
    ).fetchall()
    counts_after = {
        "projects": board.execute("SELECT COUNT(*) FROM project_finalizations").fetchone()[0],
        "members": board.execute("SELECT COUNT(*) FROM project_finalization_members").fetchone()[0],
        "routes": board.execute(
            "SELECT COUNT(*) FROM project_finalization_notification_routes"
        ).fetchone()[0],
    }
    assert replay.disposition == ADMISSION_ALREADY_ADMITTED
    assert replay.admission_key == created.admission_key
    assert route_after == route_before
    assert counts_after == counts_before


def test_admitted_project_reopen_fails_closed_without_explicit_readmission(board):
    root, _, _ = _admitted(board)
    # Reopen must also reject a historical admitted terminal row.  Construct
    # that legacy state directly; current raw terminalization is intentionally
    # fenced and is covered by the gateway lifecycle regression.
    board.execute("DROP TRIGGER pfinal_v2_fence_terminal_update")
    board.execute(
        "UPDATE project_finalizations SET terminal_outcome='BLOCKED', state='blocked' "
        "WHERE board_id='board-a' AND root_task_id=? AND generation=1",
        (root,),
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
