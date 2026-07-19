"""Production-path admission, checker, verdict, delivery and replay proof."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

import gateway.project_finalization as finalization_gateway
from gateway.project_finalization import ProjectFinalizationService
from hermes_cli import kanban_db as kb
from hermes_cli import project_runtime_registration as runtime
from hermes_cli.project_delivery_ledger import list_delivery_attempts
from hermes_cli.project_finalization_contract import (
    _validate_admitted_checker_authority,
    get_project_finalization,
    list_project_members,
    register_project_member,
)
from hermes_cli.project_finalizer import evaluate_project


BOARD = "board-e2e"
CHAT_ID = "-100-production-secret"
THREAD_ID = "42"
CHECKER_PROFILE = "checker-terra"


def _contract() -> dict:
    return {
        "version": 1,
        "scope": "production project finalization e2e",
        "allowed_files": ["src/**"],
        "forbidden_files": [],
        "base_commit": "a" * 40,
        "required_evidence": ["focused lifecycle test"],
        "required_commands": ["scripts/run_tests.sh"],
        "allow_child_creation": False,
        "forbidden_git_actions": ["push"],
        "notification_verified": True,
    }


@pytest.fixture
def production_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        runtime,
        "profile_exists",
        lambda profile: profile in {CHECKER_PROFILE, "builder-gptluna"},
    )
    kb.init_db(db_path)
    conn = kb.connect(db_path)
    try:
        yield conn, db_path, workspace
    finally:
        conn.close()


def test_admitted_project_passes_checker_delivers_once_and_replays_safely(
    production_board,
):
    conn, db_path, workspace = production_board

    # Create the route before contracting the root. The second implementation
    # task inherits that one durable route through the normal parent boundary.
    root = kb.create_task(
        conn,
        title="Ship the admitted project",
        assignee="builder-sol",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    kb.add_notify_sub(
        conn,
        task_id=root,
        platform="telegram",
        chat_id=CHAT_ID,
        thread_id=THREAD_ID,
    )
    kb.set_task_contract(conn, root, _contract())
    implementation = kb.create_task(
        conn,
        title="Implement the bounded change",
        assignee="builder-sol",
        workspace_kind="dir",
        workspace_path=str(workspace),
        parents=(root,),
        contract=_contract(),
    )

    admitted = runtime.admit_existing_project(
        conn,
        board_id=BOARD,
        root_task_id=root,
        required_task_ids=(implementation,),
        checker_profile=CHECKER_PROFILE,
        now=100,
    )
    admission_replay = runtime.admit_existing_project(
        conn,
        board_id=BOARD,
        root_task_id=root,
        required_task_ids=(implementation,),
        checker_profile=CHECKER_PROFILE,
        now=101,
    )

    assert admitted.disposition == runtime.ADMISSION_CREATED
    assert admission_replay.disposition == runtime.ADMISSION_ALREADY_ADMITTED
    assert admitted.admission_key == admission_replay.admission_key
    assert CHAT_ID not in repr(admitted)
    admission_events = [
        event for event in kb.list_events(conn, root) if event.kind == "project_admitted"
    ]
    assert len(admission_events) == 1
    assert CHAT_ID not in json.dumps(admission_events[0].payload, sort_keys=True)
    assert admission_events[0].payload["notification_route_identity"].startswith(
        "subscription:sha256:"
    )

    compatibility_repair = kb.create_task(
        conn,
        title="Compatibility repair membership",
        assignee="builder-sol",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    assert kb.complete_task(
        conn, compatibility_repair, result="compatibility repair complete"
    )
    register_project_member(
        conn,
        board_id=BOARD,
        root_task_id=root,
        generation=1,
        task_id=compatibility_repair,
        membership_kind="repair",
        required=False,
    )

    assert kb.complete_task(conn, root, result="root work complete")
    kb.recompute_ready(conn)
    assert kb.get_task(conn, implementation).status == "ready"
    assert kb.complete_task(
        conn,
        implementation,
        result="implementation complete",
        summary="focused implementation evidence accepted",
    )

    delivery_calls: list[tuple[str, str, str | None, str]] = []
    delivery_race_fences: list[str] = []

    async def accepted_delivery(
        platform: str, chat_id: str, thread_id: str | None, message: str
    ) -> dict[str, str]:
        # The provider call is an async race window. Public candidate mutation
        # must remain fenced from durable PASS through accepted delivery.
        with pytest.raises(kb.ProjectCandidateFrozenError):
            kb.assign_task(conn, root, CHECKER_PROFILE)
        delivery_race_fences.append("assign")
        with pytest.raises(kb.ProjectCandidateFrozenError):
            kb.edit_completed_task_result(
                conn,
                implementation,
                result="mutation attempted during provider delivery",
            )
        delivery_race_fences.append("result")
        with pytest.raises(kb.ProjectCandidateFrozenError):
            kb.assign_task(conn, compatibility_repair, CHECKER_PROFILE)
        delivery_race_fences.append("optional-repair")
        with pytest.raises(sqlite3.IntegrityError, match="terminal candidate is frozen"):
            kb.archive_task(conn, checker_id)
        delivery_race_fences.append("checker")
        for authority_kind in (
            "project_checker_registered",
            "project_checker_verdict_recorded",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="terminal candidate is frozen"):
                conn.execute(
                    "UPDATE task_events SET payload='{}' WHERE task_id=? AND kind=?",
                    (checker_id, authority_kind),
                )
            with pytest.raises(sqlite3.IntegrityError, match="terminal candidate is frozen"):
                conn.execute(
                    "DELETE FROM task_events WHERE task_id=? AND kind=?",
                    (checker_id, authority_kind),
                )
        # Other project bookkeeping remains intentionally mutable: only the
        # two checker authority event kinds are part of the terminal fence.
        conn.execute(
            "UPDATE task_events SET payload='{}' WHERE task_id=? AND kind='project_admitted'",
            (root,),
        )
        # A prior rejected/retry-scheduled ledger entry must not authorize a
        # stale-fence thaw while the current attempt is pending/attempting.
        # This callback runs for both the first and the retried delivery.
        with pytest.raises(sqlite3.IntegrityError, match="terminal authority is frozen"):
            conn.execute(
                "UPDATE project_finalizations SET terminal_intent=NULL, "
                "terminal_candidate_snapshot_version=NULL "
                "WHERE board_id=? AND root_task_id=? AND generation=1",
                (BOARD, root),
            )
        delivery_calls.append((platform, chat_id, thread_id, message))
        if len(delivery_calls) == 1:
            return {"rejected": True, "error": "provider refused first attempt"}
        return {"provider_message_id": "telegram-message-1"}

    clock = {"now": 200}
    service = ProjectFinalizationService(
        lambda: kb.connect(db_path),
        owner="e2e-finalizer",
        now=lambda: clock["now"],
        deliver=accepted_delivery,
        enabled=True,
        canary_scope=(f"{BOARD}/{root}",),
        cleanup_enabled=False,
    )

    checker_tick = asyncio.run(service.tick(board_id=BOARD))
    assert checker_tick.checkers_reconciled == 1
    aggregate = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert aggregate is not None
    checker_id = aggregate.final_checker_task_id
    assert checker_id.startswith("t_")
    checker_task = kb.get_task(conn, checker_id)
    assert checker_task is not None
    assert checker_task.assignee == CHECKER_PROFILE
    # The checker is an independent worker, but it must inspect the same
    # controlled source boundary as the admitted project root.  A scratch
    # workspace here would make required project commands non-runnable.
    assert checker_task.workspace_kind == "dir"
    assert checker_task.workspace_path == str(workspace)
    assert aggregate.checker_profile == CHECKER_PROFILE
    # An admitted row cannot bypass freeze/artifact/delivery sequencing with
    # raw SQL, even before a terminal candidate has been frozen.
    with pytest.raises(sqlite3.IntegrityError, match="terminal transition"):
        conn.execute(
            "UPDATE project_finalizations SET terminal_outcome='COMPLETE', state='complete' "
            "WHERE board_id=? AND root_task_id=? AND generation=1",
            (BOARD, root),
        )

    # A watcher replay while the checker is pending must preserve authority.
    pending_replay = asyncio.run(service.tick(board_id=BOARD))
    assert pending_replay.checkers_reconciled == 0
    assert get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    ).final_checker_task_id == checker_id

    claimed = kb.claim_task(conn, checker_id, claimer="e2e-checker")
    assert claimed is not None
    assert claimed.current_run_id is not None
    evidence = (
        {
            "kind": "test",
            "reference": "scripts/run_tests.sh tests/gateway/test_project_finalization_e2e.py",
            "summary": "production lifecycle passed",
        },
    )
    verdict = runtime.submit_project_checker_verdict(
        conn,
        board_id=BOARD,
        task_id=checker_id,
        run_id=claimed.current_run_id,
        worker_profile=CHECKER_PROFILE,
        verdict="PASS",
        reason="required work and evidence satisfy the admitted contract",
        evidence=evidence,
        summary="independent checker passed the candidate",
        now=210,
    )
    verdict_replay = runtime.submit_project_checker_verdict(
        conn,
        board_id=BOARD,
        task_id=checker_id,
        run_id=claimed.current_run_id,
        worker_profile=CHECKER_PROFILE,
        verdict="PASS",
        reason="required work and evidence satisfy the admitted contract",
        evidence=evidence,
        summary="independent checker passed the candidate",
        now=211,
    )
    assert verdict.disposition == "recorded"
    assert verdict.completed is True
    assert verdict_replay.disposition == "already_recorded"
    assert len(
        [
            event
            for event in kb.list_events(conn, checker_id)
            if event.kind == "project_checker_verdict_recorded"
        ]
    ) == 1

    # Publication/CAS refuse split checker authority even before the terminal
    # fence is acquired.  Each probe rolls back its malformed evidence so the
    # valid lifecycle below remains a real end-to-end flow.
    def assert_authority_probe_rejected(mutator) -> None:
        with pytest.raises(ValueError, match="checker (.*authority|.*identity)"):
            with kb.write_txn(conn):
                mutator()
                row = conn.execute(
                    "SELECT * FROM project_finalizations WHERE board_id=? AND root_task_id=? AND generation=1",
                    (BOARD, root),
                ).fetchone()
                assert row is not None
                _validate_admitted_checker_authority(conn, row)

    assert_authority_probe_rejected(
        lambda: conn.execute(
            "DELETE FROM task_events WHERE task_id=? AND kind='project_checker_registered'",
            (checker_id,),
        )
    )
    assert_authority_probe_rejected(
        lambda: conn.execute(
            "UPDATE task_events SET payload='not-json' WHERE task_id=? AND kind='project_checker_verdict_recorded'",
            (checker_id,),
        )
    )

    def forge_registration_identity() -> None:
        payload = json.loads(
            conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='project_checker_registered'",
                (checker_id,),
            ).fetchone()["payload"]
        )
        payload["checker_identity"] = "checker:sha256:" + "0" * 64
        conn.execute(
            "UPDATE task_events SET payload=? WHERE task_id=? AND kind='project_checker_registered'",
            (json.dumps(payload, sort_keys=True), checker_id),
        )

    assert_authority_probe_rejected(forge_registration_identity)
    assert_authority_probe_rejected(
        lambda: conn.execute(
            "UPDATE tasks SET assignee='forged-checker' WHERE id=?",
            (checker_id,),
        )
    )
    assert_authority_probe_rejected(
        lambda: conn.execute(
            "UPDATE tasks SET idempotency_key='checker:sha256:forged' WHERE id=?",
            (checker_id,),
        )
    )
    assert_authority_probe_rejected(
        lambda: conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "SELECT task_id, run_id, kind, payload, created_at FROM task_events "
            "WHERE task_id=? AND kind='project_checker_verdict_recorded'",
            (checker_id,),
        )
    )

    # A current process rejects supported mutations once PASS freezes the
    # candidate. These calls must not change authority before finalization.
    with pytest.raises(kb.ProjectCandidateFrozenError):
        kb.assign_task(conn, root, CHECKER_PROFILE)
    with pytest.raises(kb.ProjectCandidateFrozenError):
        kb.edit_completed_task_result(
            conn,
            implementation,
            result="changed after checker approval",
        )
    with pytest.raises(kb.ProjectCandidateFrozenError):
        kb.assign_task(conn, compatibility_repair, CHECKER_PROFILE)

    # Simulate an older process reopening required work after PASS. Unfinished
    # implementation takes precedence: no replacement checker may be minted or
    # claimed until the required member is terminal again.
    conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (root,))
    reopened_tick = asyncio.run(service.tick(board_id=BOARD))
    reopened_aggregate = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert reopened_tick.checkers_reconciled == 0
    assert reopened_tick.delivered == reopened_tick.terminalized == 0
    assert reopened_aggregate.final_checker_task_id == checker_id
    assert kb.get_task(conn, checker_id).status == "done"

    # Once required work is terminal again, the stale PASS must rotate exactly
    # one checker onto the changed candidate and still publish nothing.
    conn.execute(
        "UPDATE tasks SET status='done', assignee='builder-terra' WHERE id=?",
        (root,),
    )
    stale_pass_tick = asyncio.run(service.tick(board_id=BOARD))
    stale_aggregate = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert stale_pass_tick.checkers_reconciled == 1
    assert stale_pass_tick.delivered == stale_pass_tick.terminalized == 0
    assert delivery_calls == []
    assert stale_aggregate.checker_verdict is None
    assert stale_aggregate.final_report_path is None
    replacement_checker_id = stale_aggregate.final_checker_task_id
    assert replacement_checker_id != checker_id

    replacement_claim = kb.claim_task(
        conn,
        replacement_checker_id,
        claimer="e2e-replacement-checker",
    )
    assert replacement_claim is not None
    assert replacement_claim.current_run_id is not None
    replacement_verdict = runtime.submit_project_checker_verdict(
        conn,
        board_id=BOARD,
        task_id=replacement_checker_id,
        run_id=replacement_claim.current_run_id,
        worker_profile=CHECKER_PROFILE,
        verdict="PASS",
        reason="changed candidate received a fresh independent review",
        evidence=evidence,
        summary="replacement checker passed the changed candidate",
        now=212,
    )
    assert replacement_verdict.completed is True
    checker_id = replacement_checker_id

    rejected = asyncio.run(service.tick(board_id=BOARD))
    rejected_project = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert rejected.delivered == rejected.terminalized == 0
    assert rejected_project.terminal_outcome is None
    assert rejected_project.final_report_path and rejected_project.manifest_path
    old_manifest = json.loads(
        Path(rejected_project.manifest_path).read_text(encoding="utf-8")
    )
    old_artifacts = {
        path: Path(path).read_bytes()
        for path in (
            rejected_project.final_report_path,
            rejected_project.manifest_path,
            old_manifest["usage_summary_path"],
        )
    }

    # Reproduce a pre-v3 weak-fence writer after rejected delivery.  Keep the
    # stale terminal identity intact: on the next tick schema ensure installs
    # the upgraded authority trigger before registration must perform its
    # narrow, locked recovery thaw.
    conn.execute("DROP TRIGGER pfinal_v3_fence_authority_update")
    conn.execute("DROP TRIGGER pfinal_v2_fence_tasks_update")
    conn.execute(
        "UPDATE tasks SET body='candidate changed after rejected delivery' WHERE id=?",
        (implementation,),
    )
    stale_artifact_tick = asyncio.run(service.tick(board_id=BOARD))
    stale_artifact_project = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert stale_artifact_tick.checkers_reconciled == 1
    assert stale_artifact_tick.delivered == stale_artifact_tick.terminalized == 0
    assert stale_artifact_project.final_report_path is None
    assert stale_artifact_project.manifest_path is None
    assert all(Path(path).read_bytes() == content for path, content in old_artifacts.items())

    final_checker_id = stale_artifact_project.final_checker_task_id
    assert final_checker_id != checker_id
    final_claim = kb.claim_task(conn, final_checker_id, claimer="e2e-final-checker")
    assert final_claim is not None and final_claim.current_run_id is not None
    runtime.submit_project_checker_verdict(
        conn,
        board_id=BOARD,
        task_id=final_checker_id,
        run_id=final_claim.current_run_id,
        worker_profile=CHECKER_PROFILE,
        verdict="PASS",
        reason="post-rejection candidate received fresh independent review",
        evidence=evidence,
        summary="final checker passed the recovered candidate",
        now=213,
    )
    checker_id = final_checker_id
    clock["now"] = 300
    finalized = asyncio.run(service.tick(board_id=BOARD))
    assert finalized.delivered == finalized.terminalized == 1
    terminal = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert terminal is not None
    assert terminal.terminal_outcome == "COMPLETE"
    assert terminal.checker_verdict == "PASS"
    assert len(delivery_calls) == 2
    assert delivery_race_fences == [
        "assign",
        "result",
        "optional-repair",
        "checker",
        "assign",
        "result",
        "optional-repair",
        "checker",
    ]
    assert all(call[:3] == ("telegram", CHAT_ID, THREAD_ID) for call in delivery_calls)
    assert terminal.final_report_path not in old_artifacts
    assert terminal.manifest_path not in old_artifacts
    assert all(Path(path).read_bytes() == content for path, content in old_artifacts.items())

    terminal_replay = asyncio.run(service.tick(board_id=BOARD))
    assert terminal_replay.delivered == 0
    assert terminal_replay.terminalized == 0
    assert len(delivery_calls) == 2
    attempts = list_delivery_attempts(
        conn,
        board_id=BOARD,
        root_task_id=root,
        generation=1,
        platform="telegram",
        destination_reference=CHAT_ID,
        message_kind="project_complete",
    )
    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
    assert attempts[0].delivery_state == "retry_scheduled"
    assert attempts[1].delivery_state == "accepted"

    members = list_project_members(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert [member.task_id for member in members if member.membership_kind == "checker"] == [
        checker_id
    ]
    assert sum(
        event.kind == "project_checker_registered"
        for task in kb.list_tasks(conn, include_archived=True)
        for event in kb.list_events(conn, task.id)
    ) == 3


def _assert_admitted_implementation_non_success(
    production_board, *, scenario: str, expected_outcome: str
) -> None:
    conn, db_path, workspace = production_board
    root = kb.create_task(
        conn, title=f"{scenario} project", assignee="builder-sol",
        workspace_kind="dir", workspace_path=str(workspace),
    )
    kb.add_notify_sub(conn, task_id=root, platform="telegram", chat_id=CHAT_ID, thread_id=THREAD_ID)
    kb.set_task_contract(conn, root, _contract())
    implementation = kb.create_task(
        conn, title="Bounded implementation", assignee="builder-sol",
        workspace_kind="dir", workspace_path=str(workspace), parents=(root,), contract=_contract(),
    )
    runtime.admit_existing_project(
        conn, board_id=BOARD, root_task_id=root, required_task_ids=(implementation,),
        checker_profile=CHECKER_PROFILE, now=900,
    )
    assert kb.complete_task(conn, root, result="root completed")
    kb.recompute_ready(conn)
    if scenario == "internal_failure":
        # Make the bounded retry policy itself terminal, so ordinary checker
        # completion cannot requeue the implementation between verdict and
        # finalization.
        conn.execute("UPDATE tasks SET max_retries=1 WHERE id=?", (implementation,))
    implementation_claim = kb.claim_task(conn, implementation, claimer="implementation-worker")
    assert implementation_claim is not None
    if scenario == "needs_input":
        assert kb.block_task(
            conn, implementation, reason="operator decision required", kind="needs_input",
            expected_run_id=implementation_claim.current_run_id,
        )
    else:
        assert kb._record_task_failure(
            conn, implementation, "worker crashed", outcome="crashed", failure_limit=1,
            release_claim=True, end_run=True,
        )

    deliveries: list[str] = []

    async def accepted_delivery(platform, chat_id, thread_id, message):
        assert (platform, chat_id, thread_id) == ("telegram", CHAT_ID, THREAD_ID)
        deliveries.append(message)
        return {"provider_message_id": f"{scenario}-message"}

    service = ProjectFinalizationService(
        lambda: kb.connect(db_path), owner=f"{scenario}-finalizer", now=lambda: 1000,
        deliver=accepted_delivery, enabled=True, canary_scope=(f"{BOARD}/{root}",),
    )
    first = asyncio.run(service.tick(board_id=BOARD))
    aggregate = get_project_finalization(conn, board_id=BOARD, root_task_id=root, generation=1)
    assert first.checkers_reconciled == 1
    assert first.delivered == first.terminalized == 0
    assert aggregate is not None and aggregate.checker_verdict is None
    checker_claim = kb.claim_task(conn, aggregate.final_checker_task_id, claimer="independent-checker")
    assert checker_claim is not None and checker_claim.current_run_id is not None
    evidence = ({"kind": "test", "reference": f"production-{scenario}", "summary": "terminal candidate reviewed"},)
    for invalid in ("PASS", "FAIL_REPAIRABLE"):
        with pytest.raises(ValueError, match="FAIL_TERMINAL"):
            runtime.submit_project_checker_verdict(
                conn, board_id=BOARD, task_id=aggregate.final_checker_task_id,
                run_id=checker_claim.current_run_id, worker_profile=CHECKER_PROFILE,
                verdict=invalid, reason="must not relabel non-success", evidence=evidence, now=1001,
            )
    verdict_kwargs = {
        "board_id": BOARD,
        "task_id": aggregate.final_checker_task_id,
        "run_id": checker_claim.current_run_id,
        "worker_profile": CHECKER_PROFILE,
        "verdict": "FAIL_TERMINAL",
        "reason": "terminal implementation non-success",
        "evidence": evidence,
        "now": 1002,
    }
    if scenario == "needs_input":
        with pytest.raises(RuntimeError, match="crash after verdict commit"):
            runtime.submit_project_checker_verdict(
                conn,
                inject_failure=lambda stage: (
                    (_ for _ in ()).throw(RuntimeError("crash after verdict commit"))
                    if stage == "after_verdict_commit"
                    else None
                ),
                **verdict_kwargs,
            )
        assert kb.get_task(conn, aggregate.final_checker_task_id).status == "running"
        crash_window = asyncio.run(service.tick(board_id=BOARD))
        assert crash_window.delivered == crash_window.terminalized == 0
        assert crash_window.failures == ()
        assert deliveries == []
    assert runtime.submit_project_checker_verdict(conn, **verdict_kwargs).completed
    current = get_project_finalization(conn, board_id=BOARD, root_task_id=root, generation=1)
    live_evaluation = evaluate_project(
        conn, board_id=BOARD, root_task_id=root, generation=1, evaluation_time=1000
    )
    assert current is not None
    assert current.checker_candidate_snapshot_version == live_evaluation.candidate_snapshot_version
    assert current.checker_candidate_id == live_evaluation.candidate_snapshot_version

    finalized = asyncio.run(service.tick(board_id=BOARD))
    terminal = get_project_finalization(conn, board_id=BOARD, root_task_id=root, generation=1)
    assert finalized.delivered == finalized.terminalized == 1
    assert terminal is not None and terminal.terminal_outcome == expected_outcome
    assert terminal.checker_verdict == "FAIL_TERMINAL"
    assert deliveries == [
        "\n".join((
            f"Result: {expected_outcome}", f"Root: {root}", "Checker: FAIL_TERMINAL",
            "Artifacts: final-report.md, manifest.json, usage-summary.json",
        ))
    ]
    restarted = ProjectFinalizationService(
        lambda: kb.connect(db_path), owner=f"{scenario}-restart", now=lambda: 1010,
        deliver=accepted_delivery, enabled=True, canary_scope=(f"{BOARD}/{root}",),
    )
    replay = asyncio.run(restarted.tick(board_id=BOARD))
    assert replay.delivered == replay.terminalized == 0
    assert len(deliveries) == 1


def test_admitted_needs_input_registers_checker_then_terminalizes_once(production_board):
    _assert_admitted_implementation_non_success(
        production_board, scenario="needs_input", expected_outcome="BLOCKED"
    )


def test_admitted_internal_failure_registers_checker_then_terminalizes_once(production_board):
    _assert_admitted_implementation_non_success(
        production_board, scenario="internal_failure", expected_outcome="FAILED"
    )


def test_stale_pass_cannot_deliver_a_new_non_success_candidate(production_board):
    conn, db_path, workspace = production_board
    root = kb.create_task(
        conn, title="Stale PASS candidate", assignee="builder-sol",
        workspace_kind="dir", workspace_path=str(workspace),
    )
    kb.add_notify_sub(conn, task_id=root, platform="telegram", chat_id=CHAT_ID, thread_id=THREAD_ID)
    kb.set_task_contract(conn, root, _contract())
    implementation = kb.create_task(
        conn, title="Initially successful implementation", assignee="builder-sol",
        workspace_kind="dir", workspace_path=str(workspace), parents=(root,), contract=_contract(),
    )
    runtime.admit_existing_project(
        conn, board_id=BOARD, root_task_id=root, required_task_ids=(implementation,),
        checker_profile=CHECKER_PROFILE, now=1100,
    )
    assert kb.complete_task(conn, root, result="root complete")
    kb.recompute_ready(conn)
    assert kb.complete_task(conn, implementation, result="initial implementation complete")
    deliveries: list[str] = []

    async def accepted_delivery(_platform, _chat_id, _thread_id, message):
        deliveries.append(message)
        return {"provider_message_id": "stale-pass-message"}

    service = ProjectFinalizationService(
        lambda: kb.connect(db_path), owner="stale-pass-finalizer", now=lambda: 1200,
        deliver=accepted_delivery, enabled=True, canary_scope=(f"{BOARD}/{root}",),
    )
    assert asyncio.run(service.tick(board_id=BOARD)).checkers_reconciled == 1
    first = get_project_finalization(conn, board_id=BOARD, root_task_id=root, generation=1)
    assert first is not None
    first_claim = kb.claim_task(conn, first.final_checker_task_id, claimer="first-checker")
    assert first_claim is not None and first_claim.current_run_id is not None
    evidence = ({"kind": "review", "reference": "initial-pass", "summary": "original candidate passed"},)
    assert runtime.submit_project_checker_verdict(
        conn, board_id=BOARD, task_id=first.final_checker_task_id,
        run_id=first_claim.current_run_id, worker_profile=CHECKER_PROFILE, verdict="PASS",
        reason="the original candidate passed", evidence=evidence, now=1201,
    ).completed

    # This required-work mutation precedes every terminal fence and provider
    # acceptance. It makes a new BLOCKED candidate that cannot inherit PASS.
    conn.execute("UPDATE tasks SET status='blocked', block_kind='needs_input' WHERE id=?", (implementation,))
    rotated = asyncio.run(service.tick(board_id=BOARD))
    aggregate = get_project_finalization(conn, board_id=BOARD, root_task_id=root, generation=1)
    assert rotated.checkers_reconciled == 1
    assert rotated.delivered == rotated.terminalized == 0
    assert deliveries == []
    assert aggregate is not None and aggregate.checker_verdict is None
    assert aggregate.final_checker_task_id != first.final_checker_task_id
    fresh_claim = kb.claim_task(conn, aggregate.final_checker_task_id, claimer="fresh-checker")
    assert fresh_claim is not None and fresh_claim.current_run_id is not None
    assert runtime.submit_project_checker_verdict(
        conn, board_id=BOARD, task_id=aggregate.final_checker_task_id,
        run_id=fresh_claim.current_run_id, worker_profile=CHECKER_PROFILE, verdict="FAIL_TERMINAL",
        reason="changed candidate needs external input", evidence=evidence, now=1202,
    ).completed
    finalized = asyncio.run(service.tick(board_id=BOARD))
    terminal = get_project_finalization(conn, board_id=BOARD, root_task_id=root, generation=1)
    assert finalized.delivered == finalized.terminalized == 1
    assert terminal is not None and terminal.terminal_outcome == "BLOCKED"
    assert terminal.checker_verdict == "FAIL_TERMINAL"
    assert len(deliveries) == 1 and "Checker: FAIL_TERMINAL" in deliveries[0]
    assert "Checker: PASS" not in deliveries[0]
    restarted = ProjectFinalizationService(
        lambda: kb.connect(db_path), owner="stale-pass-restart", now=lambda: 1210,
        deliver=accepted_delivery, enabled=True, canary_scope=(f"{BOARD}/{root}",),
    )
    replay = asyncio.run(restarted.tick(board_id=BOARD))
    assert replay.delivered == replay.terminalized == 0
    assert len(deliveries) == 1


def test_admitted_route_survives_root_subscription_gc_before_checker_registration(
    production_board,
):
    conn, db_path, workspace = production_board
    root = kb.create_task(
        conn,
        title="Route durability root",
        assignee="builder-sol",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    kb.add_notify_sub(
        conn,
        task_id=root,
        platform="telegram",
        chat_id=CHAT_ID,
        thread_id=THREAD_ID,
        user_id="private-route-owner",
        notifier_profile="private-route-profile",
    )
    kb.set_task_contract(conn, root, _contract())
    implementation = kb.create_task(
        conn,
        title="Route durability implementation",
        assignee="builder-sol",
        workspace_kind="dir",
        workspace_path=str(workspace),
        parents=(root,),
        contract=_contract(),
    )
    runtime.admit_existing_project(
        conn,
        board_id=BOARD,
        root_task_id=root,
        required_task_ids=(implementation,),
        checker_profile=CHECKER_PROFILE,
        now=600,
    )

    assert kb.complete_task(conn, root, result="root complete")
    # This is the normal notifier cleanup shape: after the root completes, its
    # mutable subscription is removed before the finalizer mints a checker.
    assert kb.remove_notify_sub(
        conn,
        task_id=root,
        platform="telegram",
        chat_id=CHAT_ID,
        thread_id=THREAD_ID,
    )
    kb.recompute_ready(conn)
    assert kb.complete_task(conn, implementation, result="implementation complete")

    async def unexpected_delivery(*_args):
        raise AssertionError("checker registration must not deliver a terminal message")

    service = ProjectFinalizationService(
        lambda: kb.connect(db_path),
        owner="route-durability-finalizer",
        now=lambda: 610,
        deliver=unexpected_delivery,
        enabled=True,
        canary_scope=(f"{BOARD}/{root}",),
    )
    result = asyncio.run(service.tick(board_id=BOARD))

    assert result.checkers_reconciled == 1
    aggregate = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert aggregate is not None
    checker_routes = kb.list_notify_subs(conn, aggregate.final_checker_task_id)
    assert [(row["chat_id"], row["thread_id"]) for row in checker_routes] == [
        (CHAT_ID, THREAD_ID)
    ]
    durable = conn.execute(
        """
        SELECT platform, route_identity FROM project_finalization_notification_routes
         WHERE board_id=? AND root_task_id=? AND generation=1
        """,
        (BOARD, root),
    ).fetchone()
    assert durable is not None
    assert durable["platform"] == "telegram"
    assert durable["route_identity"] == aggregate.notification_route_identity
    assert CHAT_ID not in repr(aggregate)
    # A durable route cannot be redirected or multiplied into another
    # platform. Both cases fail closed at the persistence boundary.
    with pytest.raises(sqlite3.IntegrityError, match="notification route is immutable"):
        conn.execute(
            "UPDATE project_finalization_notification_routes SET chat_id='-100-other' "
            "WHERE board_id=? AND root_task_id=? AND generation=1",
            (BOARD, root),
        )
    with pytest.raises(sqlite3.IntegrityError, match="notification route is immutable"):
        conn.execute(
            "DELETE FROM project_finalization_notification_routes "
            "WHERE board_id=? AND root_task_id=? AND generation=1",
            (BOARD, root),
        )
    with pytest.raises(
        sqlite3.IntegrityError, match="does not match admitted authority"
    ):
        conn.execute(
            """
            INSERT INTO project_finalization_notification_routes (
                board_id, root_task_id, generation, platform, chat_id, thread_id,
                route_identity, created_at
            ) VALUES (?, ?, 1, 'slack', 'other', '', ?, 612)
            """,
            (BOARD, root, aggregate.notification_route_identity),
        )

    checker_claim = kb.claim_task(
        conn, aggregate.final_checker_task_id, claimer="route-durability-checker"
    )
    assert checker_claim is not None and checker_claim.current_run_id is not None
    runtime.submit_project_checker_verdict(
        conn,
        board_id=BOARD,
        task_id=aggregate.final_checker_task_id,
        run_id=checker_claim.current_run_id,
        worker_profile=CHECKER_PROFILE,
        verdict="FAIL_REPAIRABLE",
        reason="one bounded repair proves durable route copying",
        evidence=(
            {
                "kind": "review",
                "reference": "route-durability-repair",
                "summary": "one repair is required",
            },
        ),
        now=611,
    )
    repaired = asyncio.run(service.tick(board_id=BOARD))
    assert repaired.repaired == 1
    repair_member = next(
        member
        for member in list_project_members(
            conn, board_id=BOARD, root_task_id=root, generation=1
        )
        if member.membership_kind == "repair"
    )
    repair_routes = kb.list_notify_subs(conn, repair_member.task_id)
    assert [(row["chat_id"], row["thread_id"]) for row in repair_routes] == [
        (CHAT_ID, THREAD_ID)
    ]


def test_accepted_delivery_restarts_to_terminal_cas_after_root_subscription_gc(
    production_board,
    monkeypatch: pytest.MonkeyPatch,
):
    conn, db_path, workspace = production_board
    root = kb.create_task(
        conn,
        title="Restart durable route root",
        assignee="builder-sol",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    kb.add_notify_sub(
        conn,
        task_id=root,
        platform="telegram",
        chat_id=CHAT_ID,
        thread_id=THREAD_ID,
    )
    kb.set_task_contract(conn, root, _contract())
    implementation = kb.create_task(
        conn,
        title="Restart durable route implementation",
        assignee="builder-sol",
        workspace_kind="dir",
        workspace_path=str(workspace),
        parents=(root,),
        contract=_contract(),
    )
    runtime.admit_existing_project(
        conn,
        board_id=BOARD,
        root_task_id=root,
        required_task_ids=(implementation,),
        checker_profile=CHECKER_PROFILE,
        now=700,
    )
    assert kb.complete_task(conn, root, result="root complete")
    kb.recompute_ready(conn)
    assert kb.complete_task(conn, implementation, result="implementation complete")

    deliveries: list[str] = []

    async def accepted_delivery(
        _platform: str, _chat_id: str, _thread_id: str | None, message: str
    ) -> dict[str, str]:
        deliveries.append(message)
        return {"provider_message_id": "accepted-before-crash"}

    service = ProjectFinalizationService(
        lambda: kb.connect(db_path),
        owner="route-restart-finalizer",
        now=lambda: 710,
        deliver=accepted_delivery,
        enabled=True,
        canary_scope=(f"{BOARD}/{root}",),
    )
    assert asyncio.run(service.tick(board_id=BOARD)).checkers_reconciled == 1
    checker_id = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    ).final_checker_task_id
    claim = kb.claim_task(conn, checker_id, claimer="durable-route-checker")
    assert claim is not None and claim.current_run_id is not None
    runtime.submit_project_checker_verdict(
        conn,
        board_id=BOARD,
        task_id=checker_id,
        run_id=claim.current_run_id,
        worker_profile=CHECKER_PROFILE,
        verdict="PASS",
        reason="independent review passed durable route candidate",
        evidence=(
            {
                "kind": "test",
                "reference": "route-durability-test",
                "summary": "route durability exercised",
            },
        ),
        now=711,
    )

    real_terminal_cas = finalization_gateway.record_terminal_outcome

    def crash_after_accepted_delivery(*_args, **_kwargs):
        raise RuntimeError("simulated crash after accepted delivery")

    monkeypatch.setattr(
        finalization_gateway, "record_terminal_outcome", crash_after_accepted_delivery
    )
    crashed = asyncio.run(service.tick(board_id=BOARD))
    assert crashed.failures
    assert len(deliveries) == 1
    attempt = list_delivery_attempts(
        conn,
        board_id=BOARD,
        root_task_id=root,
        generation=1,
        platform="telegram",
        destination_reference=CHAT_ID,
        message_kind="project_complete",
    )[-1]
    assert attempt.delivery_state == "accepted"
    assert attempt.accepted is True
    assert get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    ).terminal_outcome is None

    assert kb.remove_notify_sub(
        conn,
        task_id=root,
        platform="telegram",
        chat_id=CHAT_ID,
        thread_id=THREAD_ID,
    )
    monkeypatch.setattr(finalization_gateway, "record_terminal_outcome", real_terminal_cas)

    replay = asyncio.run(service.tick(board_id=BOARD))

    assert replay.terminalized == 1
    assert len(deliveries) == 1
    assert get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1).terminal_outcome == "COMPLETE"


def test_repairable_verdict_rotates_authority_and_fresh_checker_can_pass(
    production_board,
):
    conn, db_path, workspace = production_board
    root = kb.create_task(
        conn,
        title="Repairable admitted project",
        assignee="builder-sol",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    kb.add_notify_sub(
        conn,
        task_id=root,
        platform="telegram",
        chat_id=CHAT_ID,
        thread_id=THREAD_ID,
    )
    kb.set_task_contract(conn, root, _contract())
    implementation = kb.create_task(
        conn,
        title="Initial implementation",
        assignee="builder-sol",
        workspace_kind="dir",
        workspace_path=str(workspace),
        parents=(root,),
        contract=_contract(),
    )
    runtime.admit_existing_project(
        conn,
        board_id=BOARD,
        root_task_id=root,
        required_task_ids=(implementation,),
        checker_profile=CHECKER_PROFILE,
        repair_budget=1,
        now=400,
    )
    assert kb.complete_task(conn, root, result="root complete")
    kb.recompute_ready(conn)
    assert kb.complete_task(conn, implementation, result="initial candidate")

    deliveries: list[str] = []

    async def accepted_delivery(
        platform: str, chat_id: str, thread_id: str | None, message: str
    ) -> dict[str, str]:
        deliveries.append(message)
        return {"provider_message_id": "telegram-repair-message-1"}

    clock = iter(range(500, 540))
    service = ProjectFinalizationService(
        lambda: kb.connect(db_path),
        owner="e2e-repair-finalizer",
        now=lambda: next(clock),
        deliver=accepted_delivery,
        enabled=True,
        canary_scope=(f"{BOARD}/{root}",),
    )
    assert asyncio.run(service.tick(board_id=BOARD)).checkers_reconciled == 1
    first_checker = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    ).final_checker_task_id
    first_claim = kb.claim_task(conn, first_checker, claimer="first-checker")
    assert first_claim is not None and first_claim.current_run_id is not None
    repair_evidence = (
        {
            "kind": "review",
            "reference": "review:first-candidate",
            "summary": "one bounded correction is required",
        },
    )
    runtime.submit_project_checker_verdict(
        conn,
        board_id=BOARD,
        task_id=first_checker,
        run_id=first_claim.current_run_id,
        worker_profile=CHECKER_PROFILE,
        verdict="FAIL_REPAIRABLE",
        reason="focused review found one repairable defect",
        evidence=repair_evidence,
        now=510,
    )

    # Even in the narrow verdict-to-repair window, implementation authority
    # may not be reassigned to the persisted checker profile.
    assert kb.assign_task(conn, root, CHECKER_PROFILE)
    fenced_tick = asyncio.run(service.tick(board_id=BOARD))
    assert fenced_tick.repaired == 0
    assert fenced_tick.failures
    assert not [
        member
        for member in list_project_members(
            conn, board_id=BOARD, root_task_id=root, generation=1
        )
        if member.membership_kind == "repair"
    ]
    assert kb.assign_task(conn, root, "builder-sol")

    repair_tick = asyncio.run(service.tick(board_id=BOARD))
    assert repair_tick.repaired == 1
    repairing = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert repairing.repair_generation == 1
    assert repairing.final_checker_task_id.startswith("pending-checker:sha256:")
    members = list_project_members(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    repair_ids = [member.task_id for member in members if member.membership_kind == "repair"]
    assert len(repair_ids) == 1
    repair_id = repair_ids[0]
    assert kb.get_task(conn, repair_id).assignee == "builder-gptluna"
    repair_event = next(
        event
        for event in kb.list_events(conn, repair_id)
        if event.kind == "project_repair_registered"
    )
    assert repair_event.payload["repair_worker_profile"] == "builder-gptluna"
    assert [
        member.task_id
        for member in members
        if member.membership_kind == "support" and member.task_id == first_checker
    ] == [first_checker]

    with pytest.raises(ValueError, match="not the current admitted project checker"):
        runtime.submit_project_checker_verdict(
            conn,
            board_id=BOARD,
            task_id=first_checker,
            run_id=first_claim.current_run_id,
            worker_profile=CHECKER_PROFILE,
            verdict="FAIL_REPAIRABLE",
            reason="focused review found one repairable defect",
            evidence=repair_evidence,
            now=511,
        )

    assert kb.complete_task(conn, repair_id, result="bounded repair complete")
    fresh_tick = asyncio.run(service.tick(board_id=BOARD))
    assert fresh_tick.checkers_reconciled == 1
    fresh = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    second_checker = fresh.final_checker_task_id
    assert second_checker != first_checker
    assert kb.get_task(conn, second_checker).assignee == CHECKER_PROFILE
    second_claim = kb.claim_task(conn, second_checker, claimer="fresh-checker")
    assert second_claim is not None and second_claim.current_run_id is not None
    runtime.submit_project_checker_verdict(
        conn,
        board_id=BOARD,
        task_id=second_checker,
        run_id=second_claim.current_run_id,
        worker_profile=CHECKER_PROFILE,
        verdict="PASS",
        reason="the repaired candidate satisfies the admitted contract",
        evidence=(
            {
                "kind": "test",
                "reference": "review:repaired-candidate",
                "summary": "fresh independent verification passed",
            },
        ),
        now=520,
    )

    final_tick = asyncio.run(service.tick(board_id=BOARD))
    assert final_tick.delivered == 1
    assert final_tick.terminalized == 1
    terminal = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert terminal.terminal_outcome == "COMPLETE"
    assert terminal.checker_verdict == "PASS"
    assert len(deliveries) == 1

    # Every copied ordinary task route would have produced an additional
    # Kanban notification without the project-summary suppression boundary.
    # Claim through the production notifier API after finalization to prove
    # root, implementation, replaced checker, repair, and fresh checker events
    # are all consumed silently and cannot replay beside the one summary.
    suppressed_task_ids = set()
    terminal_kinds = (
        "completed",
        "blocked",
        "gave_up",
        "crashed",
        "timed_out",
        "status",
        "archived",
        "unblocked",
    )
    for task_id in (root, implementation, first_checker, repair_id, second_checker):
        subscriptions = kb.list_notify_subs(conn, task_id)
        assert len(subscriptions) == 1
        sub = subscriptions[0]
        _, _, normal_events, suppressed = kb.claim_notifier_events_for_sub(
            conn,
            task_id=task_id,
            platform=sub["platform"],
            chat_id=sub["chat_id"],
            thread_id=sub["thread_id"],
            kinds=terminal_kinds,
        )
        assert normal_events
        assert suppressed is True
        suppressed_task_ids.add(task_id)
    assert suppressed_task_ids == {
        root,
        implementation,
        first_checker,
        repair_id,
        second_checker,
    }

    all_events = [
        event
        for task in kb.list_tasks(conn, include_archived=True)
        for event in kb.list_events(conn, task.id)
    ]
    assert sum(event.kind == "project_checker_registered" for event in all_events) == 2
    assert sum(event.kind == "project_checker_verdict_recorded" for event in all_events) == 2
