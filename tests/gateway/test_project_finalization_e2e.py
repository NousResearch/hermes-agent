"""Production-path admission, checker, verdict, delivery and replay proof."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from gateway.project_finalization import ProjectFinalizationService
from hermes_cli import kanban_db as kb
from hermes_cli import project_runtime_registration as runtime
from hermes_cli.project_delivery_ledger import list_delivery_attempts
from hermes_cli.project_finalization_contract import (
    get_project_finalization,
    list_project_members,
)


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
    monkeypatch.setattr(runtime, "profile_exists", lambda profile: profile == CHECKER_PROFILE)
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

    async def accepted_delivery(
        platform: str, chat_id: str, thread_id: str | None, message: str
    ) -> dict[str, str]:
        delivery_calls.append((platform, chat_id, thread_id, message))
        return {"provider_message_id": "telegram-message-1"}

    clock = iter(range(200, 220))
    service = ProjectFinalizationService(
        lambda: kb.connect(db_path),
        owner="e2e-finalizer",
        now=lambda: next(clock),
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
    assert kb.get_task(conn, checker_id).assignee == CHECKER_PROFILE
    assert aggregate.checker_profile == CHECKER_PROFILE

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

    finalized = asyncio.run(service.tick(board_id=BOARD))
    assert finalized.delivered == 1
    assert finalized.terminalized == 1
    terminal = get_project_finalization(
        conn, board_id=BOARD, root_task_id=root, generation=1
    )
    assert terminal is not None
    assert terminal.terminal_outcome == "COMPLETE"
    assert terminal.checker_verdict == "PASS"
    assert len(delivery_calls) == 1
    assert delivery_calls[0][:3] == ("telegram", CHAT_ID, THREAD_ID)

    terminal_replay = asyncio.run(service.tick(board_id=BOARD))
    assert terminal_replay.delivered == 0
    assert terminal_replay.terminalized == 0
    assert len(delivery_calls) == 1
    attempts = list_delivery_attempts(
        conn,
        board_id=BOARD,
        root_task_id=root,
        generation=1,
        platform="telegram",
        destination_reference=CHAT_ID,
        message_kind="project_complete",
    )
    assert len(attempts) == 1
    assert attempts[0].delivery_state == "accepted"

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
    ) == 1


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
    assert kb.get_task(conn, repair_id).assignee == "builder-sol"
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
    all_events = [
        event
        for task in kb.list_tasks(conn, include_archived=True)
        for event in kb.list_events(conn, task.id)
    ]
    assert sum(event.kind == "project_checker_registered" for event in all_events) == 2
    assert sum(event.kind == "project_checker_verdict_recorded" for event in all_events) == 2
