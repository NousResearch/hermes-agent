"""Focused disposable-DB coverage for gateway project finalization."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from gateway.project_finalization import ProjectFinalizationService
from hermes_cli import kanban_db as kb
from hermes_cli.project_delivery_ledger import (
    get_latest_delivery_attempt,
    list_delivery_attempts,
)
from hermes_cli.project_finalization_contract import (
    acquire_finalization_lock,
    create_project_finalization,
    get_project_finalization,
    record_checker_verdict,
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


def _task(conn, title):
    return kb.create_task(conn, title=title, assignee="builder-gptterra", workspace_kind="dir", workspace_path="C:/repo")


def _setup(conn, *, complete_checker=False, verdict=None, subscription=True):
    root = _task(conn, "root")
    checker = _task(conn, "checker")
    project = create_project_finalization(conn, board_id="default", root_task_id=root, final_checker_task_id=checker, repair_budget=1)
    kb.complete_task(conn, root, result="implementation done")
    if complete_checker:
        kb.complete_task(conn, checker, result="checker done")
        if verdict:
            record_checker_verdict(conn, board_id="default", root_task_id=root, generation=1, checker_task_id=checker, verdict=verdict)
    if subscription:
        kb.add_notify_sub(conn, task_id=root, platform="telegram", chat_id="-100-test", thread_id="4")
    return root, checker, project


def _service(*, enabled=True, scope=("*",), cleanup=False, receipts=None):
    receipts = receipts if receipts is not None else []

    async def deliver(platform, chat_id, thread_id, content):
        receipts.append((platform, chat_id, thread_id, content))
        return {"provider_message_id": "m-1"}

    return ProjectFinalizationService(kb.connect, owner="test-owner", now=lambda: 100, deliver=deliver, enabled=enabled, canary_scope=scope, cleanup_enabled=cleanup), receipts


def _run(service):
    return asyncio.run(service.tick(board_id="default"))


def test_disabled_by_default_is_a_noop(board):
    root, _, _ = _setup(board)
    service, receipts = _service(enabled=False)

    result = _run(service)

    assert result.processed == 0
    assert receipts == []
    assert get_project_finalization(board, board_id="default", root_task_id=root, generation=1).terminal_outcome is None


def test_waiting_checker_required_registers_once_and_reuses_identity(board):
    root, _, _ = _setup(board)
    service, receipts = _service()

    first = _run(service)
    second = _run(service)

    members = board.execute("SELECT task_id FROM project_finalization_members WHERE membership_kind='checker'").fetchall()
    assert first.checkers_reconciled == 0
    assert second.checkers_reconciled == 0
    assert len(members) == 1
    assert receipts == []
    assert get_project_finalization(board, board_id="default", root_task_id=root, generation=1).terminal_outcome is None


def test_pass_gates_artifacts_accepted_delivery_terminalization_and_cleanup_schedule(board):
    root, _, _ = _setup(board, complete_checker=True, verdict="PASS")
    service, receipts = _service(cleanup=True)

    result = _run(service)

    project = get_project_finalization(board, board_id="default", root_task_id=root, generation=1)
    attempt = get_latest_delivery_attempt(board, board_id="default", root_task_id=root, generation=1, platform="telegram", destination_reference="-100-test", message_kind="project_complete")
    assert result.delivered == result.terminalized == 1
    assert project.terminal_outcome == "COMPLETE"
    assert project.final_report_path and project.manifest_path and project.cleanup_after
    assert attempt.provider_message_id == "m-1"
    assert len(receipts) == 1


def test_repairable_checker_failure_routes_one_durable_repair(board):
    root, _, _ = _setup(board, complete_checker=True, verdict="FAIL_REPAIRABLE")
    service, receipts = _service()

    result = _run(service)

    repairs = board.execute("SELECT task_id FROM project_finalization_members WHERE membership_kind='repair'").fetchall()
    project = get_project_finalization(board, board_id="default", root_task_id=root, generation=1)
    assert result.repaired == 1
    assert len(repairs) == 1
    assert project.repair_generation == 1
    assert project.terminal_outcome is None
    assert receipts == []


def test_completed_repair_reconciles_one_fresh_checker_and_reuses_it_on_restart(board):
    root, initial_checker, _ = _setup(
        board,
        complete_checker=True,
        verdict="FAIL_REPAIRABLE",
    )
    service, receipts = _service()
    kb.set_task_contract(
        board,
        root,
        {
            "version": 1,
            "scope": "test completed repair checker reconciliation",
            "allowed_files": ["gateway/project_finalization.py"],
            "forbidden_files": [],
            "base_commit": "1" * 40,
            "required_evidence": ["repair result"],
            "required_commands": ["pytest focused lifecycle"],
            "allow_child_creation": False,
            "forbidden_git_actions": ["push"],
            "notification_verified": True,
        },
    )
    assert _run(service).repaired == 1
    repair_task_id = board.execute(
        "SELECT task_id FROM project_finalization_members WHERE membership_kind='repair'"
    ).fetchone()["task_id"]
    assert kb.claim_task(board, repair_task_id) is not None
    assert kb.complete_task(board, repair_task_id, result="repair completed")

    reconciled = _run(service)
    replayed = _run(service)

    project = get_project_finalization(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
    )
    checker_members = board.execute(
        "SELECT task_id FROM project_finalization_members "
        "WHERE membership_kind='checker' AND required=1"
    ).fetchall()
    assert reconciled.checkers_reconciled == 1
    assert replayed.checkers_reconciled == 0
    assert project.final_checker_task_id != initial_checker
    assert [row["task_id"] for row in checker_members] == [project.final_checker_task_id]
    assert board.execute(
        "SELECT COUNT(*) FROM tasks WHERE title='Check project generation 1'"
    ).fetchone()[0] == 1
    assert receipts == []


def test_blocked_checker_publishes_terminal_artifacts_and_delivers_once(board):
    root, _, _ = _setup(board, complete_checker=True, verdict="FAIL_TERMINAL")
    service, receipts = _service()

    result = _run(service)

    project = get_project_finalization(board, board_id="default", root_task_id=root, generation=1)
    assert result.delivered == result.terminalized == 1
    assert project.terminal_outcome == "BLOCKED"
    assert project.final_report_path and project.manifest_path
    assert len(receipts) == 1


def test_ambiguous_receipt_is_persisted_without_blind_resend(board):
    root, _, _ = _setup(board, complete_checker=True, verdict="PASS")
    calls = []

    async def uncertain(*_):
        calls.append(1)
        return {}

    service = ProjectFinalizationService(kb.connect, owner="test-owner", now=lambda: 100, deliver=uncertain, enabled=True, canary_scope=("*",))
    first = _run(service)
    second = _run(service)

    attempt = get_latest_delivery_attempt(board, board_id="default", root_task_id=root, generation=1, platform="telegram", destination_reference="-100-test", message_kind="project_complete")
    assert first.ambiguous == 1
    assert second.ambiguous == 1
    assert attempt.delivery_state == "ambiguous"
    assert calls == [1]
    assert get_project_finalization(board, board_id="default", root_task_id=root, generation=1).terminal_outcome is None


def test_rejected_delivery_schedules_retry_without_terminalizing(board):
    root, _, _ = _setup(board, complete_checker=True, verdict="PASS")

    async def rejected(*_):
        return {"rejected": True, "error": "provider refused"}

    service = ProjectFinalizationService(kb.connect, owner="test-owner", now=lambda: 100, deliver=rejected, enabled=True, canary_scope=("*",))
    _run(service)

    attempt = get_latest_delivery_attempt(board, board_id="default", root_task_id=root, generation=1, platform="telegram", destination_reference="-100-test", message_kind="project_complete")
    assert attempt.delivery_state == "retry_scheduled"
    assert attempt.next_retry_at == 130
    assert get_project_finalization(board, board_id="default", root_task_id=root, generation=1).terminal_outcome is None


def test_rejected_delivery_retries_are_bounded_and_persist_permanent_failure(board):
    root, _, _ = _setup(board, complete_checker=True, verdict="PASS")
    clock = {"now": 100}
    calls = []

    async def rejected(*_):
        calls.append(clock["now"])
        return {"rejected": True, "error": "provider refused"}

    def tick_from_restart():
        service = ProjectFinalizationService(
            kb.connect,
            owner="test-owner",
            now=lambda: clock["now"],
            deliver=rejected,
            enabled=True,
            canary_scope=("*",),
            cleanup_enabled=True,
        )
        return _run(service)

    first = tick_from_restart()
    before_due = tick_from_restart()
    clock["now"] = 130
    second = tick_from_restart()
    repeated_due = tick_from_restart()
    clock["now"] = 250
    third = tick_from_restart()
    after_cap = tick_from_restart()

    attempts = list_delivery_attempts(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
        platform="telegram",
        destination_reference="-100-test",
        message_kind="project_complete",
    )
    project = get_project_finalization(board, board_id="default", root_task_id=root, generation=1)
    assert first.terminalized == before_due.terminalized == second.terminalized == repeated_due.terminalized == third.terminalized == after_cap.terminalized == 0
    assert [attempt.attempt_number for attempt in attempts] == [1, 2, 3]
    assert [attempt.delivery_state for attempt in attempts] == ["retry_scheduled", "retry_scheduled", "permanent_failure"]
    assert calls == [100, 130, 250]
    assert project.terminal_outcome is None
    assert project.cleanup_after is None


def test_accepted_delivery_restart_recovers_missing_terminal_state_without_resend(board):
    root, _, _ = _setup(board, complete_checker=True, verdict="PASS")
    service, calls = _service()
    _run(service)
    board.execute("UPDATE project_finalizations SET terminal_outcome=NULL, state='open', cleanup_after=NULL WHERE root_task_id=?", (root,))

    result = _run(service)

    assert result.terminalized == 1
    assert len(calls) == 1
    assert get_project_finalization(board, board_id="default", root_task_id=root, generation=1).terminal_outcome == "COMPLETE"


def test_stale_owner_is_fenced_before_mutation(board):
    root, _, project = _setup(board, complete_checker=True, verdict="PASS")
    assert acquire_finalization_lock(board, board_id="default", root_task_id=root, generation=project.generation, owner="other", lease_seconds=100, now="100")
    service, receipts = _service()

    result = _run(service)

    assert result.skipped == 1
    assert receipts == []
    assert get_project_finalization(board, board_id="default", root_task_id=root, generation=1).terminal_outcome is None


def test_terminal_generation_does_not_reopen_or_redeliver(board):
    _, _, _ = _setup(board, complete_checker=True, verdict="PASS")
    service, calls = _service()
    _run(service)
    second = _run(service)

    assert second.processed == 0
    assert len(calls) == 1
