"""Focused disposable-DB coverage for gateway project finalization."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gateway.project_finalization import ProjectFinalizationService
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_usage_ledger as usage_ledger
from hermes_cli import project_commands
from hermes_cli.project_delivery_ledger import (
    create_delivery_attempt,
    get_latest_delivery_attempt,
    list_delivery_attempts,
    mark_delivery_attempt_attempting,
    mark_delivery_attempt_rejected,
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


def _setup(conn, *, complete_checker=False, verdict=None, subscription=True, root_title="root"):
    root = _task(conn, root_title)
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


def test_cleanup_disabled_terminal_project_has_read_only_retention_preview(board):
    root, _, _ = _setup(board, complete_checker=True, verdict="PASS")
    service, _ = _service(cleanup=False)
    assert _run(service).terminalized == 1
    project = get_project_finalization(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
    )
    assert project.cleanup_after is None
    finalized_at = datetime.fromtimestamp(project.finalized_at, tz=timezone.utc)
    cutoff = finalized_at + timedelta(days=project.retention_days)
    before = board.total_changes

    early = project_commands.project_cleanup_preview(
        board,
        board_id="default",
        root_task_id=root,
        now=cutoff - timedelta(seconds=1),
    )
    boundary = project_commands.project_cleanup_preview(
        board,
        board_id="default",
        root_task_id=root,
        now=cutoff,
    )

    assert early["retention_cutoff"] == cutoff.isoformat()
    assert early["refusal_reasons"] == ["retention_not_expired"]
    assert boundary["eligible"] is True
    assert boundary["refusal_reasons"] == []
    assert board.total_changes == before


def test_terminal_provider_payload_uses_durable_contract_and_excludes_private_data(board):
    private_task_text = "PRIVATE project prompt must not leave Hermes"
    root, _, _ = _setup(
        board,
        complete_checker=True,
        verdict="PASS",
        root_title=private_task_text,
    )
    service, receipts = _service()

    result = _run(service)

    assert result.delivered == result.terminalized == 1
    assert receipts[0] == (
        "telegram",
        "-100-test",
        "4",
        "\n".join(
            (
                "Result: COMPLETE",
                f"Root: {root}",
                "Checker: PASS",
                "Artifacts: final-report.md, manifest.json, usage-summary.json",
            )
        ),
    )
    provider_payload = receipts[0][3]
    assert private_task_text not in provider_payload
    assert "-100-test" not in provider_payload
    assert "C:/repo" not in provider_payload
    assert "action" not in provider_payload.lower()
    assert len(provider_payload) <= 512


def test_delivery_router_send_result_persists_provider_id_through_ledger(board):
    from gateway.config import GatewayConfig, Platform
    from gateway.delivery import DeliveryRouter, DeliveryTarget
    from gateway.kanban_watchers import _project_finalizer_delivery_receipt
    from gateway.platforms.base import SendResult

    class TelegramAdapter:
        async def send(self, chat_id, content, metadata=None):
            return SendResult(success=True, message_id="telegram-ledger-42")

    root, _, _ = _setup(board, complete_checker=True, verdict="PASS")
    router = DeliveryRouter(
        GatewayConfig(),
        adapters={Platform.TELEGRAM: TelegramAdapter()},
    )

    async def deliver(platform, chat_id, thread_id, content):
        target = DeliveryTarget(
            platform=Platform(platform),
            chat_id=chat_id,
            thread_id=thread_id,
            is_explicit=True,
        )
        results = await router.deliver(
            content,
            [target],
            metadata={"source": "project_finalizer"},
        )
        return _project_finalizer_delivery_receipt(results[target.to_string()])

    service = ProjectFinalizationService(
        kb.connect,
        owner="test-owner",
        now=lambda: 100,
        deliver=deliver,
        enabled=True,
        canary_scope=("*",),
    )

    result = _run(service)

    attempt = get_latest_delivery_attempt(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
        platform="telegram",
        destination_reference="-100-test",
        message_kind="project_complete",
    )
    assert result.delivered == result.terminalized == 1
    assert attempt.delivery_state == "accepted"
    assert attempt.provider_message_id == "telegram-ledger-42"
    assert get_project_finalization(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
    ).terminal_outcome == "COMPLETE"


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
    assert receipts[0][3] == "\n".join(
        (
            "Result: BLOCKED",
            f"Root: {root}",
            "Checker: FAIL_TERMINAL",
            "Artifacts: final-report.md, manifest.json, usage-summary.json",
        )
    )
    assert "Checker: PASS" not in receipts[0][3]


def test_failed_required_task_delivers_distinct_failed_payload(board):
    root, _, _ = _setup(board)
    board.execute(
        "UPDATE tasks SET status='blocked', consecutive_failures=1, "
        "last_failure_error='worker crashed' WHERE id=?",
        (root,),
    )
    run_id = board.execute(
        "INSERT INTO task_runs "
        "(task_id, status, started_at, ended_at, outcome, error) "
        "VALUES (?, 'gave_up', 1, 2, 'gave_up', 'worker crashed')",
        (root,),
    ).lastrowid
    usage_ledger.record_run_usage(
        board,
        board="default",
        task_id=root,
        run_id=run_id,
        call_kind="primary",
        api_call_index=0,
        provider="test-provider",
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        token_source="provider_authoritative",
        profile="builder-gptterra",
    )
    service, receipts = _service()

    result = _run(service)

    project = get_project_finalization(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
    )
    assert result.delivered == result.terminalized == 1
    assert project.terminal_outcome == "FAILED"
    assert receipts[0][3] == "\n".join(
        (
            "Result: FAILED",
            f"Root: {root}",
            "Checker: NOT_RECORDED",
            "Artifacts: final-report.md, manifest.json, usage-summary.json",
        )
    )
    manifest = json.loads(Path(project.manifest_path).read_text(encoding="utf-8"))
    usage = json.loads(Path(manifest["usage_summary_path"]).read_text(encoding="utf-8"))
    assert manifest["checker_verdict"] is None
    assert manifest["repair_tasks"] == []
    assert {item["task_id"] for item in manifest["required_tasks"]} == {root}
    assert usage["usage_status"] == "partial"
    assert usage["total_input_tokens"] == 10


@pytest.mark.parametrize(
    ("outcome", "checker_verdict"),
    (("COMPLETE", "PASS"), ("BLOCKED", "FAIL_TERMINAL"), ("FAILED", None)),
)
def test_terminal_snapshot_fences_all_candidate_writers_during_async_delivery(
    board, outcome, checker_verdict
):
    root, _, _ = _setup(
        board,
        complete_checker=checker_verdict is not None,
        verdict=checker_verdict,
    )
    if outcome == "FAILED":
        board.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1, "
            "last_failure_error='worker crashed' WHERE id=?",
            (root,),
        )
        board.execute(
            "INSERT INTO task_runs "
            "(task_id, status, started_at, ended_at, outcome, error) "
            "VALUES (?, 'gave_up', 1, 2, 'gave_up', 'worker crashed')",
            (root,),
        )
    unrelated = _task(board, "unrelated parent")
    before_task = kb.get_task(board, root)
    before_counts = {
        "comments": board.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id=?", (root,)
        ).fetchone()[0],
        "attachments": board.execute(
            "SELECT COUNT(*) FROM task_attachments WHERE task_id=?", (root,)
        ).fetchone()[0],
        "runs": board.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (root,)
        ).fetchone()[0],
        "members": board.execute(
            "SELECT COUNT(*) FROM project_finalization_members WHERE task_id=?",
            (root,),
        ).fetchone()[0],
    }
    fence_hits = []

    def reprioritize_with_event():
        with kb.write_txn(board):
            board.execute("UPDATE tasks SET priority=99 WHERE id=?", (root,))
            board.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'reprioritized', '{}', 100)",
                (root,),
            )

    mutators = (
        ("title", lambda: board.execute("UPDATE tasks SET title='changed' WHERE id=?", (root,))),
        ("body", lambda: board.execute("UPDATE tasks SET body='changed' WHERE id=?", (root,))),
        ("status", lambda: board.execute("UPDATE tasks SET status='todo' WHERE id=?", (root,))),
        ("assign", lambda: kb.assign_task(board, root, "other-profile")),
        ("priority-event", reprioritize_with_event),
        ("comment", lambda: kb.add_comment(board, root, "reviewer", "late note")),
        (
            "attachment",
            lambda: kb.add_attachment(
                board,
                root,
                filename="late.txt",
                stored_path="C:/tmp/late.txt",
            ),
        ),
        ("link", lambda: kb.link_tasks(board, unrelated, root)),
        (
            "run",
            lambda: board.execute(
                "INSERT INTO task_runs (task_id, status, started_at) "
                "VALUES (?, 'running', 100)",
                (root,),
            ),
        ),
        (
            "event",
            lambda: board.execute(
                "INSERT INTO task_events (task_id, kind, created_at) "
                "VALUES (?, 'edited', 100)",
                (root,),
            ),
        ),
        (
            "membership",
            lambda: board.execute(
                "DELETE FROM project_finalization_members WHERE task_id=?", (root,)
            ),
        ),
    )

    async def deliver(*_):
        frozen = get_project_finalization(
            board, board_id="default", root_task_id=root, generation=1
        )
        assert frozen.terminal_intent == outcome
        assert frozen.terminal_candidate_snapshot_version is not None
        for label, mutate in mutators:
            with pytest.raises(
                (kb.ProjectCandidateFrozenError, sqlite3.IntegrityError),
                match="project .*candidate is frozen",
            ):
                mutate()
            fence_hits.append(label)
        return {"provider_message_id": f"accepted-{outcome.lower()}"}

    service = ProjectFinalizationService(
        kb.connect,
        owner="terminal-fence-test",
        now=lambda: 100,
        deliver=deliver,
        enabled=True,
        canary_scope=("*",),
    )
    result = _run(service)

    assert result.delivered == result.terminalized == 1
    assert len(fence_hits) == len(mutators)
    terminal = get_project_finalization(
        board, board_id="default", root_task_id=root, generation=1
    )
    assert terminal.terminal_outcome == outcome
    after_task = kb.get_task(board, root)
    assert (after_task.title, after_task.body, after_task.status, after_task.priority) == (
        before_task.title,
        before_task.body,
        before_task.status,
        before_task.priority,
    )
    assert board.execute(
        "SELECT COUNT(*) FROM task_comments WHERE task_id=?", (root,)
    ).fetchone()[0] == before_counts["comments"]
    assert board.execute(
        "SELECT COUNT(*) FROM task_attachments WHERE task_id=?", (root,)
    ).fetchone()[0] == before_counts["attachments"]
    assert board.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (root,)
    ).fetchone()[0] == before_counts["runs"]
    assert board.execute(
        "SELECT COUNT(*) FROM project_finalization_members WHERE task_id=?", (root,)
    ).fetchone()[0] == before_counts["members"]
    assert board.execute(
        "SELECT COUNT(*) FROM task_links WHERE parent_id=? AND child_id=?",
        (unrelated, root),
    ).fetchone()[0] == 0


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


def test_restart_recovers_durable_rejected_delivery_before_retry(board):
    root, _, _ = _setup(board, complete_checker=True, verdict="PASS")
    identity = dict(
        board_id="default",
        root_task_id=root,
        generation=1,
        platform="telegram",
        destination_reference="-100-test",
        message_kind="project_complete",
    )
    created = create_delivery_attempt(
        board,
        thread_reference="4",
        attempt_number=1,
        **identity,
    )
    mark_delivery_attempt_attempting(
        board,
        attempt_number=created.attempt_number,
        now=100,
        **identity,
    )
    mark_delivery_attempt_rejected(
        board,
        attempt_number=created.attempt_number,
        redacted_error="provider refused before process exit",
        now=100,
        **identity,
    )
    clock = {"now": 100}
    calls = []

    async def accepted(*_):
        calls.append(clock["now"])
        return {"provider_message_id": "m-recovered"}

    def tick_from_restart():
        return _run(
            ProjectFinalizationService(
                kb.connect,
                owner="test-owner",
                now=lambda: clock["now"],
                deliver=accepted,
                enabled=True,
                canary_scope=("*",),
            )
        )

    scheduled = tick_from_restart()
    durable = get_latest_delivery_attempt(board, **identity)
    assert scheduled.delivered == scheduled.terminalized == 0
    assert durable.delivery_state == "retry_scheduled"
    assert durable.next_retry_at == 130
    assert calls == []

    clock["now"] = 130
    completed = tick_from_restart()
    attempts = list_delivery_attempts(board, **identity)
    assert completed.delivered == completed.terminalized == 1
    assert [attempt.delivery_state for attempt in attempts] == [
        "retry_scheduled",
        "accepted",
    ]
    assert calls == [130]


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
