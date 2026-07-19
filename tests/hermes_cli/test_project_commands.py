"""Focused tests for read-only Kanban project commands and presentation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import project_commands as commands
from hermes_cli import project_delivery_ledger as delivery
from hermes_cli import kanban as kanban_cli
from hermes_cli.project_finalization_contract import (
    create_project_finalization,
    record_checker_verdict,
    record_cleanup_journal,
    record_failure_envelope,
    record_final_artifacts,
    record_terminal_outcome,
    register_project_member,
    schedule_project_cleanup,
)


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    conn = kb.connect(db_path=tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _project(conn: sqlite3.Connection, *, title: str = "project goal"):
    root = kb.create_task(conn, title=title, body="prompt: must not be rendered")
    checker = kb.create_task(conn, title="final checker")
    finalization = create_project_finalization(
        conn,
        board_id="default",
        root_task_id=root,
        final_checker_task_id=checker,
        repair_budget=1,
    )
    register_project_member(
        conn,
        board_id="default",
        root_task_id=root,
        generation=finalization.generation,
        task_id=checker,
        membership_kind="checker",
        required=True,
    )
    return root, checker, finalization


def _complete(conn: sqlite3.Connection, task_id: str) -> None:
    assert kb.complete_task(conn, task_id, result="worker response body")


def _accepted_delivery(conn: sqlite3.Connection, root: str, generation: int = 1) -> None:
    delivery.start_delivery_attempt(
        conn,
        board_id="default",
        root_task_id=root,
        generation=generation,
        platform="telegram",
        destination_reference="chat:123",
        thread_reference="thread:1",
        message_kind="summary",
        attempt_number=1,
    )
    delivery.mark_delivery_attempt_accepted(
        conn,
        board_id="default",
        root_task_id=root,
        generation=generation,
        platform="telegram",
        destination_reference="chat:123",
        message_kind="summary",
        attempt_number=1,
        provider_message_id="provider-message-1",
        now=100,
    )


def _full_terminal_project(conn: sqlite3.Connection, tmp_path):
    root, checker, finalization = _project(conn)
    _complete(conn, root)
    _complete(conn, checker)
    record_checker_verdict(
        conn,
        board_id="default",
        root_task_id=root,
        generation=finalization.generation,
        checker_task_id=checker,
        verdict="PASS",
    )
    report = tmp_path / "final-report.md"
    manifest = tmp_path / "manifest.json"
    usage_summary = tmp_path / "usage-summary.json"
    report.write_text("report", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    usage_summary.write_text("{}", encoding="utf-8")
    record_final_artifacts(
        conn,
        board_id="default",
        root_task_id=root,
        generation=1,
        report_path=str(report),
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        manifest_path=str(manifest),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        usage_summary_json="usage-summary-1",
    )
    record_terminal_outcome(conn, board_id="default", root_task_id=root, generation=1, outcome="COMPLETE")
    schedule_project_cleanup(
        conn,
        board_id="default",
        root_task_id=root,
        generation=1,
        cleanup_after="1970-01-01T00:00:00+00:00",
    )
    _accepted_delivery(conn, root)
    return root, checker, finalization, report, manifest


def test_active_list_excludes_terminal_clutter_and_is_stably_ordered(board):
    active_root, _, _ = _project(board, title="active")
    terminal_root, terminal_checker, terminal = _project(board, title="terminal")
    _complete(board, terminal_root)
    _complete(board, terminal_checker)
    record_checker_verdict(
        board,
        board_id="default",
        root_task_id=terminal_root,
        generation=terminal.generation,
        checker_task_id=terminal_checker,
        verdict="PASS",
    )
    record_terminal_outcome(
        board,
        board_id="default",
        root_task_id=terminal_root,
        generation=terminal.generation,
        outcome="COMPLETE",
    )

    first = commands.list_active_projects(board, board_id="default")
    second = commands.list_active_projects(board, board_id="default")

    assert first == second
    assert [item["root_task_id"] for item in first["projects"]] == [active_root]
    assert first["projects"][0]["required_progress"] == {"completed": 0, "total": 2}


def test_status_reports_evaluator_progress_and_active_checker(board):
    root, checker, finalization = _project(board)
    _complete(board, root)

    result = commands.project_status(board, board_id="default", root_task_id=root)

    assert result["evaluator"]["state"] == "WAITING"
    assert result["evaluator"]["required_progress"] == {"completed": 1, "total": 2}
    assert result["active_checker"]["task_id"] == checker
    assert result["active_checker"]["status"] == "ready"
    assert result["finalization"]["generation"] == finalization.generation


def test_terminal_complete_status_uses_durable_outcome_not_active_generation_validation(board, tmp_path):
    root, _, _, _, _ = _full_terminal_project(board, tmp_path)

    result = commands.project_status(board, board_id="default", root_task_id=root)
    shown = commands.project_show(board, board_id="default", root_task_id=root)
    rendered = commands.render_project_result(result, mode="status")

    assert result["evaluator"]["state"] == "COMPLETE"
    assert result["evaluator"]["terminal_outcome"] == "COMPLETE"
    assert result["evaluator"]["required_progress"] == {"completed": 2, "total": 2}
    assert result["blocker"] is None
    assert result["next_action"] == "review terminal project history"
    assert "Evaluator: COMPLETE" in rendered
    assert "MALFORMED" not in rendered
    assert "prompt: must not be rendered" not in json.dumps(shown, sort_keys=True)
    assert "chat:123" not in json.dumps(shown, sort_keys=True)


def test_repairable_verdict_and_active_repair_are_rendered(board):
    root, checker, finalization = _project(board)
    repair = kb.create_task(board, title="repair worker")
    register_project_member(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
        task_id=repair,
        membership_kind="repair",
        required=False,
    )
    _complete(board, root)
    _complete(board, checker)
    record_checker_verdict(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
        checker_task_id=checker,
        verdict="FAIL_REPAIRABLE",
    )

    result = commands.project_status(board, board_id="default", root_task_id=root)
    text = commands.render_project_result(result, mode="status")

    assert result["evaluator"]["state"] == "REPAIRABLE"
    assert result["active_repair"]["task_id"] == repair
    assert "Checker verdict: FAIL_REPAIRABLE" in text
    assert "Active repair: " + repair in text


def test_final_report_lookup_includes_paths_hashes_and_missing_accuracy(board, tmp_path):
    root, _, _, report, manifest = _full_terminal_project(board, tmp_path)

    result = commands.project_final_report(board, board_id="default", root_task_id=root)

    assert result["report"]["exists"] is True
    assert result["report"]["hash_matches"] is True
    assert result["manifest"]["exists"] is True
    assert result["manifest"]["hash_matches"] is True
    assert result["report"]["path"] == str(report)
    assert result["manifest"]["path"] == str(manifest)

    report.unlink()
    missing = commands.project_final_report(board, board_id="default", root_task_id=root)
    assert missing["report"]["exists"] is False
    assert missing["report"]["hash_matches"] is False


def test_delivery_status_keeps_technical_outcome_and_provider_identity_separate(board, tmp_path):
    root, _, _, _, _ = _full_terminal_project(board, tmp_path)

    result = commands.project_delivery_status(board, board_id="default", root_task_id=root)

    assert result["technical_project_outcome"] == "COMPLETE"
    assert result["delivery_state"] == "accepted"
    assert result["provider_message_id"] == "provider-message-1"
    assert result["retry_state"] == "none"
    assert result["ambiguity"] is False


def test_delivery_ambiguity_and_retry_state_are_explicit(board):
    root, _, finalization = _project(board)
    delivery.start_delivery_attempt(
        board,
        board_id="default",
        root_task_id=root,
        generation=finalization.generation,
        platform="telegram",
        destination_reference="chat:123",
        thread_reference=None,
        message_kind="summary",
        attempt_number=1,
    )
    delivery.mark_delivery_attempt_ambiguous(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
        platform="telegram",
        destination_reference="chat:123",
        message_kind="summary",
        attempt_number=1,
        redacted_error="provider timeout; retry requires operator resolution",
        now=100,
    )

    result = commands.project_delivery_status(board, board_id="default", root_task_id=root)

    assert result["delivery_state"] == "ambiguous"
    assert result["ambiguity"] is True
    assert result["retry_state"] == "none"
    assert result["technical_project_outcome"] == "nonterminal"


def test_cleanup_preview_is_stable_and_read_only(board, tmp_path):
    root, checker, _, _, _ = _full_terminal_project(board, tmp_path)
    before = board.total_changes
    before_rows = tuple(
        board.execute(
            "SELECT state, terminal_outcome, cleanup_after FROM project_finalizations WHERE root_task_id = ?",
            (root,),
        ).fetchone()
    )

    first = commands.project_cleanup_preview(
        board,
        board_id="default",
        root_task_id=root,
        now=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )
    second = commands.project_cleanup_preview(
        board,
        board_id="default",
        root_task_id=root,
        now=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )

    assert first == second
    assert first["eligible"] is True
    assert [action["task_id"] for action in first["actions"]] == sorted((root, checker))
    assert all(action["action"] == "archive" for action in first["actions"])
    assert board.total_changes == before
    assert tuple(
        board.execute(
            "SELECT state, terminal_outcome, cleanup_after FROM project_finalizations WHERE root_task_id = ?",
            (root,),
        ).fetchone()
    ) == before_rows


def test_history_includes_archived_zero_run_cards_failures_delivery_and_journal(board):
    root, checker, finalization = _project(board)
    kb.archive_task(board, checker)
    record_failure_envelope(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
        task_id=root,
        run_id=7,
        provider="openai",
        model="model-a",
        failure_class="provider_timeout",
        status_code=504,
        retry_after=30,
        redacted_error="temporary network failure",
    )
    _accepted_delivery(board, root)
    record_cleanup_journal(
        board,
        board_id="default",
        root_task_id=root,
        generation=1,
        plan_sha256="0" * 64,
        mode="preview",
        status="scheduled",
    )

    history = commands.project_history(board, board_id="default", root_task_id=root)

    assert any(item["task_id"] == checker for item in history["archived_zero_run_cards"])
    assert history["delivery_attempts"][0]["provider_message_id"] == "provider-message-1"
    assert history["failure_envelopes"][0]["failure_class"] == "provider_timeout"
    assert history["cleanup_journal"][0]["status"] == "scheduled"


def test_missing_and_malformed_projects_fail_closed(board):
    missing = commands.project_status(board, board_id="default", root_task_id="missing-root")
    assert missing == {
        "ok": False,
        "found": False,
        "error": "project_not_found",
        "root_task_id": "missing-root",
    }

    root, _, _ = _project(board)
    board.execute("DELETE FROM tasks WHERE id = ?", (root,))
    malformed = commands.project_status(board, board_id="default", root_task_id=root)
    assert malformed["evaluator"]["state"] == "MALFORMED"
    assert malformed["evaluator"]["finalization_eligible"] is False
    assert malformed["blocker"]


def test_show_is_concise_and_privacy_safe(board):
    root, _, _ = _project(board, title="safe goal")
    result = commands.project_show(board, board_id="default", root_task_id=root)
    text = commands.render_project_result(result, mode="show")

    assert "safe goal" in text
    assert "prompt: must not be rendered" not in text
    assert "worker response body" not in text
    assert len(text.splitlines()) < 30


def test_existing_run_slash_path_dispatches_project_command(board, monkeypatch, capsys):
    root, _, _ = _project(board)
    db_path = board.execute("PRAGMA database_list").fetchone()[2]
    monkeypatch.setenv("HERMES_KANBAN_DB", db_path)

    output = kanban_cli.run_slash("project status " + root + " --json")

    payload = json.loads(output)
    assert payload["identity"]["root_task_id"] == root
    assert payload["evaluator"]["state"] == "WAITING"
    assert capsys.readouterr().out == ""


def test_parser_registers_all_nested_project_commands():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    kanban_cli.build_parser(subparsers)

    commands_to_parse = [
        ["kanban", "project", "admit", "root", "--required-task", "impl", "--checker-profile", "reviewer"],
        ["kanban", "project", "list-active"],
        ["kanban", "project", "status", "root"],
        ["kanban", "project", "show", "root"],
        ["kanban", "project", "history", "root"],
        ["kanban", "project", "final-report", "root"],
        ["kanban", "project", "delivery-status", "root"],
        ["kanban", "project", "cleanup-preview", "root"],
    ]
    parsed = [parser.parse_args(argv) for argv in commands_to_parse]
    assert [item.project_action for item in parsed] == [
        "admit",
        "list-active",
        "status",
        "show",
        "history",
        "final-report",
        "delivery-status",
        "cleanup-preview",
    ]
    admitted = parsed[0]
    assert admitted.required_task_ids == ["impl"]
    assert admitted.checker_profile == "reviewer"
    assert admitted.repair_budget == 1
    assert admitted.notification_policy == "project_summary"


def test_read_commands_do_not_change_project_or_history_tables(board):
    root, _, _ = _project(board)
    before = {
        table: tuple(board.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall())
        for table in (
            "project_finalizations",
            "project_finalization_members",
            "project_delivery_attempts",
            "project_failure_envelopes",
            "project_cleanup_journal",
        )
    }
    commands.project_status(board, board_id="default", root_task_id=root)
    commands.project_history(board, board_id="default", root_task_id=root)
    commands.project_cleanup_preview(board, board_id="default", root_task_id=root)
    after = {
        table: tuple(board.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall())
        for table in before
    }
    assert after == before
