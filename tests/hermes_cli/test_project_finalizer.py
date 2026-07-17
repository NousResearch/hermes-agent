"""Focused tests for HOF-010's read-only deterministic project evaluator."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.project_finalization_contract import (
    create_project_finalization,
    record_checker_verdict,
    register_project_member,
)
from hermes_cli.project_finalizer import evaluate_project


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


def _task(conn: sqlite3.Connection, title: str) -> str:
    return kb.create_task(conn, title=title)


def _complete(conn: sqlite3.Connection, task_id: str) -> None:
    assert kb.complete_task(conn, task_id, result="done")


def _project(conn: sqlite3.Connection, *, root: str, checker: str, budget: int = 1):
    return create_project_finalization(
        conn,
        board_id="board-a",
        root_task_id=root,
        final_checker_task_id=checker,
        repair_budget=budget,
    )


def test_unfinished_ancestor_graph_waits_deterministically_without_writes(board):
    parent = _task(board, "parent")
    root = _task(board, "root")
    checker = _task(board, "checker")
    kb.link_tasks(board, parent, root)
    project = _project(board, root=root, checker=checker)
    register_project_member(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
        task_id=checker,
        membership_kind="checker",
        required=True,
    )

    before = board.total_changes
    first = evaluate_project(
        board, board_id="board-a", root_task_id=root, evaluation_time=100
    )
    second = evaluate_project(
        board, board_id="board-a", root_task_id=root, evaluation_time=100
    )

    assert board.total_changes == before
    assert first == second
    assert first.evaluation_state == "WAITING"
    assert first.required_task_ids == tuple(sorted((parent, root, checker)))
    assert first.unfinished_task_ids == tuple(sorted((parent, root, checker)))
    assert first.snapshot_version.startswith("sha256:")


def test_done_implementation_without_checker_pass_waits_for_checker(board):
    root = _task(board, "root")
    checker = _task(board, "checker")
    _project(board, root=root, checker=checker)
    for task_id in (root, checker):
        _complete(board, task_id)

    result = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)

    assert result.evaluation_state == "WAITING"
    assert result.terminal_outcome is None
    assert result.failure_reason == "checker_required"
    assert result.finalization_eligible is False


def test_missing_or_stale_checker_authority_waits_for_current_checker(board):
    root = _task(board, "root")
    checker = _task(board, "checker")
    stale_checker = _task(board, "stale checker")
    project = _project(board, root=root, checker=checker)
    register_project_member(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
        task_id=stale_checker,
        membership_kind="checker",
        required=True,
    )
    for task_id in (root, checker, stale_checker):
        _complete(board, task_id)

    stale = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)
    board.execute("DELETE FROM tasks WHERE id = ?", (checker,))
    missing = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=101)

    assert stale.evaluation_state == "WAITING"
    assert stale.failure_reason == "checker_required"
    assert missing.evaluation_state == "WAITING"
    assert missing.failure_reason == "checker_required"


def test_blocked_current_checker_blocks_with_checker_evidence(board):
    root = _task(board, "root")
    checker = _task(board, "checker")
    project = _project(board, root=root, checker=checker)
    register_project_member(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
        task_id=checker,
        membership_kind="checker",
        required=True,
    )
    _complete(board, root)
    board.execute(
        "UPDATE tasks SET status = 'blocked', block_kind = 'needs_input' WHERE id = ?",
        (checker,),
    )

    result = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)

    assert result.evaluation_state == "BLOCKED"
    assert result.terminal_outcome == "BLOCKED"
    assert result.failure_reason == "checker_blocked"
    assert f"task:{checker}" in result.evidence_references


def test_done_required_graph_and_passed_checker_is_complete_eligible(board):
    parent = _task(board, "parent")
    root = _task(board, "root")
    checker = _task(board, "checker")
    kb.link_tasks(board, parent, root)
    project = _project(board, root=root, checker=checker)
    for task_id in (parent, root, checker):
        _complete(board, task_id)
    record_checker_verdict(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
        checker_task_id=checker,
        verdict="PASS",
    )

    result = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)

    assert result.evaluation_state == "COMPLETE_ELIGIBLE"
    assert result.terminal_outcome == "COMPLETE"
    assert result.finalization_eligible is True
    assert result.repair_eligible is False
    assert result.successful_task_ids == tuple(sorted((parent, root, checker)))
    assert result.checker_task_id == checker
    assert result.checker_verdict == "PASS"


def test_repairable_checker_failure_below_budget_is_repairable(board):
    root = _task(board, "root")
    checker = _task(board, "checker")
    project = _project(board, root=root, checker=checker, budget=1)
    for task_id in (root, checker):
        _complete(board, task_id)
    record_checker_verdict(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
        checker_task_id=checker,
        verdict="FAIL_REPAIRABLE",
    )

    result = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)

    assert result.evaluation_state == "REPAIRABLE"
    assert result.terminal_outcome is None
    assert result.repair_eligible is True
    assert result.finalization_eligible is False


@pytest.mark.parametrize(
    ("verdict", "block_kind", "expected_reason"),
    [
        ("FAIL_TERMINAL", None, "checker_fail_terminal"),
        (None, "needs_input", "external_or_human_block"),
    ],
)
def test_terminal_checker_or_human_block_is_blocked(board, verdict, block_kind, expected_reason):
    root = _task(board, "root")
    checker = _task(board, "checker")
    project = _project(board, root=root, checker=checker)
    for task_id in (root, checker):
        _complete(board, task_id)
    if verdict:
        record_checker_verdict(
            board,
            board_id="board-a",
            root_task_id=root,
            generation=project.generation,
            checker_task_id=checker,
            verdict=verdict,
        )
    else:
        board.execute(
            "UPDATE tasks SET status = 'blocked', block_kind = ? WHERE id = ?",
            (block_kind, root),
        )

    result = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)

    assert result.evaluation_state == "BLOCKED"
    assert result.terminal_outcome == "BLOCKED"
    assert result.failure_reason == expected_reason


def test_unrecovered_terminal_crash_is_failed(board):
    root = _task(board, "root")
    checker = _task(board, "checker")
    _project(board, root=root, checker=checker)
    _complete(board, checker)
    board.execute(
        "UPDATE tasks SET status = 'blocked', consecutive_failures = 1, "
        "last_failure_error = 'worker crashed' WHERE id = ?",
        (root,),
    )
    board.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, error) "
        "VALUES (?, 'gave_up', 1, 2, 'gave_up', 'worker crashed')",
        (root,),
    )

    result = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)

    assert result.evaluation_state == "FAILED"
    assert result.terminal_outcome == "FAILED"
    assert result.failure_reason == "unrecovered_internal_failure"
    assert result.failed_task_ids == (root,)


def test_recoverable_crash_stays_waiting_until_the_retry_budget_is_exhausted(board):
    root = _task(board, "root")
    checker = _task(board, "checker")
    _project(board, root=root, checker=checker)
    board.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, error) "
        "VALUES (?, 'crashed', 1, 2, 'crashed', 'worker crashed')",
        (root,),
    )

    result = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)

    assert result.evaluation_state == "WAITING"
    assert result.failed_task_ids == ()


@pytest.mark.parametrize("malformation", ["missing_root", "cycle", "missing_member", "multiple_active"])
def test_malformed_graphs_are_bounded_findings(board, malformation):
    root = _task(board, "root")
    checker = _task(board, "checker")
    project = _project(board, root=root, checker=checker)

    if malformation == "missing_root":
        board.execute("DELETE FROM tasks WHERE id = ?", (root,))
    elif malformation == "cycle":
        parent = _task(board, "parent")
        board.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (parent, root))
        board.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (root, parent))
    elif malformation == "missing_member":
        register_project_member(
            board,
            board_id="board-a",
            root_task_id=root,
            generation=project.generation,
            task_id="missing-task",
            membership_kind="support",
            required=True,
        )
    else:
        board.execute(
            "INSERT INTO project_finalizations "
            "(board_id, root_task_id, generation, state, final_checker_task_id, "
            "repair_generation, repair_budget, notification_policy, retention_days, "
            "created_at, updated_at, version) "
            "VALUES ('board-a', ?, 2, 'open', ?, 0, 1, 'project_summary', 3, 1, 1, 1)",
            (root, checker),
        )

    result = evaluate_project(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
        evaluation_time=100,
    )

    assert result.evaluation_state == "MALFORMED"
    assert result.failure_reason
    assert result.blocker


def test_optional_support_does_not_block_complete_eligibility(board):
    root = _task(board, "root")
    checker = _task(board, "checker")
    optional = _task(board, "optional support")
    project = _project(board, root=root, checker=checker)
    register_project_member(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
        task_id=optional,
        membership_kind="support",
        required=False,
    )
    for task_id in (root, checker):
        _complete(board, task_id)
    record_checker_verdict(
        board,
        board_id="board-a",
        root_task_id=root,
        generation=project.generation,
        checker_task_id=checker,
        verdict="PASS",
    )

    result = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)

    assert result.evaluation_state == "COMPLETE_ELIGIBLE"
    assert result.optional_task_ids == (optional,)
    assert optional not in result.unfinished_task_ids


def test_evaluation_time_participates_in_snapshot_identity(board):
    root = _task(board, "root")
    checker = _task(board, "checker")
    _project(board, root=root, checker=checker)

    earlier = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=100)
    later = evaluate_project(board, board_id="board-a", root_task_id=root, evaluation_time=101)

    assert earlier.snapshot_version != later.snapshot_version
