"""Outcome-conditional dependency regression tests."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb


APPROVED_SHA = "a" * 40
OTHER_SHA = "b" * 40


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kb.connect() as connection:
        yield connection


def _complete_review(conn, *, outcome_code: str, subject_sha: str) -> str:
    parent = kb.create_task(conn, title="review", assignee="reviewer")
    assert kb.complete_task(
        conn,
        parent,
        summary=f"review returned {outcome_code}",
        outcome_code=outcome_code,
        subject_sha=subject_sha,
    )
    return parent


def _conditional_child(conn, parent: str, *, subject_sha: str = APPROVED_SHA) -> str:
    child = kb.create_task(conn, title="release", assignee="release")
    kb.link_tasks(
        conn,
        parent,
        child,
        accepted_outcome_codes=["APPROVED"],
        subject_sha=subject_sha,
    )
    return child


def test_reject_is_normal_completion_but_does_not_satisfy_release(conn):
    parent = _complete_review(conn, outcome_code="REJECT", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)

    review = kb.get_task(conn, parent)
    run = kb.latest_run(conn, parent)
    assert review.status == "done"
    assert run.outcome == "completed"
    assert run.outcome_code == "REJECT"
    assert run.subject_sha == APPROVED_SHA
    assert kb.get_task(conn, child).status == "todo"
    assert kb.recompute_ready(conn) == 0
    assert kb.claim_task(conn, child) is None


def test_approved_exact_sha_promotes_and_is_claimable(conn):
    parent = _complete_review(conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)

    assert kb.get_task(conn, child).status == "ready"
    claimed = kb.claim_task(conn, child)
    assert claimed is not None
    assert claimed.status == "running"


@pytest.mark.parametrize(
    ("outcome_code", "subject_sha"),
    [
        (None, APPROVED_SHA),
        ("UNKNOWN", APPROVED_SHA),
        ("APPROVED", None),
        ("APPROVED", "a" * 12),
        ("APPROVED", "A" * 40),
        ("APPROVED", OTHER_SHA),
    ],
)
def test_malformed_or_mismatched_evidence_fails_closed(
    conn, outcome_code, subject_sha
):
    parent = _complete_review(conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA)
    run = kb.latest_run(conn, parent)
    conn.execute(
        "UPDATE task_runs SET outcome_code = ?, subject_sha = ? WHERE id = ?",
        (outcome_code, subject_sha, run.id),
    )
    child = _conditional_child(conn, parent)

    assert kb.get_task(conn, child).status == "todo"
    blockers = kb.dependency_blockers(conn, child)
    assert [blocker["parent_id"] for blocker in blockers] == [parent]


def test_legacy_unconditional_edge_keeps_completion_only_behavior(conn):
    parent = kb.create_task(conn, title="legacy parent")
    child = kb.create_task(conn, title="legacy child", parents=[parent])

    assert kb.complete_task(conn, parent, result="done")
    assert kb.get_task(conn, child).status == "ready"
    assert kb.claim_task(conn, child) is not None


def test_claim_rechecks_evidence_and_demotes_stale_ready_child(conn):
    parent = _complete_review(conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)
    assert kb.get_task(conn, child).status == "ready"

    run = kb.latest_run(conn, parent)
    conn.execute("UPDATE task_runs SET outcome_code = 'REJECT' WHERE id = ?", (run.id,))

    assert kb.claim_task(conn, child) is None
    assert kb.get_task(conn, child).status == "todo"
    event = kb.list_events(conn, child)[-1]
    assert event.kind == "claim_rejected"
    assert event.payload["reason"] == "dependencies_unsatisfied"


def test_candidate_revision_change_invalidates_stale_ready_child(conn):
    parent = _complete_review(conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)
    assert kb.get_task(conn, child).status == "ready"

    conn.execute(
        "UPDATE task_links SET subject_sha = ? WHERE parent_id = ? AND child_id = ?",
        (OTHER_SHA, parent, child),
    )

    assert kb.claim_task(conn, child) is None
    assert kb.get_task(conn, child).status == "todo"
    assert kb.dependency_blockers(conn, child)[0]["reason"] == "subject_sha_mismatch"


def test_reopen_then_archive_does_not_reuse_stale_approval(conn):
    parent = _complete_review(conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)
    assert kb.get_task(conn, child).status == "ready"

    conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (parent,))
    kb._append_event(conn, parent, "status", {"status": "todo"})
    assert kb.archive_task(conn, parent)
    conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))

    assert kb.recompute_ready(conn) == 0
    blockers = kb.dependency_blockers(conn, child)
    assert blockers[0]["reason"] == "stale_evidence_after_reopen"


def test_event_gc_preserves_conditional_approval_provenance(conn):
    parent = _complete_review(conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)
    conn.execute("UPDATE task_events SET created_at = 0 WHERE task_id = ?", (parent,))

    kb.gc_events(conn, older_than_seconds=1)

    assert kb.dependency_blockers(conn, child) == []
    assert kb.get_task(conn, child).status == "ready"


def test_event_gc_bounds_status_provenance_to_first_invalidating_event(conn):
    parent = _complete_review(
        conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA
    )
    child = _conditional_child(conn, parent)
    kb._append_event(conn, parent, "status", {"status": "archived"})
    kb._append_event(conn, parent, "status", {"status": "todo"})
    kb._append_event(conn, parent, "status", {"status": "ready"})
    kb._append_event(conn, parent, "status", {"status": "archived"})
    conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (parent,))
    conn.execute("UPDATE task_events SET created_at = 0 WHERE task_id = ?", (parent,))

    kb.gc_events(conn, older_than_seconds=1)

    status_events = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'status'",
        (parent,),
    ).fetchall()
    assert [json.loads(event["payload"])["status"] for event in status_events] == [
        "todo"
    ]
    assert kb.dependency_blockers(conn, child)[0]["reason"] == (
        "stale_evidence_after_reopen"
    )


def test_event_gc_deletes_status_churn_without_post_completion_invalidation(conn):
    parent = _complete_review(
        conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA
    )
    child = _conditional_child(conn, parent)
    kb._append_event(conn, parent, "status", {"status": "archived"})
    conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (parent,))
    conn.execute("UPDATE task_events SET created_at = 0 WHERE task_id = ?", (parent,))

    kb.gc_events(conn, older_than_seconds=1)

    status_count = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'status'",
        (parent,),
    ).fetchone()[0]
    assert status_count == 0
    assert kb.dependency_blockers(conn, child) == []


def test_event_gc_deletes_status_churn_without_structured_run(conn):
    task = kb.create_task(conn, title="legacy terminal task")
    assert kb.complete_task(conn, task, result="done")
    kb._append_event(conn, task, "status", {"status": "todo"})
    kb._append_event(conn, task, "status", {"status": "ready"})
    kb._append_event(conn, task, "status", {"status": "archived"})
    conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (task,))
    conn.execute("UPDATE task_events SET created_at = 0 WHERE task_id = ?", (task,))

    kb.gc_events(conn, older_than_seconds=1)

    status_count = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'status'",
        (task,),
    ).fetchone()[0]
    assert status_count == 0


def test_review_claim_rechecks_conditional_dependencies(conn):
    parent = _complete_review(conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (child,))
    run = kb.latest_run(conn, parent)
    conn.execute("UPDATE task_runs SET outcome_code = 'REJECT' WHERE id = ?", (run.id,))

    assert kb.claim_review_task(conn, child, claimer="reviewer") is None
    assert kb.get_task(conn, child).status == "review"
    event = kb.list_events(conn, child)[-1]
    assert event.kind == "claim_rejected"
    assert event.payload["source_status"] == "review"


def test_linking_unsatisfied_condition_demotes_ready_child(conn):
    parent = _complete_review(conn, outcome_code="REJECT", subject_sha=APPROVED_SHA)
    child = kb.create_task(conn, title="release")
    assert kb.get_task(conn, child).status == "ready"

    kb.link_tasks(
        conn,
        parent,
        child,
        accepted_outcome_codes=["APPROVED"],
        subject_sha=APPROVED_SHA,
    )

    assert kb.get_task(conn, child).status == "todo"


def test_force_promotion_cannot_bypass_conditional_dependency(conn):
    parent = _complete_review(conn, outcome_code="REJECT", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)

    ok, error = kb.promote_task(conn, child, actor="operator", force=True)

    assert ok is False
    assert "conditional" in error
    assert kb.get_task(conn, child).status == "todo"


def test_promote_force_help_discloses_conditional_dependency_limit():
    root = argparse.ArgumentParser()
    kanban_parser = kanban_cli.build_parser(root.add_subparsers())
    commands = next(
        action
        for action in kanban_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    help_text = " ".join(commands.choices["promote"].format_help().split())

    assert "cannot override conditional dependencies" in help_text


def test_manual_promotion_rechecks_evidence_inside_write_transaction(
    conn, monkeypatch
):
    parent = _complete_review(conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)
    conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))
    run = kb.latest_run(conn, parent)
    original = kb.dependency_blockers
    calls = 0

    def mutate_after_first_check(connection, child_id):
        nonlocal calls
        blockers = original(connection, child_id)
        calls += 1
        if calls == 1:
            connection.execute(
                "UPDATE task_runs SET outcome_code = 'REJECT' WHERE id = ?", (run.id,)
            )
        return blockers

    monkeypatch.setattr(kb, "dependency_blockers", mutate_after_first_check)

    ok, error = kb.promote_task(conn, child, actor="operator")

    assert ok is False
    assert "conditional" in error
    assert calls == 2
    assert kb.get_task(conn, child).status == "todo"


def test_edge_condition_round_trips_as_canonical_json(conn):
    parent = _complete_review(conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA)
    child = _conditional_child(conn, parent)

    row = conn.execute(
        "SELECT accepted_outcome_codes, subject_sha FROM task_links "
        "WHERE parent_id = ? AND child_id = ?",
        (parent, child),
    ).fetchone()
    assert json.loads(row["accepted_outcome_codes"]) == ["APPROVED"]
    assert row["subject_sha"] == APPROVED_SHA


def test_gc_preserves_reopen_history_before_conditional_edge_exists(conn):
    parent = _complete_review(
        conn, outcome_code="APPROVED", subject_sha=APPROVED_SHA
    )
    conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (parent,))
    kb._append_event(conn, parent, "status", {"status": "todo"})
    assert kb.archive_task(conn, parent)
    conn.execute("UPDATE task_events SET created_at = 0 WHERE task_id = ?", (parent,))

    kb.gc_events(conn, older_than_seconds=1)
    child = kb.create_task(conn, title="release")
    kb.link_tasks(
        conn,
        parent,
        child,
        accepted_outcome_codes=["APPROVED"],
        subject_sha=APPROVED_SHA,
    )

    assert kb.get_task(conn, child).status == "todo"
    assert kb.dependency_blockers(conn, child)[0]["reason"] == (
        "stale_evidence_after_reopen"
    )


def test_existing_board_adds_nullable_condition_and_evidence_columns(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    legacy = sqlite3.connect(db_path)
    legacy.executescript(kb.SCHEMA_SQL)
    legacy.execute("ALTER TABLE task_links DROP COLUMN accepted_outcome_codes")
    legacy.execute("ALTER TABLE task_links DROP COLUMN subject_sha")
    legacy.execute("ALTER TABLE task_runs DROP COLUMN outcome_code")
    legacy.execute("ALTER TABLE task_runs DROP COLUMN subject_sha")
    legacy.commit()
    legacy.close()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kb.connect() as migrated:
        link_cols = {
            row["name"] for row in migrated.execute("PRAGMA table_info(task_links)")
        }
        run_cols = {
            row["name"] for row in migrated.execute("PRAGMA table_info(task_runs)")
        }
    assert {"accepted_outcome_codes", "subject_sha"} <= link_cols
    assert {"outcome_code", "subject_sha"} <= run_cols