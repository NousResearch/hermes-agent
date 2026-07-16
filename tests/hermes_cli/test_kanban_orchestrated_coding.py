"""DB-level contracts for Kanban Orchestrated Coding slice 3."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn: sqlite3.Connection, task_id: str, kind: str) -> list[kb.Event]:
    return [event for event in kb.list_events(conn, task_id) if event.kind == kind]


def _count_tasks_with_key(conn: sqlite3.Connection, key: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key = ?",
        (key,),
    ).fetchone()
    return int(row["n"]) if row else 0


def test_verdict_on_auditor_task_id_does_not_open_executor_gate(kanban_home):
    with kb.connect() as conn:
        executor = kb.create_task(
            conn,
            title="execute guarded change",
            assignee="executor",
            plan_audit_required=True,
        )
        auditor = kb.create_task(
            conn,
            title="audit executor plan",
            assignee="plan-auditor",
        )

        kb.record_plan_audit_verdict(
            conn,
            auditor,
            approved=True,
            reviewer="plan-auditor",
            reason="recorded on the wrong task id",
        )
        wrong_claim = kb.claim_task(conn, executor, claimer="executor")
        executor_after_wrong_id = kb.get_task(conn, executor)

        kb.record_plan_audit_verdict(
            conn,
            executor,
            approved=True,
            reviewer="plan-auditor",
            reason="recorded on the gated executor task id",
        )
        right_claim = kb.claim_task(conn, executor, claimer="executor")
        executor_after_right_id = kb.get_task(conn, executor)

    assert wrong_claim is None
    assert executor_after_wrong_id is not None
    assert executor_after_wrong_id.status == "ready"
    assert right_claim is not None
    assert executor_after_right_id is not None
    assert executor_after_right_id.status == "running"


def test_ready_executor_still_cannot_claim_before_plan_approval(kanban_home):
    with kb.connect() as conn:
        planner = kb.create_task(conn, title="write initial plan", assignee="planner")
        executor = kb.create_task(
            conn,
            title="execute after plan",
            assignee="executor",
            parents=(planner,),
            plan_audit_required=True,
        )
        assert kb.get_task(conn, executor).status == "todo"

        assert kb.complete_task(conn, planner, summary="initial plan written")
        promoted = kb.get_task(conn, executor)
        assert promoted is not None
        assert promoted.status == "ready"

        claimed = kb.claim_task(conn, executor, claimer="executor")
        after_claim_attempt = kb.get_task(conn, executor)
        requested = _events(conn, executor, "plan_audit_requested")

    assert claimed is None
    assert after_claim_attempt is not None
    assert after_claim_attempt.status == "ready"
    assert len(requested) == 1
    assert requested[-1].payload == {"rejected_rounds": 0, "limit": 2}


def test_replayed_rejected_round_is_idempotent_and_does_not_exhaust_early(
    kanban_home,
):
    with kb.connect() as conn:
        executor = kb.create_task(
            conn,
            title="executor with retry-safe audit",
            assignee="executor",
            plan_audit_required=True,
            plan_audit_max_rounds=2,
        )
        metadata = {"round": 1, "kind": "revise_plan"}

        kb.record_plan_audit_verdict(
            conn,
            executor,
            approved=False,
            reviewer="plan-auditor",
            reason="missing files",
            metadata=metadata,
        )
        kb.record_plan_audit_verdict(
            conn,
            executor,
            approved=False,
            reviewer="plan-auditor",
            reason="missing files",
            metadata=metadata,
        )

        claimed = kb.claim_task(conn, executor, claimer="executor")
        task = kb.get_task(conn, executor)
        rejected_events = _events(conn, executor, "plan_audit_rejected")
        requested_events = _events(conn, executor, "plan_audit_requested")

    assert len(rejected_events) == 1
    assert claimed is None
    assert task is not None
    assert task.status == "ready"
    assert task.block_kind is None
    assert requested_events[-1].payload == {"rejected_rounds": 1, "limit": 2}


def test_legacy_duplicate_rejected_events_with_same_round_do_not_exhaust_early(
    kanban_home,
):
    with kb.connect() as conn:
        executor = kb.create_task(
            conn,
            title="executor with historical duplicate audit event",
            assignee="executor",
            plan_audit_required=True,
            plan_audit_max_rounds=2,
        )
        payload = json.dumps({
            "reason": "legacy replay duplicate",
            "metadata": {"round": 1, "kind": "revise_plan"},
        })
        now = int(time.time())
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'plan_audit_rejected', ?, ?)",
                (executor, payload, now),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'plan_audit_rejected', ?, ?)",
                (executor, payload, now),
            )

        claimed = kb.claim_task(conn, executor, claimer="executor")
        task = kb.get_task(conn, executor)
        requested_events = _events(conn, executor, "plan_audit_requested")

    assert claimed is None
    assert task is not None
    assert task.status == "ready"
    assert task.block_kind is None
    assert requested_events[-1].payload == {"rejected_rounds": 1, "limit": 2}


def test_unique_rejected_rounds_exhaust_to_needs_input(kanban_home):
    with kb.connect() as conn:
        executor = kb.create_task(
            conn,
            title="executor with exhausted audit",
            assignee="executor",
            plan_audit_required=True,
            plan_audit_max_rounds=2,
        )

        kb.record_plan_audit_verdict(
            conn,
            executor,
            approved=False,
            metadata={"round": 1, "kind": "revise_plan"},
        )
        kb.record_plan_audit_verdict(
            conn,
            executor,
            approved=False,
            metadata={"round": 2, "kind": "revise_plan"},
        )

        claimed = kb.claim_task(conn, executor, claimer="executor")
        task = kb.get_task(conn, executor)
        exhausted = _events(conn, executor, "plan_audit_exhausted")

    assert claimed is None
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"
    assert exhausted[-1].payload == {"rejected_rounds": 2, "limit": 2}


def test_plan_audit_actuator_reject_revision_is_idempotent_then_approval_opens_gate(
    kanban_home,
):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root coding goal", assignee="lead")
        auditor = kb.create_task(conn, title="audit round 1", assignee="auditor")
        executor = kb.create_task(
            conn,
            title="executor",
            assignee="executor",
            parents=(auditor,),
            plan_audit_required=True,
            plan_audit_max_rounds=2,
        )
        assert kb.claim_task(conn, executor, claimer="executor") is None

        result = kb.apply_plan_audit_actuation(
            conn,
            executor_task_id=executor,
            auditor_task_id=auditor,
            root_task_id=root,
            approved=False,
            reviewer="auditor",
            reason="plan needs concrete files",
            metadata={"round": 1, "kind": "revise_plan"},
            planner_assignee="planner",
            auditor_assignee="auditor",
        )
        replay = kb.apply_plan_audit_actuation(
            conn,
            executor_task_id=executor,
            auditor_task_id=auditor,
            root_task_id=root,
            approved=False,
            reviewer="auditor",
            reason="plan needs concrete files",
            metadata={"round": 1, "kind": "revise_plan"},
            planner_assignee="planner",
            auditor_assignee="auditor",
        )
        assert result.action == "revision_created"
        assert replay.action == "revision_created"
        assert result.planner_task_id == replay.planner_task_id
        assert result.auditor_revision_task_id == replay.auditor_revision_task_id
        assert result.auditor_completed is True

        planner_key = f"koc:{root}:{executor}:plan-round:2:planner"
        auditor_key = f"koc:{root}:{executor}:plan-round:2:auditor"
        planner_r2 = result.planner_task_id
        auditor_r2 = result.auditor_revision_task_id
        assert planner_r2 is not None
        assert auditor_r2 is not None
        assert kb.get_task(conn, executor).status == "todo"

        assert _count_tasks_with_key(conn, planner_key) == 1
        assert _count_tasks_with_key(conn, auditor_key) == 1
        assert len(_events(conn, executor, "plan_audit_rejected")) == 1
        auditor_after_replay = kb.get_task(conn, auditor)
        assert auditor_after_replay is not None
        assert auditor_after_replay.status == "done"

        assert kb.complete_task(conn, planner_r2, summary="round 2 plan ready")
        assert kb.complete_task(conn, auditor_r2, summary="round 2 audit approved")
        ready_executor = kb.get_task(conn, executor)
        assert ready_executor is not None
        assert ready_executor.status == "ready"

        kb.record_plan_audit_verdict(
            conn,
            executor,
            approved=True,
            reviewer="auditor",
            reason="round 2 plan is concrete",
            metadata={"round": 2},
        )
        claimed = kb.claim_task(conn, executor, claimer="executor")
        executor_after_approval = kb.get_task(conn, executor)

    assert claimed is not None
    assert executor_after_approval is not None
    assert executor_after_approval.status == "running"


def test_plan_audit_actuator_human_input_blocks_and_completes_auditor(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root coding goal", assignee="lead")
        executor = kb.create_task(
            conn,
            title="executor needing user decision",
            assignee="executor",
            plan_audit_required=True,
        )
        auditor = kb.create_task(conn, title="audit plan", assignee="auditor")
        assert kb.claim_task(conn, auditor, claimer="auditor") is not None

        result = kb.apply_plan_audit_actuation(
            conn,
            executor_task_id=executor,
            auditor_task_id=auditor,
            root_task_id=root,
            approved=False,
            reviewer="auditor",
            reason="needs product decision",
            metadata={"round": 1, "kind": "needs_user_decision"},
            comment="PLAN AUDIT NEEDS INPUT: choose between API A and API B.",
        )

        executor_after_block = kb.get_task(conn, executor)
        auditor_after_completion = kb.get_task(conn, auditor)
        comments = kb.list_comments(conn, executor)

    assert result.action == "blocked"
    assert executor_after_block is not None
    assert executor_after_block.status == "blocked"
    assert executor_after_block.block_kind == "needs_input"
    assert auditor_after_completion is not None
    assert auditor_after_completion.status == "done"
    assert comments[-1].body.startswith("PLAN AUDIT NEEDS INPUT:")


def test_plan_audit_actuator_replay_does_not_duplicate_comments(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root coding goal", assignee="lead")
        auditor = kb.create_task(conn, title="audit plan", assignee="auditor")
        executor = kb.create_task(
            conn,
            title="executor needing user decision",
            assignee="executor",
            plan_audit_required=True,
        )
        comment = "PLAN AUDIT NEEDS INPUT: choose between API A and API B."

        first = kb.apply_plan_audit_actuation(
            conn,
            executor_task_id=executor,
            auditor_task_id=auditor,
            root_task_id=root,
            approved=False,
            reviewer="auditor",
            reason="needs product decision",
            metadata={"round": 1, "kind": "needs_user_decision"},
            comment=comment,
        )
        replay = kb.apply_plan_audit_actuation(
            conn,
            executor_task_id=executor,
            auditor_task_id=auditor,
            root_task_id=root,
            approved=False,
            reviewer="auditor",
            reason="needs product decision",
            metadata={"round": 1, "kind": "needs_user_decision"},
            comment=comment,
        )
        comments = kb.list_comments(conn, executor)

    assert first.comment_id is not None
    assert replay.comment_id == first.comment_id
    assert [c.body for c in comments].count(comment) == 1


def test_plan_audit_block_preserves_block_recurrence_signal(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="root coding goal", assignee="lead")
        auditor = kb.create_task(conn, title="audit plan", assignee="auditor")
        executor = kb.create_task(
            conn,
            title="executor with prior unblock loop",
            assignee="executor",
            plan_audit_required=True,
        )
        assert kb.block_task(
            conn,
            executor,
            reason="first needs input",
            kind="needs_input",
        )
        assert kb.unblock_task(conn, executor)

        result = kb.apply_plan_audit_actuation(
            conn,
            executor_task_id=executor,
            auditor_task_id=auditor,
            root_task_id=root,
            approved=False,
            reviewer="auditor",
            reason="needs product decision",
            metadata={"round": 1, "kind": "needs_user_decision"},
            comment="PLAN AUDIT NEEDS INPUT: choose between API A and API B.",
        )
        task = kb.get_task(conn, executor)
        loop_events = _events(conn, executor, "block_loop_detected")

    assert result.action == "blocked"
    assert task is not None
    assert task.status == "triage"
    assert task.block_kind == "needs_input"
    assert task.block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT
    assert loop_events[-1].payload["source"] == "plan_audit"
