"""Focused tests for safe project retention planning and application."""
from dataclasses import replace
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.project_finalization_contract import (
    create_project_finalization,
    record_checker_verdict,
    record_delivery_attempt,
    record_final_artifacts,
    record_terminal_outcome,
    register_project_member,
    schedule_project_cleanup,
)
from hermes_cli.project_retention_cleanup import (
    CleanupAction,
    apply_cleanup_plan,
    plan_project_cleanup,
)


NOW = datetime(2026, 7, 16, 12, tzinfo=timezone.utc)


def _finalized(tmp_path, *, cleanup_after=None, include_artifacts=True, delivery_state="delivered", accepted=True):
    conn = kb.connect(tmp_path / "board.db")
    root = kb.create_task(conn, title="root", initial_status="running")
    child = kb.create_task(conn, title="child", initial_status="running")
    unrelated = kb.create_task(conn, title="unrelated", initial_status="running")
    conn.execute("UPDATE tasks SET status='done', completed_at=? WHERE id IN (?, ?)", (int(NOW.timestamp()), root, child))
    finalization = create_project_finalization(
        conn, board_id="board", root_task_id=root, final_checker_task_id=child,
    )
    register_project_member(
        conn, board_id="board", root_task_id=root, generation=1,
        task_id=child, membership_kind="required", required=True,
    )
    record_checker_verdict(
        conn, board_id="board", root_task_id=root, generation=1,
        checker_task_id=child, verdict="PASS",
    )
    record_terminal_outcome(conn, board_id="board", root_task_id=root, generation=1, outcome="COMPLETE")
    if include_artifacts:
        artifact_root = tmp_path / "evidence"
        report = artifact_root / "final-report.md"
        manifest = artifact_root / "manifest.json"
        usage = artifact_root / "usage-summary.json"
        artifact_root.mkdir()
        report.write_text("report", encoding="utf-8")
        manifest.write_text("{}", encoding="utf-8")
        usage.write_text("{}", encoding="utf-8")
        record_final_artifacts(
            conn, board_id="board", root_task_id=root, generation=1,
            report_path=str(report), report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            manifest_path=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )
    record_delivery_attempt(
        conn, board_id="board", root_task_id=root, generation=1,
        idempotency_key="delivery-1", platform="test", attempt_number=1,
        delivery_state=delivery_state, accepted=accepted,
    )
    schedule_project_cleanup(
        conn, board_id="board", root_task_id=root, generation=1,
        cleanup_after=(cleanup_after or (NOW - timedelta(hours=1))).isoformat(),
    )
    return conn, root, child, unrelated


def _rehash(plan):
    payload = json.dumps(plan.canonical_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return replace(plan, plan_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest())


def _mutation_snapshot(conn):
    return (
        conn.execute("SELECT id, status FROM tasks ORDER BY id").fetchall(),
        conn.execute("SELECT COUNT(*) FROM project_cleanup_journal").fetchone()[0],
    )


def test_early_cleanup_and_dry_run_are_refused_and_mutation_free(tmp_path):
    conn, root, child, unrelated = _finalized(tmp_path, cleanup_after=NOW + timedelta(hours=1))
    before = conn.execute("SELECT id, status FROM tasks ORDER BY id").fetchall()
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    assert "retention_not_expired" in plan.refusal_reasons
    result = apply_cleanup_plan(conn, plan, dry_run=True)
    assert result.dry_run is True
    assert conn.execute("SELECT id, status FROM tasks ORDER BY id").fetchall() == before
    assert conn.execute("SELECT COUNT(*) FROM project_cleanup_journal").fetchone()[0] == 0


def test_missing_artifact_and_incomplete_delivery_fail_closed(tmp_path):
    conn, root, child, unrelated = _finalized(tmp_path, include_artifacts=False, delivery_state="pending", accepted=None)
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    assert "missing_artifacts" in plan.refusal_reasons
    assert "nonterminal_delivery" in plan.refusal_reasons
    with pytest.raises(ValueError, match="not eligible"):
        apply_cleanup_plan(conn, plan)


def test_ambiguous_terminal_delivery_is_refused(tmp_path):
    conn, root, child, unrelated = _finalized(tmp_path)
    record_delivery_attempt(
        conn, board_id="board", root_task_id=root, generation=1,
        idempotency_key="delivery-2", platform="test", attempt_number=2,
        delivery_state="delivered", accepted=True,
    )
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    assert plan.refusal_reasons == ("ambiguous_delivery",)


def test_apply_archives_only_explicit_members_and_is_idempotent(tmp_path):
    conn, root, child, unrelated = _finalized(tmp_path)
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    assert plan.eligible is True
    assert [action.task_id for action in plan.actions] == sorted([root, child])
    first = apply_cleanup_plan(conn, plan)
    assert set(first.applied_task_ids) == {root, child}
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (unrelated,)).fetchone()[0] == "ready"
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (root,)).fetchone()[0] == "archived"
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (child,)).fetchone()[0] == "archived"
    assert conn.execute("SELECT COUNT(*) FROM project_cleanup_journal").fetchone()[0] == 1
    second = apply_cleanup_plan(conn, plan)
    assert second.applied_task_ids == ()
    assert conn.execute("SELECT COUNT(*) FROM project_cleanup_journal").fetchone()[0] == 1


def test_forged_eligible_plan_for_unrelated_task_is_refused_without_mutation(tmp_path):
    conn, root, child, unrelated = _finalized(tmp_path)
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    forged = _rehash(replace(
        plan,
        eligible_task_ids=(unrelated,),
        actions=(CleanupAction(unrelated),),
    ))
    before = _mutation_snapshot(conn)
    with pytest.raises(ValueError, match="stale or tampered"):
        apply_cleanup_plan(conn, forged)
    assert _mutation_snapshot(conn) == before


@pytest.mark.parametrize("status", ("running", "ready", "blocked"))
def test_stale_member_that_becomes_active_is_refused_without_mutation(tmp_path, status):
    conn, root, child, unrelated = _finalized(tmp_path)
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, child))
    before = _mutation_snapshot(conn)
    with pytest.raises(ValueError, match="currently authorized"):
        apply_cleanup_plan(conn, plan)
    assert _mutation_snapshot(conn) == before


@pytest.mark.parametrize("field", ("eligible_task_ids", "actions"))
def test_tampered_plan_actions_or_eligible_ids_are_refused_without_mutation(tmp_path, field):
    conn, root, child, unrelated = _finalized(tmp_path)
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    tampered = replace(plan, **{
        field: (root,) if field == "eligible_task_ids" else (CleanupAction(root),),
    })
    before = _mutation_snapshot(conn)
    with pytest.raises(ValueError, match="stale or tampered"):
        apply_cleanup_plan(conn, _rehash(tampered))
    assert _mutation_snapshot(conn) == before


def test_generation_or_hash_mismatch_is_refused_without_mutation(tmp_path):
    conn, root, child, unrelated = _finalized(tmp_path)
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    before = _mutation_snapshot(conn)
    with pytest.raises(ValueError, match="current finalization generation"):
        apply_cleanup_plan(conn, _rehash(replace(plan, generation=2)))
    assert _mutation_snapshot(conn) == before
    with pytest.raises(ValueError, match="hash is invalid"):
        apply_cleanup_plan(conn, replace(plan, plan_sha256="0" * 64))
    assert _mutation_snapshot(conn) == before


def test_unscheduled_cleanup_derives_durable_retention_days_cutoff(tmp_path):
    conn, root, child, unrelated = _finalized(tmp_path)
    conn.execute("UPDATE project_finalizations SET cleanup_after=NULL WHERE root_task_id=?", (root,))
    finalization = conn.execute(
        "SELECT finalized_at, retention_days FROM project_finalizations WHERE root_task_id=?",
        (root,),
    ).fetchone()
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    expected = datetime.fromtimestamp(
        finalization["finalized_at"],
        tz=timezone.utc,
    ) + timedelta(days=finalization["retention_days"])
    assert plan.retention_cutoff == expected.isoformat()
    assert "retention_not_expired" in plan.refusal_reasons
    # Injected duration produces a deterministic cutoff and allows the test to
    # exercise the configured policy without a hard-coded production default.
    plan = plan_project_cleanup(
        conn, board_id="board", root_task_id=root, now=NOW,
        retention_duration=timedelta(hours=48),
    )
    assert "retention_not_expired" in plan.refusal_reasons


def test_journal_boundary_receives_exact_plan_and_artifacts_remain_accessible(tmp_path):
    conn, root, child, unrelated = _finalized(tmp_path)
    plan = plan_project_cleanup(conn, board_id="board", root_task_id=root, now=NOW)
    calls = []
    def journal_writer(connection, **kwargs):
        calls.append(kwargs)
        return kwargs
    result = apply_cleanup_plan(conn, plan, journal_writer=journal_writer)
    assert result.journal["plan_sha256"] == plan.plan_sha256
    assert calls[0]["status"] == "applied"
    assert calls[0]["archived_task_count"] == 2
    for path in plan.evidence_paths:
        assert path
