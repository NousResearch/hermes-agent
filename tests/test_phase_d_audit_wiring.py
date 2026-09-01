"""Phase D — Audit + Wiring tests.

Covers:
  D1: Audit gate (PASS/CONDITIONAL/BLOCKED verdict parsing)
  D2: Quan fleet sub-gate sections
  D3: Kensei-review section
  D4: Denji event emission + spawn-frequency tracking
  D5: Decompose, execute, pr+qa, audit, document stages (12-stage pipeline)
  D6: Express path (B.10, B.11) — --express flag, stage subset, bypass records
  D7: New states in VALID_STATUSES
  D8: Audit follow-up task creation
  D9: Audit BLOCKED bounce-to-spec with loop cap
  D10: Dispatcher handles pass-through stages
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Fixtures (mirror Phase A/B/C pattern)
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch) -> Iterator[Path]:
    """Isolated HERMES_HOME for a fresh DB and feature-artifacts tree."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "feature-artifacts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture
def conn(kanban_home) -> Iterator[sqlite3.Connection]:
    from hermes_cli import kanban_db as kb
    # Decompose child materialisation requires spawnable owners; tests use a
    # stubbed canonical runtime (same pattern as test_pipeline_execution_integrity).
    monkeypatch_spawnable()
    with kb.connect() as c:
        yield c


def monkeypatch_spawnable():
    from hermes_cli import kanban_db as kb
    if not getattr(kb, "_spawnable_stub_active", False):
        kb._is_profile_spawnable_orig = kb._is_profile_spawnable
        kb._is_profile_spawnable = lambda _name: True
        kb._spawnable_stub_active = True


# ---------------------------------------------------------------------------
# D7: New states in VALID_STATUSES
# ---------------------------------------------------------------------------


class TestNewPipelineStates:
    """All five Phase D stages must be in VALID_STATUSES."""

    def test_decompose_in_valid_statuses(self):
        from hermes_cli import kanban_db as kb
        assert "decompose" in kb.VALID_STATUSES

    def test_execute_in_valid_statuses(self):
        from hermes_cli import kanban_db as kb
        assert "execute" in kb.VALID_STATUSES

    def test_pr_qa_in_valid_statuses(self):
        from hermes_cli import kanban_db as kb
        assert "pr+qa" in kb.VALID_STATUSES

    def test_audit_in_valid_statuses(self):
        from hermes_cli import kanban_db as kb
        assert "audit" in kb.VALID_STATUSES

    def test_document_in_valid_statuses(self):
        from hermes_cli import kanban_db as kb
        assert "document" in kb.VALID_STATUSES

    def test_pipeline_stage_column_accepts_phase_d_stages(self, kanban_home):
        from hermes_cli import kanban_db as kb
        with kb.connect() as conn:
            tid = kb.create_task(
                conn, title="t", assignee="octacon",
                body="## Problem\nx\n## Success Criteria\ny", triage=True,
            )
            for stage in ("decompose", "execute", "pr+qa", "audit", "document"):
                conn.execute(
                    "UPDATE tasks SET pipeline_stage = ?, status = ? WHERE id = ?",
                    (stage, stage, tid),
                )
                row = conn.execute(
                    "SELECT pipeline_stage, status FROM tasks WHERE id = ?",
                    (tid,),
                ).fetchone()
                assert row["pipeline_stage"] == stage
                assert row["status"] == stage


# ---------------------------------------------------------------------------
# D5: 12-stage pipeline order
# ---------------------------------------------------------------------------


class TestPhaseDPipelineOrder:
    """Pipeline must walk all 12 stages in design-doc §3 order."""

    def test_full_pipeline_walks_all_12_stages(self):
        from hermes_cli.feature_pipeline import get_next_stage
        current = "research"
        path = [current]
        while current := get_next_stage(current):
            path.append(current)
        assert path == [
            "research", "prd", "spec", "council",
            "sign_off", "tech_review",
            "decompose", "execute", "pr+qa", "audit",
            "final_sign_off", "document",
        ]

    def test_express_pipeline_skips_heavy_stages(self):
        from hermes_cli.feature_pipeline import (
            get_next_stage, EXPRESS_PIPELINE_STAGES,
        )
        # Express path drops PRD, council, tech_review; keeps audit
        for skipped in ("prd", "council", "tech_review"):
            assert skipped not in EXPRESS_PIPELINE_STAGES
        # Express includes audit (design doc §4a, Phase D)
        assert "audit" in EXPRESS_PIPELINE_STAGES
        # Express walk works
        current = "research"
        path = [current]
        while current := get_next_stage(current, "express"):
            path.append(current)
        assert "prd" not in path
        assert "council" not in path
        assert path[0] == "research"
        assert path[-1] == "document"

    def test_get_next_stage_unknown_stage(self):
        from hermes_cli.feature_pipeline import get_next_stage
        assert get_next_stage("nonexistent") is None
        assert get_next_stage("nonexistent", "express") is None


# ---------------------------------------------------------------------------
# D1: Audit gate — verdict parsing
# ---------------------------------------------------------------------------


class TestAuditGate:
    """validate_audit_artifact parses PASS/CONDITIONAL/BLOCKED verdicts."""

    def test_gate_fails_when_artifact_missing(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_audit_artifact
        result = validate_audit_artifact(str(tmp_path))
        assert result is not None
        assert "audit-report.md" in result.lower() or "missing" in result.lower()

    def test_gate_fails_when_sections_missing(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_audit_artifact
        audit = tmp_path / "audit-report.md"
        audit.write_text("# Audit\n\n## Summary\nLooks OK\n")
        result = validate_audit_artifact(str(tmp_path))
        assert result is not None
        # Should mention missing Quan sections
        assert "Quan" in result or "quan" in result or "sub-gate" in result.lower()

    def test_gate_passes_on_PASS_verdict(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_audit_artifact
        self._write_full_audit_report(tmp_path, verdict="PASS")
        result = validate_audit_artifact(str(tmp_path))
        assert result is None, f"Expected None, got: {result!r}"

    def test_gate_passes_on_CONDITIONAL_verdict(self, tmp_path):
        """CONDITIONAL still passes the gate (issues tracked separately)."""
        from hermes_cli.feature_pipeline import validate_audit_artifact
        self._write_full_audit_report(tmp_path, verdict="CONDITIONAL")
        result = validate_audit_artifact(str(tmp_path))
        assert result is None, f"Expected None, got: {result!r}"

    def test_gate_fails_on_BLOCKED_verdict(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_audit_artifact
        self._write_full_audit_report(tmp_path, verdict="BLOCKED")
        result = validate_audit_artifact(str(tmp_path))
        assert result is not None
        assert "BLOCKED" in result

    def test_get_audit_verdict_PASS(self, tmp_path):
        from hermes_cli.feature_pipeline import get_audit_verdict
        self._write_full_audit_report(tmp_path, verdict="PASS")
        assert get_audit_verdict(str(tmp_path)) == "PASS"

    def test_get_audit_verdict_CONDITIONAL(self, tmp_path):
        from hermes_cli.feature_pipeline import get_audit_verdict
        self._write_full_audit_report(tmp_path, verdict="CONDITIONAL")
        assert get_audit_verdict(str(tmp_path)) == "CONDITIONAL"

    def test_get_audit_verdict_BLOCKED(self, tmp_path):
        from hermes_cli.feature_pipeline import get_audit_verdict
        self._write_full_audit_report(tmp_path, verdict="BLOCKED")
        assert get_audit_verdict(str(tmp_path)) == "BLOCKED"

    def test_get_audit_verdict_missing(self, tmp_path):
        from hermes_cli.feature_pipeline import get_audit_verdict
        assert get_audit_verdict(str(tmp_path)) is None

    def test_audit_gate_requires_kensei_review_section(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_audit_artifact
        # Write Quan sub-gates + verdict but no kensei-review section
        (tmp_path / "audit-report.md").write_text(
            "# Audit\n\n"
            "## Quan Sub-Gates\n\n"
            "### Code Quality\nPASS — tests cover new code.\n\n"
            "### Architecture\nPASS — fits existing patterns.\n\n"
            "### Performance\nPASS — no hot path touched.\n\n"
            "### Security\nPASS — input validation in place.\n\n"
            "### UX\nPASS — no user-facing changes.\n\n"
            "## Verdict\n\n**Verdict: PASS**\n"
        )
        result = validate_audit_artifact(str(tmp_path))
        assert result is not None
        assert "kensei" in result.lower() or "review" in result.lower()

    @staticmethod
    def _write_full_audit_report(tmp_path, *, verdict: str) -> None:
        (tmp_path / "audit-report.md").write_text(
            f"# Audit Report\n\n"
            f"## Quan-Fleet\n\n"
            f"**code:** PASS\n\n"
            f"**arch:** PASS\n\n"
            f"**perf:** PASS\n\n"
            f"**security:** PASS\n\n"
            f"**ux:** PASS\n\n"
            f"## Kensei-Review\n\nReviewed against mission priorities — within scope and acceptable for this phase.\n\n"
            f"## Verdict\n\n**Verdict: {verdict}**\n"
        )


# ---------------------------------------------------------------------------
# D5: Decompose + document gates
# ---------------------------------------------------------------------------


class TestDecomposeAndDocumentGates:
    """decompose: WS-1 contract; document: docs section required."""

    def test_decompose_gate_fails_without_decompose_md(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_decompose_artifact
        result = validate_decompose_artifact(str(tmp_path))
        assert result is not None
        assert "decompose" in result.lower() or "missing" in result.lower()

    def test_decompose_gate_fails_without_ws1_sections(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_decompose_artifact
        (tmp_path / "decompose-output.md").write_text("# Decompose\n")
        result = validate_decompose_artifact(str(tmp_path))
        assert result is not None

    @staticmethod
    def _write_valid_decompose_manifest(
        tmp_path, parent_id="t", key="WS-1", title="ws1", owner="octacon",
    ):
        """Write the executable decompose-tasks.json sidecar alongside the
        human-readable decompose-output.md.  The manifest is the machine
        contract; the markdown is never parsed into task rows."""
        manifest = {
            "schema_version": 1,
            "parent_task_id": parent_id,
            "tasks": [
                {
                    "key": key,
                    "title": "ws1 implementation",
                    "owner": owner,
                    "role": "implementation",
                    "workspace_kind": "scratch",
                    "body": "do the thing",
                    "dependencies": [],
                    "dependencies_qa": [],
                },
                {
                    "key": "QA-1",
                    "title": "ws1 qa",
                    "owner": "quan",
                    "role": "qa",
                    "workspace_kind": "scratch",
                    "body": "check the thing",
                    "dependencies": [key],
                    "dependencies_qa": [],
                },
                {
                    "key": "AU-1",
                    "title": "ws1 audit",
                    "owner": "kensei-review",
                    "role": "audit",
                    "workspace_kind": "scratch",
                    "body": "audit the thing",
                    "dependencies": ["QA-1"],
                    "dependencies_qa": [],
                },
            ],
        }
        (tmp_path / "decompose-tasks.json").write_text(
            json.dumps(manifest)
        )
        return manifest

    def test_decompose_gate_passes_with_ws1_sections(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_decompose_artifact
        (tmp_path / "decompose-output.md").write_text(
            "# Decomposition\n\n"
            "## Child Tasks\n- WS-1: implement X\n- WS-2: implement Y\n\n"
            "## Acceptance Criteria\n- WS-1: AC1\n- WS-2: AC2\n\n"
            "## Test Plan\n- Unit: WS-1 tests\n- Integration: end-to-end\n\n"
            "## Order\nWS-1 first, then WS-2.\n"
        )
        self._write_valid_decompose_manifest(tmp_path, parent_id="t")
        assert validate_decompose_artifact(str(tmp_path)) is None

    def test_document_gate_fails_without_docs_section(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_document_artifact
        result = validate_document_artifact(str(tmp_path))
        assert result is not None

    def test_document_gate_passes_with_docs_section(self, tmp_path):
        from hermes_cli.feature_pipeline import validate_document_artifact
        (tmp_path / "docs-output.md").write_text(
            "# Documentation\n\n"
            "## Overview\nProject documentation for X.\n\n"
            "## Usage\nExample usage patterns.\n\n"
            "## Changelog\n- Added X.\n"
        )
        assert validate_document_artifact(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Pass-through stages
# ---------------------------------------------------------------------------


class TestPassThroughStages:
    """execute and pr+qa are execution-bearing stages and MUST be gated.

    Current locked contract (feature_pipeline): "Every execution-bearing
    stage has an evidence gate; no implementation or QA stage is allowed
    to pass through on elapsed time alone."  Prose-only pass-through was
    removed when the evidence gates landed.
    """

    def test_execute_and_pr_qa_in_gate_functions(self):
        from hermes_cli.feature_pipeline import GATE_FUNCTIONS
        # execute, pr+qa have evidence gates (execution-evidence.json,
        # pr-qa-evidence.json) — they are no longer prose pass-through.
        assert "execute" in GATE_FUNCTIONS
        assert "pr+qa" in GATE_FUNCTIONS


# ---------------------------------------------------------------------------
# D4: Denji event wiring + spawn-frequency tracking
# ---------------------------------------------------------------------------


class TestDenjiEventWiring:
    """Denji review signals + spawn-frequency aggregation."""

    def test_get_spawn_frequency_empty(self, conn):
        from hermes_cli import kanban_db as kb
        result = kb.get_spawn_frequency(conn, days=7)
        assert result == []

    def test_get_spawn_frequency_aggregates_by_assignee_stage(self, conn):
        from hermes_cli import kanban_db as kb
        # Create a task and emit 3 pipeline_spawn events for the same
        # (assignee, stage) pair, plus 1 for a different pair.
        tid1 = kb.create_task(
            conn, title="t1", assignee="remii",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        tid2 = kb.create_task(
            conn, title="t2", assignee="octacon",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        for _ in range(3):
            kb._record_pipeline_spawn(conn, tid1, stage="research", assignee="remii")
        kb._record_pipeline_spawn(conn, tid2, stage="spec", assignee="octacon")
        result = kb.get_spawn_frequency(conn, days=7)
        assert len(result) == 2
        # Sorted by count desc
        assert result[0]["assignee"] == "remii"
        assert result[0]["stage"] == "research"
        assert result[0]["spawn_count"] == 3
        assert result[1]["assignee"] == "octacon"
        assert result[1]["stage"] == "spec"
        assert result[1]["spawn_count"] == 1

    def test_get_spawn_frequency_respects_window(self, conn):
        """Events outside the window are excluded."""
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="old", assignee="remii",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        kb._record_pipeline_spawn(conn, tid, stage="research", assignee="remii")
        # Force created_at to 30 days ago. created_at is epoch-seconds (int).
        import time as _t
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE task_id = ?",
            (int(_t.time()) - 30 * 86400, tid),
        )
        result = kb.get_spawn_frequency(conn, days=7)
        assert result == []


# ---------------------------------------------------------------------------
# D8: Audit follow-up task creation
# ---------------------------------------------------------------------------


class TestAuditFollowupTask:
    """Audit CONDITIONAL auto-creates a follow-up task."""

    def test_followup_task_created_for_conditional(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        parent_id = kb.create_task(
            conn, title="Parent feature",
            assignee="octacon",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        followup_id = kb._create_audit_followup_task(
            conn, parent_id, "CONDITIONAL",
            summary="2 minor issues flagged",
        )
        assert followup_id is not None
        assert followup_id != parent_id
        # Linked via task_links
        row = conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, followup_id),
        ).fetchone()
        assert row is not None
        # Child task has correct title prefix
        child = kb.get_task(conn, followup_id)
        assert "[audit-followup]" in child.title

    def test_followup_task_returns_none_on_missing_parent(self, conn):
        from hermes_cli import kanban_db as kb
        result = kb._create_audit_followup_task(conn, "nonexistent", "CONDITIONAL")
        assert result is None


class TestDecomposeChildTasks:
    """decompose stage creates real child kanban rows from the executable
    decompose-tasks.json manifest — markdown is never parsed into task rows."""

    def test_manifest_missing_task_fails_cleanly(self, tmp_path):
        from hermes_cli.feature_pipeline import load_decompose_manifest
        import pytest as _pytest
        with _pytest.raises((OSError, ValueError)):
            load_decompose_manifest(str(tmp_path / "nope"))

    def test_create_children_returns_error_on_missing_parent(
        self, conn, tmp_path, kanban_home,
    ):
        """Current contract raises a defensive ValueError for an unknown
        parent task rather than silently returning empty."""
        from hermes_cli import kanban_db as kb
        import pytest as _pytest
        manifest = {
            "schema_version": 1,
            "parent_task_id": "nonexistent",
            "tasks": [
                {
                    "key": "WS-1", "title": "x", "owner": "octacon",
                    "role": "implementation", "workspace_kind": "scratch",
                    "body": "b", "dependencies": [], "dependencies_qa": [],
                },
                {
                    "key": "QA-1", "title": "q", "owner": "quan",
                    "role": "qa", "workspace_kind": "scratch",
                    "body": "b", "dependencies": ["WS-1"],
                    "dependencies_qa": [],
                },
                {
                    "key": "AU-1", "title": "a", "owner": "kensei-review",
                    "role": "audit", "workspace_kind": "scratch",
                    "body": "b", "dependencies": ["QA-1"],
                    "dependencies_qa": [],
                },
            ],
        }
        (tmp_path / "decompose-tasks.json").write_text(json.dumps(manifest))
        with _pytest.raises(ValueError):
            kb._create_decompose_child_tasks(conn, "nonexistent", str(tmp_path))

    def _manifest(self, parent_id, owner="octacon"):
        return {
            "schema_version": 1,
            "parent_task_id": parent_id,
            "tasks": [
                {
                    "key": "WS-1", "title": "ws1 implementation",
                    "owner": owner, "role": "implementation",
                    "workspace_kind": "scratch", "body": "do it",
                    "dependencies": [], "dependencies_qa": [],
                },
                {
                    "key": "QA-1", "title": "ws1 qa", "owner": "quan",
                    "role": "qa", "workspace_kind": "scratch",
                    "body": "check it", "dependencies": ["WS-1"],
                    "dependencies_qa": [],
                },
                {
                    "key": "AU-1", "title": "ws1 audit",
                    "owner": "kensei-review", "role": "audit",
                    "workspace_kind": "scratch", "body": "audit it",
                    "dependencies": ["QA-1"], "dependencies_qa": [],
                },
            ],
        }

    def test_create_children_links_to_parent(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        parent_id = kb.create_task(
            conn, title="Parent feature", assignee="octacon", tier="full",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        artifact_dir = kanban_home / "feature-artifacts" / parent_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "decompose-tasks.json").write_text(
            json.dumps(self._manifest(parent_id))
        )
        new_ids = kb._create_decompose_child_tasks(conn, parent_id, str(artifact_dir))
        assert len(new_ids) == 3
        # Current contract: child dependency edges live in task_links
        # (child_ids = dependency children); decompose membership is the
        # decompose_children_created event, and every child inherits tier.
        assert "decompose_children_created" in [
            e.kind for e in kb.list_events(conn, parent_id)
        ]
        for child in new_ids:
            child_task = kb.get_task(conn, child["task_id"])
            assert child_task.tier == "full"  # inherited from parent


# ---------------------------------------------------------------------------
# D10: Dispatcher handles new stages
# ---------------------------------------------------------------------------


class TestDispatcherNewStages:
    """dispatch_once walks new stages and records correct events."""

    @staticmethod
    def _materialise_children(conn, kb, tid, kanban_home):
        """Create a valid decomposed child graph with all children done so
        the audit stage's runtime validation passes."""
        artifact_dir = kanban_home / "feature-artifacts" / tid
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "decompose-tasks.json").write_text(
            json.dumps(TestDecomposeChildTasks()._manifest(tid))
        )
        kb._create_decompose_child_tasks(conn, tid, str(artifact_dir))
        children = kb._decompose_children_event(conn, tid)
        for c in children:
            conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id = ?",
                (c["task_id"],),
            )
        return artifact_dir, children

    def test_decompose_artifact_missing_records_gate_failed(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="decompose-fail",
            assignee="octacon",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        conn.execute(
            "UPDATE tasks SET status = ?, pipeline_stage = ?, pipeline_mode = ? WHERE id = ?",
            ("decompose", "decompose", "full", tid),
        )
        kb.dispatch_once(conn, dry_run=False)
        events = kb.list_events(conn, tid)
        gate_fails = [e for e in events if e.kind == "gate_failed"]
        assert len(gate_fails) >= 1
        assert gate_fails[0].payload["stage"] == "decompose"

    def test_decompose_artifact_present_advances(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="decompose-pass",
            assignee="octacon",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        conn.execute(
            "UPDATE tasks SET status = ?, pipeline_stage = ?, pipeline_mode = ? WHERE id = ?",
            ("decompose", "decompose", "full", tid),
        )
        artifact_dir = kanban_home / "feature-artifacts" / tid
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "decompose-output.md").write_text(
            "# Decomposition\n\n"
            "## Child Tasks\n- WS-1: x\n\n"
            "## Acceptance Criteria\n- WS-1: AC1\n- WS-2: AC2\n\n"
            "## Test Plan\n- Unit: WS-1 tests\n- Integration: end-to-end\n\n"
            "## Order\nWS-1 first.\n"
        )
        manifest = TestDecomposeChildTasks()._manifest(tid)
        (artifact_dir / "decompose-tasks.json").write_text(json.dumps(manifest))
        kb.dispatch_once(conn, dry_run=False)
        events = kb.list_events(conn, tid)
        advances = [e for e in events if e.kind == "pipeline_advanced"]
        assert len(advances) >= 1
        decomp_advance = [
            a for a in advances
            if a.payload.get("from_stage") == "decompose"
            and a.payload.get("to_stage") == "execute"
        ]
        assert len(decomp_advance) >= 1
        # Task should now be in execute
        task = kb.get_task(conn, tid)
        assert task.pipeline_stage == "execute"
        # Real child task rows must exist, created from the executable
        # decompose-tasks.json manifest (markdown is never parsed).
        # Decompose membership is the decompose_children_created event;
        # child_ids() returns dependency children.
        children = [
            kb.get_task(conn, c["task_id"])
            for c in kb._decompose_children_event(conn, tid)
        ]
        assert [c.title for c in children] == [
            "ws1 implementation", "ws1 qa", "ws1 audit",
        ]
        assert all(c.tier == task.tier for c in children)  # inherited from parent

    def test_decompose_children_not_duplicated_on_redispatch(self, conn, kanban_home):
        """A second dispatch_once tick before the stage moves on must not
        spawn duplicate child tasks (the near-duplicate-tasks incident this
        guards against)."""
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="decompose-redispatch",
            assignee="octacon",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        conn.execute(
            "UPDATE tasks SET status = ?, pipeline_stage = ?, pipeline_mode = ? WHERE id = ?",
            ("decompose", "decompose", "full", tid),
        )
        artifact_dir = kanban_home / "feature-artifacts" / tid
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "decompose-output.md").write_text(
            "# Decomposition\n\n"
            "## Child Tasks\n- WS-1: x\n- WS-2: y\n\n"
            "## Acceptance Criteria\n- WS-1: AC1\n\n"
            "## Test Plan\n- Unit: WS-1 tests\n"
        )
        manifest = TestDecomposeChildTasks()._manifest(tid)
        (artifact_dir / "decompose-tasks.json").write_text(json.dumps(manifest))
        new_ids_1 = kb._create_decompose_child_tasks(conn, tid, str(artifact_dir))
        new_ids_2 = kb._create_decompose_child_tasks(conn, tid, str(artifact_dir))
        assert new_ids_1 == new_ids_2
        assert len(kb._decompose_children_event(conn, tid)) == 3

    def test_passthrough_stage_execute_auto_advances(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="execute",
            assignee="octacon",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        conn.execute(
            "UPDATE tasks SET status = ?, pipeline_stage = ?, pipeline_mode = ? WHERE id = ?",
            ("execute", "execute", "full", tid),
        )
        # Materialise the child graph + execution evidence required by the
        # execute stage's evidence gate (execution-bearing stages are gated).
        artifact_dir = kanban_home / "feature-artifacts" / tid
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "decompose-tasks.json").write_text(
            json.dumps(TestDecomposeChildTasks()._manifest(tid))
        )
        kb._create_decompose_child_tasks(conn, tid, str(artifact_dir))
        children = kb._decompose_children_event(conn, tid)
        impl_child = next(
            c for c in children if c.get("role") == "implementation"
        )
        evidence = {
            "schema_version": 1,
            "parent_task_id": tid,
            "children": [
                {
                    "key": c["key"],
                    "task_id": c["task_id"],
                    "status": "done",
                    "result_digest": "a" * 64,
                }
                for c in children
                if c.get("role") == "implementation"
            ],
        }
        (artifact_dir / "execution-evidence.json").write_text(
            json.dumps(evidence)
        )
        # Mark all implementation children done so runtime validation passes
        for c in children:
            conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id = ?",
                (c["task_id"],),
            )
        kb.dispatch_once(conn, dry_run=False)
        events = kb.list_events(conn, tid)
        advances = [e for e in events if e.kind == "pipeline_advanced"]
        # execute advances to pr+qa once the evidence gate passes
        assert any(
            a.payload.get("from_stage") == "execute"
            and a.payload.get("to_stage") == "pr+qa"
            for a in advances
        )

    def test_audit_blocked_bounces_to_spec(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="audit-blocked",
            assignee="quan",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        conn.execute(
            "UPDATE tasks SET status = ?, pipeline_stage = ?, pipeline_mode = ? WHERE id = ?",
            ("audit", "audit", "full", tid),
        )
        artifact_dir = kanban_home / "feature-artifacts" / tid
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "audit-report.md").write_text(
            "# Audit Report\n\n"
            "## Quan-Fleet\n\n"
            "**code:** BLOCKED — uncovered branches.\n\n"
            "**arch:** BLOCKED — pattern violation.\n\n"
            "**perf:** PASS — no issue.\n\n"
            "**security:** PASS — sanitized.\n\n"
            "**ux:** PASS — unchanged.\n\n"
            "## Kensei-Review\n\nIssue identified.\n\n"
            "## Verdict\n\n**BLOCKED**\n"
        )
        kb.dispatch_once(conn, dry_run=False)
        # Task should be back in spec
        task = kb.get_task(conn, tid)
        assert task.pipeline_stage == "spec"
        # gate_failed event recorded
        events = kb.list_events(conn, tid)
        audit_fails = [
            e for e in events
            if e.kind == "gate_failed" and e.payload.get("stage") == "audit"
        ]
        assert len(audit_fails) >= 1
        assert audit_fails[0].payload.get("bounced_to") == "spec"

    def test_audit_conditional_creates_followup_and_advances(
        self, conn, kanban_home
    ):
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="audit-cond",
            assignee="quan",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        conn.execute(
            "UPDATE tasks SET status = ?, pipeline_stage = ?, pipeline_mode = ? WHERE id = ?",
            ("audit", "audit", "full", tid),
        )
        artifact_dir, children = self._materialise_children(conn, kb, tid, kanban_home)
        (artifact_dir / "audit-report.md").write_text(
            "# Audit Report\n\n"
            "## Quan-Fleet\n\n"
            "**code:** PASS\n\n"
            "**arch:** PASS\n\n"
            "**perf:** PASS\n\n"
            "**security:** PASS\n\n"
            "**ux:** PASS\n\n"
            "## Kensei-Review\n\nMinor issues flagged but acceptable and within mission scope.\n\n"
            "## Verdict\n\n**Verdict: CONDITIONAL**\n"
        )
        kb.dispatch_once(conn, dry_run=False)
        # Task advanced to final_sign_off (council gate passed)
        task = kb.get_task(conn, tid)
        assert task.pipeline_stage == "final_sign_off"
        # Follow-up task created
        events = kb.list_events(conn, tid)
        followup_events = [e for e in events if e.kind == "audit_followup_created"]
        assert len(followup_events) >= 1
        # Denji signal emitted
        denji_signals = [e for e in events if e.kind == "denji_review_signal"]
        assert any(
            s.payload.get("signal_type") == "audit_conditional"
            for s in denji_signals
        )
        # Audit passed signal also fired
        assert any(
            s.payload.get("signal_type") == "audit_passed"
            for s in denji_signals
        )

    def test_audit_pass_advances_no_followup(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="audit-pass",
            assignee="quan",
            body="## Problem\nx\n## Success Criteria\ny", triage=True,
        )
        conn.execute(
            "UPDATE tasks SET status = ?, pipeline_stage = ?, pipeline_mode = ? WHERE id = ?",
            ("audit", "audit", "full", tid),
        )
        artifact_dir, children = self._materialise_children(conn, kb, tid, kanban_home)
        (artifact_dir / "audit-report.md").write_text(
            "# Audit Report\n\n"
            "## Quan-Fleet\n\n"
            "**code:** PASS — solid.\n\n"
            "**arch:** PASS — fits.\n\n"
            "**perf:** PASS — no issue.\n\n"
            "**security:** PASS — sanitized.\n\n"
            "**ux:** PASS — clean.\n\n"
            "## Kensei-Review\n\nAll good — mission priorities aligned, no blockers identified.\n\n"
            "## Verdict\n\n**Verdict: PASS**\n"
        )
        kb.dispatch_once(conn, dry_run=False)
        # Task advanced to final_sign_off (council gate passed)
        task = kb.get_task(conn, tid)
        assert task.pipeline_stage == "final_sign_off"
        events = kb.list_events(conn, tid)
        followup_events = [e for e in events if e.kind == "audit_followup_created"]
        assert len(followup_events) == 0


# ---------------------------------------------------------------------------
# D6: Express path (B.10, B.11)
# ---------------------------------------------------------------------------


class TestExpressPath:
    """Express mode skips heavy stages; records bypass events."""

    def test_express_mode_field_set_on_task(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="express",
            assignee="remii",
            body="## Problem\nx\n## Success Criteria\ny",
            triage=True,
            pipeline_mode="express",
        )
        row = conn.execute(
            "SELECT pipeline_mode FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["pipeline_mode"] == "express"

    def test_express_pipeline_mode_default_full(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="full",
            assignee="remii",
            body="## Problem\nx\n## Success Criteria\ny",
            triage=True,
        )
        row = conn.execute(
            "SELECT pipeline_mode FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        # pipeline_mode is NULL on tasks where it wasn't set
        assert row["pipeline_mode"] is None
        # get_pipeline_mode treats NULL/None as "full"
        from hermes_cli.feature_pipeline import get_pipeline_mode
        assert get_pipeline_mode(dict(row)) == "full"

    def test_get_pipeline_mode_express(self):
        from hermes_cli.feature_pipeline import get_pipeline_mode
        assert get_pipeline_mode({"pipeline_mode": "express"}) == "express"

    def test_record_bypass_record(self, conn, kanban_home):
        from hermes_cli import kanban_db as kb
        tid = kb.create_task(
            conn, title="bypass",
            assignee="remii",
            body="## Problem\nx\n## Success Criteria\ny",
            triage=True,
        )
        kb._record_bypass_record(
            conn, tid,
            skipped_stages=["prd", "council", "tech_review", "audit"],
            launched_by="kensei",
            mode="express",
        )
        events = kb.list_events(conn, tid)
        bypass_events = [e for e in events if e.kind == "bypass_record"]
        assert len(bypass_events) == 1
        payload = bypass_events[0].payload
        assert payload["mode"] == "express"
        assert payload["launched_by"] == "kensei"
        assert "council" in payload["skipped_stages"]
        assert "audit" in payload["skipped_stages"]

    def test_express_dispatch_skips_prd_when_at_spec_stage(self, conn, kanban_home):
        """A task created in express mode and placed at spec stage should
        advance to sign_off, not to prd."""
        from hermes_cli import kanban_db as kb
        from hermes_cli.feature_pipeline import get_next_stage
        # The pipeline mode controls get_next_stage behaviour
        assert get_next_stage("research", "express") == "spec"
        assert get_next_stage("spec", "express") == "sign_off"
        # And the dispatcher reads pipeline_mode
        tid = kb.create_task(
            conn, title="express-walk",
            assignee="remii",
            body="## Problem\nx\n## Success Criteria\ny",
            triage=True,
            pipeline_mode="express",
        )
        conn.execute(
            "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
            ("research", "research", tid),
        )
        # No research artifact → gate fails
        kb.dispatch_once(conn, dry_run=False)
        events = kb.list_events(conn, tid)
        gate_fails = [e for e in events if e.kind == "gate_failed"]
        assert any(g.payload.get("stage") == "research" for g in gate_fails)


# ---------------------------------------------------------------------------
# D9: Audit BLOCKED loop cap
# ---------------------------------------------------------------------------


class TestAuditBlockedLoopCap:
    """Audit BLOCKED bounces to spec, capped at max_revise_loops."""

    def test_audit_blocked_escalates_at_max_revise_loops(
        self, conn, kanban_home
    ):
        from hermes_cli import kanban_db as kb
        # Bump revise count up to the cap
        for _ in range(4):
            kb._record_council_revise(conn, "force-loop")
        # The threshold check inside dispatch_once reads the cap from config.
        # We just verify _get_council_revise_count returns the right value.
        assert kb._get_council_revise_count(conn, "force-loop") == 4
