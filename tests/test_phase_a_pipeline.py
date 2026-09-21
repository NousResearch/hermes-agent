#!/usr/bin/env python3
"""Tests for Phase A — Front-Half Gates (feature pipeline).

Covers:
- New kanban states (research, prd, spec, council)
- Pipeline stage migration (pipeline_stage column)
- Gate functions (intake, research, prd, spec)
- Artifact storage helpers
- Pipeline state machine transitions
"""
import json
import os
import sqlite3
import tempfile
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.feature_pipeline import (
    validate_intake_brief,
    validate_research_artifact,
    validate_prd_artifact,
    validate_spec_artifact,
    get_artifact_path,
    write_artifact,
    read_artifact,
    advance_pipeline,
    get_next_stage,
    PIPELINE_STAGES,
)


# ---------------------------------------------------------------------------
# Phase A.1 — Data model tests
# ---------------------------------------------------------------------------

class TestNewPipelineStates:
    """Verify new feature-pipeline states exist in VALID_STATUSES."""

    def test_research_in_valid_statuses(self):
        assert "research" in kb.VALID_STATUSES

    def test_prd_in_valid_statuses(self):
        assert "prd" in kb.VALID_STATUSES

    def test_spec_in_valid_statuses(self):
        assert "spec" in kb.VALID_STATUSES

    def test_council_in_valid_statuses(self):
        assert "council" in kb.VALID_STATUSES

    def test_existing_states_unchanged(self):
        """Original states must not be removed."""
        expected = {"triage", "todo", "scheduled", "ready", "running",
                    "blocked", "review", "done", "archived", "backlog"}
        assert expected.issubset(kb.VALID_STATUSES)


# ---------------------------------------------------------------------------
# Phase A.1 — Pipeline stage migration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class TestPipelineStageMigration:
    """Verify pipeline_stage column is added by migration."""

    def test_pipeline_stage_column_exists_on_fresh_db(self, kanban_home):
        """A fresh DB should have the pipeline_stage column."""
        with kb.connect() as conn:
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
            assert "pipeline_stage" in cols

    def test_pipeline_stage_null_by_default(self, kanban_home):
        """New tasks should have NULL pipeline_stage."""
        with kb.connect() as conn:
            kb.create_task(conn, title="test task", assignee="test", triage=True)
            row = conn.execute("SELECT pipeline_stage FROM tasks LIMIT 1").fetchone()
            assert row["pipeline_stage"] is None

    def test_pipeline_stage_accepts_valid_values(self, kanban_home):
        """pipeline_stage should accept all 12 pipeline stages (full path)."""
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="test", assignee="test", triage=True)
            for stage in ("research", "prd", "spec", "council", "sign_off",
                          "tech_review", "decompose", "execute", "pr+qa",
                          "audit", "final_sign_off", "document"):
                conn.execute("UPDATE tasks SET pipeline_stage = ? WHERE id = ?", (stage, task_id))
                row = conn.execute("SELECT pipeline_stage FROM tasks WHERE id = ?", (task_id,)).fetchone()
                assert row["pipeline_stage"] == stage


# ---------------------------------------------------------------------------
# Gate function tests
# ---------------------------------------------------------------------------

class TestIntakeGate:
    """validate_intake_brief: checks body has problem + success criteria."""

    def test_passes_with_both_sections(self):
        body = "## Problem\nBuild a thing.\n\n## Success Criteria\nIt works."
        assert validate_intake_brief(body) is None

    def test_fails_without_problem(self):
        body = "## Success Criteria\nIt works."
        result = validate_intake_brief(body)
        assert result is not None
        assert "Problem" in result

    def test_fails_without_success_criteria(self):
        body = "## Problem\nBuild a thing."
        result = validate_intake_brief(body)
        assert result is not None
        assert "Success Criteria" in result

    def test_fails_with_empty_body(self):
        assert validate_intake_brief("") is not None

    def test_fails_with_none_body(self):
        assert validate_intake_brief(None) is not None


class TestResearchGate:
    """validate_research_artifact: checks research-brief.md exists with content."""

    def test_passes_with_valid_artifact(self, tmp_path):
        artifact_dir = tmp_path / "task-001"
        artifact_dir.mkdir()
        (artifact_dir / "research-brief.md").write_text(
            "# Research Brief\n\n## Findings\nNo prior art found.\n"
        )
        assert validate_research_artifact(str(artifact_dir)) is None

    def test_fails_without_artifact(self, tmp_path):
        artifact_dir = tmp_path / "task-002"
        artifact_dir.mkdir()
        result = validate_research_artifact(str(artifact_dir))
        assert result is not None
        assert "research-brief.md" in result

    def test_fails_with_empty_artifact(self, tmp_path):
        artifact_dir = tmp_path / "task-003"
        artifact_dir.mkdir()
        (artifact_dir / "research-brief.md").write_text("")
        assert validate_research_artifact(str(artifact_dir)) is not None

    def test_fails_without_findings_section(self, tmp_path):
        artifact_dir = tmp_path / "task-004"
        artifact_dir.mkdir()
        (artifact_dir / "research-brief.md").write_text("# Research Brief\n\nSome notes.\n")
        result = validate_research_artifact(str(artifact_dir))
        assert result is not None
        assert "Findings" in result


class TestPrdGate:
    """validate_prd_artifact: checks prd.md has required sections."""

    def test_passes_with_all_sections(self, tmp_path):
        artifact_dir = tmp_path / "task-005"
        artifact_dir.mkdir()
        (artifact_dir / "prd.md").write_text(
            "# PRD\n\n## Problem\nBuild a thing.\n\n## Users\nPMs.\n\n"
            "## Scope\nIn scope: everything.\n\n## Out of Scope\nNothing.\n\n"
            "## Metrics\nSuccess rate > 90%.\n"
        )
        assert validate_prd_artifact(str(artifact_dir)) is None

    def test_fails_without_problem(self, tmp_path):
        artifact_dir = tmp_path / "task-006"
        artifact_dir.mkdir()
        (artifact_dir / "prd.md").write_text(
            "# PRD\n\n## Users\nPMs.\n\n## Scope\nIn scope.\n\n"
            "## Out of Scope\nNone.\n\n## Metrics\n90%.\n"
        )
        result = validate_prd_artifact(str(artifact_dir))
        assert result is not None
        assert "Problem" in result

    def test_fails_without_prd_file(self, tmp_path):
        artifact_dir = tmp_path / "task-007"
        artifact_dir.mkdir()
        result = validate_prd_artifact(str(artifact_dir))
        assert result is not None
        assert "prd.md" in result


class TestSpecGate:
    """validate_spec_artifact: checks spec.md has required sections."""

    def test_passes_with_all_sections(self, tmp_path):
        artifact_dir = tmp_path / "task-008"
        artifact_dir.mkdir()
        (artifact_dir / "spec.md").write_text(
            "# Spec\n\n## Architecture\nMonolith.\n\n"
            "## Interfaces\nREST API.\n\n## Test Strategy\nUnit + integration.\n"
        )
        assert validate_spec_artifact(str(artifact_dir)) is None

    def test_fails_without_architecture(self, tmp_path):
        artifact_dir = tmp_path / "task-009"
        artifact_dir.mkdir()
        (artifact_dir / "spec.md").write_text(
            "# Spec\n\n## Interfaces\nREST.\n\n## Test Strategy\nUnit.\n"
        )
        result = validate_spec_artifact(str(artifact_dir))
        assert result is not None
        assert "Architecture" in result

    def test_fails_without_spec_file(self, tmp_path):
        artifact_dir = tmp_path / "task-010"
        artifact_dir.mkdir()
        result = validate_spec_artifact(str(artifact_dir))
        assert result is not None
        assert "spec.md" in result


# ---------------------------------------------------------------------------
# Artifact helper tests
# ---------------------------------------------------------------------------

class TestArtifactHelpers:
    """Test artifact path resolution and read/write helpers."""

    def test_get_artifact_path(self, tmp_path):
        path = get_artifact_path(str(tmp_path), "task-001", "research-brief.md")
        assert path.endswith("task-001/research-brief.md")
        assert str(tmp_path) in path

    def test_write_and_read_artifact(self, tmp_path):
        content = "# Research Brief\n\n## Findings\nNothing found.\n"
        write_artifact(str(tmp_path), "task-002", "research-brief.md", content)
        result = read_artifact(str(tmp_path), "task-002", "research-brief.md")
        assert result == content

    def test_read_nonexistent_returns_none(self, tmp_path):
        result = read_artifact(str(tmp_path), "task-003", "research-brief.md")
        assert result is None


# ---------------------------------------------------------------------------
# Pipeline state machine tests
# ---------------------------------------------------------------------------

class TestPipelineStateMachine:
    """Test pipeline stage transitions and gate checks."""

    def test_get_next_stage(self):
        assert get_next_stage("research") == "prd"
        assert get_next_stage("prd") == "spec"
        assert get_next_stage("spec") == "council"
        assert get_next_stage("council") == "sign_off"
        assert get_next_stage("sign_off") == "tech_review"
        assert get_next_stage("tech_review") == "decompose"
        assert get_next_stage("decompose") == "execute"
        assert get_next_stage("execute") == "pr+qa"
        assert get_next_stage("pr+qa") == "audit"
        assert get_next_stage("audit") == "final_sign_off"
        assert get_next_stage("final_sign_off") == "document"
        assert get_next_stage("document") is None

    def test_get_next_stage_invalid(self):
        assert get_next_stage("invalid") is None

    def test_advance_pipeline_research_passes(self, tmp_path):
        """Research gate passes when artifact exists with findings."""
        artifact_dir = tmp_path / "task-010"
        artifact_dir.mkdir()
        (artifact_dir / "research-brief.md").write_text(
            "# Research\n\n## Findings\nNothing found.\n"
        )
        result = advance_pipeline("task-010", "research", str(tmp_path))
        assert result["advanced"] is True
        assert result["gate_passed"] is True
        assert result["to_stage"] == "prd"

    def test_advance_pipeline_research_fails(self, tmp_path):
        """Research gate fails without artifact."""
        artifact_dir = tmp_path / "task-011"
        artifact_dir.mkdir()
        result = advance_pipeline("task-011", "research", str(tmp_path))
        assert result["advanced"] is False
        assert result["gate_passed"] is False
        assert result["gate_message"] is not None

    def test_advance_pipeline_prd_passes(self, tmp_path):
        """PRD gate passes when artifact has all required sections."""
        artifact_dir = tmp_path / "task-012"
        artifact_dir.mkdir()
        (artifact_dir / "prd.md").write_text(
            "# PRD\n\n## Problem\nX.\n\n## Users\nY.\n\n"
            "## Scope\nZ.\n\n## Out of Scope\nW.\n\n## Metrics\n90%.\n"
        )
        result = advance_pipeline("task-012", "prd", str(tmp_path))
        assert result["advanced"] is True
        assert result["gate_passed"] is True
        assert result["to_stage"] == "spec"

    def test_advance_pipeline_spec_passes(self, tmp_path):
        """Spec gate passes when artifact has all required sections."""
        artifact_dir = tmp_path / "task-013"
        artifact_dir.mkdir()
        (artifact_dir / "spec.md").write_text(
            "# Spec\n\n## Architecture\nA.\n\n## Interfaces\nB.\n\n## Test Strategy\nC.\n"
        )
        result = advance_pipeline("task-013", "spec", str(tmp_path))
        assert result["advanced"] is True
        assert result["gate_passed"] is True
        assert result["to_stage"] == "council"

    def test_advance_pipeline_council_approved(self, tmp_path):
        """Council gate passes when verdict artifact says APPROVED."""
        artifact_dir = tmp_path / "task-014"
        artifact_dir.mkdir()
        (artifact_dir / "prd.md").write_text("## Problem\nX\n\n## Users\nY")
        (artifact_dir / "spec.md").write_text("## Architecture\nA\n\n## Interfaces\nB\n\n## Test Strategy\nC")
        # C-a: the gate reads the machine-readable JSON verdict; the .md is
        # only a human-readable companion.
        (artifact_dir / "council-verdict.json").write_text(
            json.dumps({"verdict": "APPROVED", "issues": []})
        )
        result = advance_pipeline("task-014", "council", str(tmp_path))
        assert result["advanced"] is True
        assert result["gate_passed"] is True
        assert result["to_stage"] == "sign_off"
        assert result["gate_message"] is None

    def test_advance_pipeline_council_revise(self, tmp_path):
        """Council gate fails when verdict artifact says REVISE."""
        artifact_dir = tmp_path / "task-015"
        artifact_dir.mkdir()
        artifact_dir.mkdir(exist_ok=True)
        (artifact_dir / "council-verdict.json").write_text(
            json.dumps({
                "verdict": "REVISE",
                "issues": [
                    {"severity": "high", "description": "Missing error handling in API layer"},
                    {"severity": "medium", "description": "Test strategy too vague"},
                ],
            })
        )
        result = advance_pipeline("task-015", "council", str(tmp_path))
        assert result["advanced"] is False
        assert result["gate_passed"] is False
        assert "REVISE" in result["gate_message"]
        assert "[HIGH]" in result["gate_message"]
        assert "[MEDIUM]" in result["gate_message"]

    def test_pipeline_stages_order(self):
        """Pipeline stages are in correct order (design doc §3)."""
        assert PIPELINE_STAGES == [
            "research", "prd", "spec", "council",
            "sign_off", "tech_review",
            "decompose", "execute", "pr+qa", "audit",
            "final_sign_off", "document",
        ]


# ---------------------------------------------------------------------------
# Phase A.9 — Notifier event tests (gate_failed)
# ---------------------------------------------------------------------------

class TestGateFailedEvent:
    """Verify gate_failed events are written when pipeline gates fail."""

    def test_gate_failed_event_written_on_gate_failure(self, kanban_home):
        """dispatch_once with dry_run=False writes gate_failed when gate fails."""
        import json
        with kb.connect() as conn:
            # Create a task in research stage with no artifact (gate will fail)
            tid = kb.create_task(
                conn, title="gate-fail test",
                body="## Problem\nTest.\n## Success Criteria\nWorks.",
                assignee="test", triage=True,
            )
            # Move to research stage
            conn.execute(
                "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                ("research", "research", tid),
            )
            # Run dispatch — gate will fail because no artifact exists
            result = kb.dispatch_once(conn, dry_run=False)
            # Check that gate_failed event was written
            events = kb.list_events(conn, tid)
            gate_fail_events = [e for e in events if e.kind == "gate_failed"]
            assert len(gate_fail_events) >= 1
            assert gate_fail_events[0].payload["stage"] == "research"
            assert "research-brief.md" in gate_fail_events[0].payload.get("reason", "")

    def test_gate_failed_event_not_written_on_dry_run(self, kanban_home):
        """dry_run=True should NOT write gate_failed events."""
        with kb.connect() as conn:
            tid = kb.create_task(
                conn, title="dry-run gate test",
                body="## Problem\nTest.\n## Success Criteria\nWorks.",
                assignee="test", triage=True,
            )
            conn.execute(
                "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                ("research", "research", tid),
            )
            result = kb.dispatch_once(conn, dry_run=True)
            events = kb.list_events(conn, tid)
            gate_fail_events = [e for e in events if e.kind == "gate_failed"]
            assert len(gate_fail_events) == 0

    def test_gate_failed_event_payload_has_stage_and_reason(self, kanban_home):
        """gate_failed payload contains stage and reason fields."""
        with kb.connect() as conn:
            tid = kb.create_task(
                conn, title="payload check",
                body="## Problem\nTest.\n## Success Criteria\nWorks.",
                assignee="test", triage=True,
            )
            conn.execute(
                "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                ("research", "research", tid),
            )
            kb.dispatch_once(conn, dry_run=False)
            events = kb.list_events(conn, tid)
            gate_fail_events = [e for e in events if e.kind == "gate_failed"]
            assert len(gate_fail_events) >= 1
            ev = gate_fail_events[0]
            assert "stage" in ev.payload
            assert "reason" in ev.payload
            assert isinstance(ev.payload["stage"], str)
            assert isinstance(ev.payload["reason"], str)


# ---------------------------------------------------------------------------
# Phase A.10 — Integration: full pipeline walk
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """End-to-end: triage → research → prd → spec → council."""

    def _create_artifact(self, stage, artifact_dir):
        """Create a minimal passing artifact for the given stage."""
        from pathlib import Path
        ad = Path(artifact_dir)
        ad.mkdir(parents=True, exist_ok=True)
        if stage == "research":
            (ad / "research-brief.md").write_text(
                "# Research Brief\n\n## Problem\nTest problem.\n\n"
                "## Findings\nKey finding.\n\n"
                "## Alternatives Considered\nAlt A, Alt B.\n\n"
                "## Recommendation\nUse Alt A.\n\n"
                "## Evidence\nSource 1.\n\n"
                "## Cost Analysis\nMinimal.\n\n"
                "## Confidence\nHigh.\n"
            )
        elif stage == "prd":
            (ad / "prd.md").write_text(
                "# PRD\n\n## Problem\nTest.\n\n## Users\nDevs.\n\n"
                "## Scope\nIn scope.\n\n## Out of Scope\nNothing.\n\n"
                "## Metrics\n95%.\n"
            )
        elif stage == "spec":
            (ad / "spec.md").write_text(
                "# Spec\n\n## Architecture\nMonolith.\n\n"
                "## Interfaces\nREST.\n\n## Test Strategy\nUnit + integration.\n"
            )

    def test_full_pipeline_walk(self, kanban_home):
        """Walk a task through all pipeline stages with artifacts."""
        from pathlib import Path
        import os

        # Set artifact base dir inside isolated home
        artifact_base = os.path.join(str(kanban_home), "feature-artifacts")
        os.environ["HERMES_HOME"] = str(kanban_home)

        with kb.connect() as conn:
            # 1. Create a tier=full intake task
            tid = kb.create_task(
                conn,
                title="Full pipeline integration test",
                body="## Problem\nTest problem.\n\n## Success Criteria\nIt works.",
                assignee="remii",  # research stage owner
                triage=True,
            )
            # Move to research stage (simulating triage processor)
            conn.execute(
                "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                ("research", "research", tid),
            )
            conn.commit()

            # 2. Stage: research → prd
            # No artifact yet → gate should fail
            result = kb.dispatch_once(conn, dry_run=False)
            events = kb.list_events(conn, tid)
            gate_fails = [e for e in events if e.kind == "gate_failed"]
            assert len(gate_fails) >= 1

            # Create research artifact
            artifact_dir = os.path.join(artifact_base, tid)
            self._create_artifact("research", artifact_dir)

            # Assign to remii (research owner)
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", ("remii", tid))
            conn.commit()

            # Dispatch again — gate should pass now
            result = kb.dispatch_once(conn, dry_run=False)
            events = kb.list_events(conn, tid)
            advanced = [e for e in events if e.kind == "pipeline_advanced"]
            research_advances = [e for e in advanced
                                if e.payload.get("from_stage") == "research"
                                and e.payload.get("to_stage") == "prd"]
            assert len(research_advances) >= 1

            # 3. Stage: prd → spec
            # Move assignee to kensei (PRD owner)
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", ("kensei", tid))
            conn.commit()

            # No PRD artifact yet → gate should fail
            result = kb.dispatch_once(conn, dry_run=False)
            events = kb.list_events(conn, tid)
            prd_gate_fails = [e for e in events
                             if e.kind == "gate_failed"
                             and e.payload.get("stage") == "prd"]
            assert len(prd_gate_fails) >= 1

            # Create PRD artifact
            self._create_artifact("prd", artifact_dir)

            # Dispatch — gate should pass
            result = kb.dispatch_once(conn, dry_run=False)
            events = kb.list_events(conn, tid)
            prd_advances = [e for e in events
                           if e.kind == "pipeline_advanced"
                           and e.payload.get("from_stage") == "prd"
                           and e.payload.get("to_stage") == "spec"]
            assert len(prd_advances) >= 1

            # 4. Stage: spec → council
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", ("octacon", tid))
            conn.commit()

            # Create spec artifact
            self._create_artifact("spec", artifact_dir)

            # Dispatch — gate should pass, pipeline complete
            result = kb.dispatch_once(conn, dry_run=False)
            events = kb.list_events(conn, tid)
            spec_advances = [e for e in events
                            if e.kind == "pipeline_advanced"
                            and e.payload.get("from_stage") == "spec"
                            and e.payload.get("to_stage") == "council"]
            assert len(spec_advances) >= 1

            # 5. Verify all event types fired
            all_kinds = {e.kind for e in events}
            assert "gate_failed" in all_kinds
            assert "pipeline_advanced" in all_kinds
            # Task should now be in council status
            task = kb.get_task(conn, tid)
            assert task.status == "council"
            assert task.pipeline_stage == "council"

    def test_full_pipeline_walk_dry_run(self, kanban_home):
        """Dry-run walk should report advancement without side effects."""
        import os
        artifact_base = os.path.join(str(kanban_home), "feature-artifacts")
        os.environ["HERMES_HOME"] = str(kanban_home)

        with kb.connect() as conn:
            tid = kb.create_task(
                conn,
                title="Dry run pipeline test",
                body="## Problem\nTest.\n\n## Success Criteria\nWorks.",
                assignee="remii",
                triage=True,
            )
            conn.execute(
                "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                ("research", "research", tid),
            )
            conn.commit()

            # Create artifact so gate passes
            artifact_dir = os.path.join(artifact_base, tid)
            self._create_artifact("research", artifact_dir)

            # Dry run — should report advancement but not change DB
            result = kb.dispatch_once(conn, dry_run=True)
            assert len(result.pipeline_advanced) >= 1
            assert result.pipeline_advanced[0][0] == tid
            assert result.pipeline_advanced[0][1] == "research"
            assert result.pipeline_advanced[0][2] == "prd"

            # DB should be unchanged (still research)
            task = kb.get_task(conn, tid)
            assert task.status == "research"
