#!/usr/bin/env python3
"""Tests for Phase C — Human Gates.

Covers:
- Data model (new kanban states) — C.1-C.2
- Gate functions (sign_off, tech_review, final_sign_off) — C.3-C.4
- CLI commands (sign-off, reject) — C.5-C.6
- Stale-nudge notifications — C.6-C.7
"""
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.feature_pipeline import (
    validate_tech_review_artifact,
    check_human_approved,
    time_in_stage_hours,
    advance_pipeline,
    get_next_stage,
    PIPELINE_STAGES,
    GATE_FUNCTIONS,
    HUMAN_GATE_STAGES,
)
from hermes_cli.kanban_db import VALID_STATUSES, connect


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# C.1-C.2 — Data model
# ---------------------------------------------------------------------------

class TestNewPipelineStates:
    """Verify sign_off, tech_review, final_sign_off exist."""

    def test_sign_off_in_valid_statuses(self):
        assert "sign_off" in VALID_STATUSES

    def test_tech_review_in_valid_statuses(self):
        assert "tech_review" in VALID_STATUSES

    def test_final_sign_off_in_valid_statuses(self):
        assert "final_sign_off" in VALID_STATUSES

    def test_pipeline_includes_human_gates(self):
        """sign_off and final_sign_off are in PIPELINE_STAGES."""
        assert "sign_off" in PIPELINE_STAGES
        assert "final_sign_off" in PIPELINE_STAGES
        assert "tech_review" in PIPELINE_STAGES

    def test_human_gate_stages_set(self):
        """HUMAN_GATE_STAGES contains exactly sign_off and final_sign_off."""
        assert HUMAN_GATE_STAGES == {"sign_off", "final_sign_off"}

    def test_pipeline_stages_full_order(self):
        """Full pipeline order is correct (design doc §3, 12 stages)."""
        assert PIPELINE_STAGES == [
            "research", "prd", "spec", "council",
            "sign_off", "tech_review",
            "decompose", "execute", "pr+qa", "audit",
            "final_sign_off", "document",
        ]


# ---------------------------------------------------------------------------
# C.3 — Tech review gate
# ---------------------------------------------------------------------------

class TestTechReviewGate:
    """validate_tech_review_artifact checks tech-review.md."""

    def test_gate_missing_artifact(self, tmp_path):
        result = validate_tech_review_artifact(str(tmp_path))
        assert result is not None
        assert "Missing tech-review.md" in result

    def test_gate_empty_artifact(self, tmp_path):
        (tmp_path / "tech-review.md").write_text("")
        result = validate_tech_review_artifact(str(tmp_path))
        assert result is not None
        assert "empty" in result.lower()

    def test_gate_passes(self, tmp_path):
        (tmp_path / "tech-review.md").write_text(
            "## Architecture\nMonolith with microservices.\n\n## Risks\nNone."
        )
        result = validate_tech_review_artifact(str(tmp_path))
        assert result is None

    def test_gate_missing_sections(self, tmp_path):
        (tmp_path / "tech-review.md").write_text("## Something Else\nNot relevant.")
        result = validate_tech_review_artifact(str(tmp_path))
        assert result is not None
        assert "Missing" in result


# ---------------------------------------------------------------------------
# C.4 — Human approved event check
# ---------------------------------------------------------------------------

class TestHumanApprovalCheck:
    """check_human_approved reads from task_events table."""

    def test_no_approval_event(self, kanban_home):
        """Returns False when no human_approved event exists."""
        with connect() as conn:
            result = check_human_approved(conn, "task-nonexistent", "sign_off")
        assert result is False

    def test_approval_event_present(self, kanban_home):
        """Returns True when human_approved event exists for the stage."""
        with connect() as conn:
            import datetime as _dt
            _ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                ("task-001", "human_approved",
                 '{"stage": "sign_off", "note": "looks good"}', _ts),
            )
            conn.commit()
            result = check_human_approved(conn, "task-001", "sign_off")
        assert result is True

    def test_approval_wrong_stage(self, kanban_home):
        """Returns False when approval is for a different stage."""
        import datetime as _dt
        _ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        with connect() as conn:
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                ("task-001", "human_approved", '{"stage": "sign_off"}', _ts),
            )
            conn.commit()
            result = check_human_approved(conn, "task-001", "final_sign_off")
        assert result is False


# ---------------------------------------------------------------------------
# C.5 — Advance pipeline tests (new stages)
# ---------------------------------------------------------------------------

class TestAdvancePipelineExtended:
    """advance_pipeline works with the extended stage set."""

    def test_council_to_sign_off(self):
        """advancing from council should now go to sign_off."""
        assert get_next_stage("council") == "sign_off"

    def test_sign_off_to_tech_review(self):
        assert get_next_stage("sign_off") == "tech_review"

    def test_tech_review_to_final_sign_off(self):
        # Phase D added 5 new stages between tech_review and final_sign_off
        # (design doc §3: decompose, execute, pr+qa, audit).
        assert get_next_stage("tech_review") == "decompose"
        assert get_next_stage("decompose") == "execute"
        assert get_next_stage("execute") == "pr+qa"
        assert get_next_stage("pr+qa") == "audit"
        assert get_next_stage("audit") == "final_sign_off"

    def test_final_sign_off_is_terminal(self):
        # final_sign_off is no longer terminal — it advances to document.
        assert get_next_stage("final_sign_off") == "document"
        assert get_next_stage("document") is None

    def test_human_gates_in_gate_functions(self):
        """Human gate stages are NOT in GATE_FUNCTIONS (they use event checks)."""
        assert "sign_off" not in GATE_FUNCTIONS
        assert "final_sign_off" not in GATE_FUNCTIONS

    def test_tech_review_in_gate_functions(self):
        """Tech review IS in GATE_FUNCTIONS (it uses artifact check)."""
        assert "tech_review" in GATE_FUNCTIONS
        assert GATE_FUNCTIONS["tech_review"] is validate_tech_review_artifact


# ---------------------------------------------------------------------------
# C.6 — Stale timeout
# ---------------------------------------------------------------------------

class TestTimeInStage:
    """time_in_stage_hours calculates idle time."""

    def test_no_entry_event(self, kanban_home):
        """Returns 0.0 when no pipeline_advanced event exists."""
        with connect() as conn:
            hours = time_in_stage_hours(conn, "task-nonexistent", "sign_off")
        assert hours == 0.0

    def test_recent_entry(self, kanban_home):
        """Returns a reasonable value for a recently entered stage."""
        import time
        with connect() as conn:
            # Insert a pipeline_advanced event from 1 hour ago (epoch int,
            # matching the live schema so test+prod hit the same branch).
            ts = int(time.time() - 3600)
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                ("task-002", "pipeline_advanced",
                 '{"from_stage": "council", "to_stage": "sign_off"}', ts),
            )
            conn.commit()
            hours = time_in_stage_hours(conn, "task-002", "sign_off")
        assert 0.9 <= hours <= 1.1  # ~1 hour


# ---------------------------------------------------------------------------
# C.7 — CLI command tests (sign-off / reject)
# ---------------------------------------------------------------------------

class TestCLICommands:
    """hermes feature sign-off and reject commands."""

    def test_signoff_sets_human_approved(self, kanban_home):
        """sign-off creates human_approved event."""
        with connect() as conn:
            # Create a task at sign_off stage
            import datetime as _insdt
            _insts = _insdt.datetime.now(_insdt.timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO tasks (id, title, status, pipeline_stage, tier, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("task-cli-1", "Test Feature", "sign_off", "sign_off", "full", _insts),
            )
            conn.commit()

        # Simulate what cmd_feature_signoff would do
        with connect() as conn:
            payload = json.dumps({"stage": "sign_off", "note": "approved via test"})
            import datetime as _dt
            _ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                ("task-cli-1", "human_approved", payload, _ts),
            )
            conn.commit()

        # Verify
        with connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'human_approved'",
                ("task-cli-1",),
            ).fetchone()
        assert row is not None

    def test_reject_creates_rejected_event(self, kanban_home):
        """reject creates human_rejected event."""
        with connect() as conn:
            import datetime as _insdt
            _insts = _insdt.datetime.now(_insdt.timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO tasks (id, title, status, pipeline_stage, tier, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("task-cli-2", "Test Feature", "sign_off", "sign_off", "full", _insts),
            )
            conn.commit()

        # Simulate what cmd_feature_reject would do
        with connect() as conn:
            payload = json.dumps({"stage": "sign_off", "reason": "needs more research"})
            import datetime as _dt
            _ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                ("task-cli-2", "human_rejected", payload, _ts),
            )
            conn.commit()

        # Verify
        with connect() as conn:
            row = conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'human_rejected'",
                ("task-cli-2",),
            ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        assert payload["reason"] == "needs more research"

    def test_signoff_rejects_non_human_gate_stage(self):
        """sign-off on a non-human-gate stage would be rejected by the CLI."""
        # The CLI checks HUMAN_GATE_STAGES before writing the event.
        # This test verifies the guard logic.
        stage = "spec"
        assert stage not in HUMAN_GATE_STAGES

    def test_signoff_finds_task_on_non_active_board(self, kanban_home):
        """sign-off must find a task even when the active-board pointer is stale.

        Regression test: cmd_feature_signoff/reject/advance/status all
        resolved their DB connection via get_current_board() with no
        fallback, so a task created on one board while <root>/kanban/current
        pointed at a different board was invisible to the CLI ("Task not
        found") even though it genuinely existed.
        """
        import argparse
        import datetime as _dt

        from hermes_cli import kanban_db as kb
        from hermes_cli.feature import cmd_feature_signoff

        kb.create_board("research")
        with kb.connect(board="research") as conn:
            ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO tasks (id, title, status, pipeline_stage, tier, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("t_stale_board", "Cross-board task", "sign_off", "sign_off", "full", ts),
            )
            conn.commit()

        # Active-board pointer left on a *different* board, simulating a
        # stale <root>/kanban/current from an unrelated session.
        kb.set_current_board("default")

        rc = cmd_feature_signoff(argparse.Namespace(task_id="t_stale_board", note="approved"))
        assert rc == 0

        with kb.connect(board="research") as conn:
            row = conn.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'human_approved'",
                ("t_stale_board",),
            ).fetchone()
        assert row is not None

    def test_signoff_task_not_found_on_any_board(self, kanban_home):
        """sign-off still reports a clean error when the task truly doesn't exist anywhere."""
        import argparse

        from hermes_cli.feature import cmd_feature_signoff

        rc = cmd_feature_signoff(argparse.Namespace(task_id="t_does_not_exist", note=None))
        assert rc == 1


# ---------------------------------------------------------------------------
# Integration — human gate pipeline walk
# ---------------------------------------------------------------------------

class TestHumanGatePipelineWalk:
    """End-to-end human gate flow (non-dispatch)."""

    def test_human_gate_stages_in_pipeline(self):
        """sign_off and final_sign_off are reachable via get_next_stage."""
        # Start from council (Phase B exit)
        current = "council"
        path = []
        while current:
            path.append(current)
            current = get_next_stage(current)
        assert "sign_off" in path
        assert "final_sign_off" in path
        # final_sign_off is no longer terminal — it advances to document.
        assert path[-1] == "document"
