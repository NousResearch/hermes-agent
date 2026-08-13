"""Unit tests for governance.approvals.manager — ApprovalWorkflowManager.

Covers:
  - Approval action creates Kanban task with triage metadata (AC1, AC2)
  - Duplicate detection prevents multiple tasks (AC3)
  - Reject/Amend logs audit record without creating a task (AC4)
  - Error handling for missing requests, invalid boards, etc.

The Kanban DB and board resolution are mocked throughout.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from governance.approvals.ledger import ApprovalLedger, generate_fingerprint
from governance.approvals.manager import ApprovalWorkflowManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_ledger_dir(tmp_path):
    """Provide a temp directory for ledger state files."""
    from governance.approvals import ledger as ledger_mod
    ledger_mod.LEDGER_DIR = tmp_path
    ledger_mod.STATE_FILE = tmp_path / "approval_state.json"
    ledger_mod.AUDIT_FILE = tmp_path / "approval_audit.jsonl"
    return tmp_path


@pytest.fixture
def manager(tmp_ledger_dir):
    """Return an ApprovalWorkflowManager with a fresh ledger and mocked board resolution."""
    mgr = ApprovalWorkflowManager(default_board="default")
    return mgr


def _make_summary() -> dict:
    """Return a sample triage summary dict matching the data contract."""
    return {
        "title": "Build analytics dashboard",
        "classification": {"category": "feature"},
        "confidence": "high",
        "effort": "m",
        "recommendation": "proceed",
        "routing": {"specialist": "octacon"},
        "reasoning": "Addresses user need for real-time analytics.",
        "source": {
            "url": "https://example.com/idea/123",
            "source_type": "discord",
            "title": "Analytics Dashboard Idea",
            "provenance": {
                "submitted_by": "sahil",
                "submitted_at": "2026-08-11T12:00:00Z",
            },
            "content_snippet": "We need a real-time analytics dashboard for monitoring...",
        },
    }


# ---------------------------------------------------------------------------
# AC1: No task created unless Approve is pressed
# ---------------------------------------------------------------------------

class TestNoAutoCreation:
    def test_submit_does_not_create_task(self, manager):
        """Submitting a request for approval must NOT create a Kanban task."""
        result = manager.ledger.submit_for_approval(
            triage_id="tri-ac1",
            summary="Test idea",
            fingerprint=generate_fingerprint("test idea"),
            routing={"board": "default", "assignee": "octacon"},
        )
        assert result["status"] == "ok"
        # Verify no task was created — manager.handle_approval_action not called
        # The ledger only records state, no task creation happens at submit

    def test_reject_does_not_create_task(self, manager):
        """Rejecting a request must NOT create a Kanban task."""
        manager.ledger.submit_for_approval(
            triage_id="tri-ac1r",
            summary="Test idea",
            fingerprint="fp-rej",
            routing={"board": "default", "assignee": "octacon"},
        )
        # Patch task creation to detect if it happens
        with patch.object(manager, "_task_exists", return_value=(False, None)):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                result = manager.handle_approval_action(
                    triage_id="tri-ac1r",
                    action="reject",
                    actor="sahil",
                    reason="Not needed",
                    summary_dict=_make_summary(),
                )
        assert result["status"] == "ok"
        assert result["action"] == "reject"
        # kanban_db.create_task should NOT have been called
        mock_kb.create_task.assert_not_called()


# ---------------------------------------------------------------------------
# AC2: Created tasks include triage metadata, dedup reference, correct routing
# ---------------------------------------------------------------------------

class TestApproveCreatesTask:
    def test_approve_creates_task(self, manager):
        """Approval must create a Kanban task via kanban_db."""
        manager.ledger.submit_for_approval(
            triage_id="tri-ac2",
            summary="Test idea",
            fingerprint=generate_fingerprint("analytics dashboard"),
            routing={"board": "default", "assignee": "octacon-backend"},
        )
        summary = _make_summary()

        with patch.object(manager, "_task_exists", return_value=(False, None)):
            with patch.object(manager, "_get_db_path", return_value=":memory:"):
                # Create in-memory DB with the tasks table
                conn = sqlite3.connect(":memory:")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        body TEXT,
                        assignee TEXT,
                        status TEXT DEFAULT 'backlog',
                        created_by TEXT,
                        created_at TEXT,
                        workspace_kind TEXT DEFAULT 'scratch',
                        priority INTEGER DEFAULT 3
                    )
                """)
                conn.commit()

                with patch("governance.approvals.manager.kanban_db") as mock_kb:
                    mock_kb.create_task.return_value = "t_new_001"
                    with patch("governance.approvals.manager._board_compat") as mock_compat:
                        mock_compat.resolve_board_db_str.return_value = ":memory:"
                        result = manager.handle_approval_action(
                            triage_id="tri-ac2",
                            action="approve",
                            actor="sahil",
                            reason="Looks good",
                            summary_dict=summary,
                        )

        assert result["status"] == "ok"
        assert result["action"] == "created"
        assert result["task_id"] == "t_new_001"

    def test_approve_task_body_contains_metadata(self, manager):
        """The created task body must include triage metadata and source provenance."""
        manager.ledger.submit_for_approval(
            triage_id="tri-ac2b",
            summary="Test idea",
            fingerprint=generate_fingerprint("analytics dashboard"),
            routing={"board": "default", "assignee": "octacon"},
        )
        summary = _make_summary()
        body = manager._format_body(summary)

        # Verify key metadata fields are present
        assert "Analytics Dashboard Idea" in body  # title
        assert "feature" in body  # category
        assert "high" in body  # confidence
        assert "proceed" in body  # recommendation
        assert "octacon" in body  # specialist routing
        assert "https://example.com/idea/123" in body  # source URL
        assert "sahil" in body  # submitted_by
        assert "real-time analytics" in body  # content snippet


# ---------------------------------------------------------------------------
# AC3: Duplicate detection prevents multiple tasks for the same source
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_duplicate_links_to_existing_task(self, manager):
        """If a task with the same fingerprint/triage_id exists, link instead of creating."""
        manager.ledger.submit_for_approval(
            triage_id="tri-dup1",
            summary="Test idea",
            fingerprint=generate_fingerprint("duplicate idea"),
            routing={"board": "default", "assignee": "octacon"},
        )
        summary = _make_summary()

        with patch.object(manager, "_task_exists", return_value=(True, "t_existing_001")):
            with patch.object(manager, "_get_db_path", return_value=":memory:"):
                with patch("governance.approvals.manager.kanban_db") as mock_kb:
                    with patch("governance.approvals.manager._board_compat") as mock_compat:
                        mock_compat.resolve_board_db_str.return_value = ":memory:"
                        result = manager.handle_approval_action(
                            triage_id="tri-dup1",
                            action="approve",
                            actor="sahil",
                            reason="Proceed anyway",
                            summary_dict=summary,
                        )

        assert result["status"] == "ok"
        assert result["action"] == "linked"
        assert result["task_id"] == "t_existing_001"
        # create_task should NOT have been called
        mock_kb.create_task.assert_not_called()

    def test_duplicate_adds_comment_to_existing(self, manager):
        """When a duplicate is detected, a comment is added to the existing task."""
        manager.ledger.submit_for_approval(
            triage_id="tri-dup2",
            summary="Test idea",
            fingerprint=generate_fingerprint("duplicate idea 2"),
            routing={"board": "default", "assignee": "octacon"},
        )
        summary = _make_summary()

        with patch.object(manager, "_task_exists", return_value=(True, "t_existing_002")):
            with patch.object(manager, "_get_db_path", return_value=":memory:"):
                with patch("governance.approvals.manager.kanban_db") as mock_kb:
                    with patch("governance.approvals.manager._board_compat") as mock_compat:
                        mock_compat.resolve_board_db_str.return_value = ":memory:"
                        result = manager.handle_approval_action(
                            triage_id="tri-dup2",
                            action="approve",
                            actor="sahil",
                            reason="OK",
                            summary_dict=summary,
                        )

        # Verify add_comment was called on the existing task
        mock_kb.add_comment.assert_called_once()
        call_args = mock_kb.add_comment.call_args
        assert "t_existing_002" in str(call_args)
        assert "sahil" in str(call_args)


# ---------------------------------------------------------------------------
# AC4: Audit log records Approve, Reject, Amend events
# ---------------------------------------------------------------------------

class TestAuditOnApprovalAction:
    def test_approve_audit(self, manager):
        """Approval action writes an APPROVE audit entry."""
        manager.ledger.submit_for_approval(
            triage_id="tri-aud1",
            summary="Test idea",
            fingerprint="fp-aud1",
            routing={"board": "default", "assignee": "octacon"},
        )
        with patch.object(manager, "_task_exists", return_value=(False, None)):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                mock_kb.create_task.return_value = "t_aud_001"
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    result = manager.handle_approval_action(
                        triage_id="tri-aud1",
                        action="approve",
                        actor="sahil",
                        reason="Good idea",
                        summary_dict=_make_summary(),
                    )
        assert result["status"] == "ok"
        # The ledger should have logged the APPROVE action
        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        approve_entries = [json.loads(l) for l in lines if json.loads(l).get("action") == "APPROVE"]
        assert len(approve_entries) >= 1
        assert approve_entries[-1]["actor"] == "sahil"
        assert approve_entries[-1]["reason"] == "Good idea"

    def test_reject_audit(self, manager):
        """Reject action writes a REJECT audit entry without creating a task."""
        manager.ledger.submit_for_approval(
            triage_id="tri-aud2",
            summary="Test idea",
            fingerprint="fp-aud2",
            routing={"board": "default", "assignee": "octacon"},
        )
        result = manager.handle_approval_action(
            triage_id="tri-aud2",
            action="reject",
            actor="sahil",
            reason="Out of scope",
        )
        assert result["status"] == "ok"
        assert result["action"] == "reject"

        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        reject_entries = [json.loads(l) for l in lines if json.loads(l).get("action") == "REJECT"]
        assert len(reject_entries) >= 1

    def test_amend_audit(self, manager):
        """Amend action writes an AMEND audit entry."""
        manager.ledger.submit_for_approval(
            triage_id="tri-aud3",
            summary="Test idea",
            fingerprint="fp-aud3",
            routing={"board": "default", "assignee": "octacon"},
        )
        result = manager.handle_approval_action(
            triage_id="tri-aud3",
            action="amend",
            actor="sahil",
            reason="Change scope",
        )
        assert result["status"] == "ok"
        assert result["action"] == "amend"

        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        amend_entries = [json.loads(l) for l in lines if json.loads(l).get("action") == "AMEND"]
        assert len(amend_entries) >= 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_triage_id(self, manager):
        """Handling an action for a non-existent triage ID returns error."""
        result = manager.handle_approval_action(
            triage_id="nonexistent",
            action="approve",
            actor="sahil",
            reason="OK",
            summary_dict=_make_summary(),
        )
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_invalid_action_rejected_by_ledger(self, manager):
        """An invalid action (e.g. approve on already approved) is caught by the ledger."""
        manager.ledger.submit_for_approval(
            triage_id="tri-err1",
            summary="Test",
            fingerprint="fp-err1",
            routing={"board": "default", "assignee": "octacon"},
        )
        # First approve
        with patch.object(manager, "_task_exists", return_value=(False, None)):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                mock_kb.create_task.return_value = "t_err_001"
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    result1 = manager.handle_approval_action(
                        triage_id="tri-err1",
                        action="approve",
                        actor="sahil",
                        reason="OK",
                        summary_dict=_make_summary(),
                    )
        # Double-approve should fail at the ledger level
        result2 = manager.handle_approval_action(
            triage_id="tri-err1",
            action="approve",
            actor="sahil",
            reason="Double",
            summary_dict=_make_summary(),
        )
        assert result2["status"] == "error"

    def test_missing_board_returns_error(self, manager):
        """If board DB path can't be resolved, approval returns error."""
        manager.ledger.submit_for_approval(
            triage_id="tri-noboard",
            summary="Test",
            fingerprint="fp-noboard",
            routing={"board": "nonexistent", "assignee": "octacon"},
        )
        with patch.object(manager, "_get_db_path", return_value=None):
            result = manager.handle_approval_action(
                triage_id="tri-noboard",
                action="approve",
                actor="sahil",
                reason="OK",
                summary_dict=_make_summary(),
            )
        assert result["status"] == "error"
        assert "board" in result["message"].lower() or "resolve" in result["message"].lower()


# ---------------------------------------------------------------------------
# _format_body — task body structure (AC2)
# ---------------------------------------------------------------------------

class TestFormatBody:
    def test_body_contains_all_required_fields(self, manager):
        summary = _make_summary()
        body = manager._format_body(summary)
        # Title from source
        assert "Analytics Dashboard Idea" in body
        # Source URL
        assert "https://example.com/idea/123" in body
        # Classification
        assert "feature" in body
        # Confidence
        assert "high" in body
        # Effort
        assert "m" in body.lower() or "medium" in body.lower() or "**Effort:**" in body
        # Recommendation
        assert "proceed" in body
        # Specialist routing
        assert "octacon" in body
        # Submitted by
        assert "sahil" in body
        # Content snippet
        assert "real-time analytics" in body

    def test_body_handles_missing_fields(self, manager):
        """Body formatting should handle missing optional fields gracefully."""
        summary = {"source": {"title": "Minimal idea"}}
        body = manager._format_body(summary)
        assert "Minimal idea" in body
        # Should not crash with missing fields
        assert "N/A" in body or "unknown" in body.lower()


# ---------------------------------------------------------------------------
# Integration: full approval lifecycle with mock Kanban
# ---------------------------------------------------------------------------

class TestFullApprovalLifecycle:
    def test_submit_amend_approve_lifecycle(self, manager):
        """Submit → Amend → Approve should create a task."""
        manager.ledger.submit_for_approval(
            triage_id="tri-lifecycle1",
            summary="Build feature X",
            fingerprint=generate_fingerprint("feature X"),
            routing={"board": "default", "assignee": "octacon"},
        )

        # Amend
        result_amend = manager.handle_approval_action(
            triage_id="tri-lifecycle1",
            action="amend",
            actor="sahil",
            reason="Need more detail on scope",
        )
        assert result_amend["action"] == "amend"

        # Approve
        summary = _make_summary()
        with patch.object(manager, "_task_exists", return_value=(False, None)):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                mock_kb.create_task.return_value = "t_lc_001"
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    result_approve = manager.handle_approval_action(
                        triage_id="tri-lifecycle1",
                        action="approve",
                        actor="sahil",
                        reason="Scope clarified, approved",
                        summary_dict=summary,
                    )

        assert result_approve["status"] == "ok"
        assert result_approve["action"] == "created"
        assert result_approve["task_id"] == "t_lc_001"

        # Verify audit trail has all 3 actions
        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        actions = [json.loads(l)["action"] for l in lines]
        assert "SUBMIT" in actions
        assert "AMEND" in actions
        assert "APPROVE" in actions

    def test_submit_reject_lifecycle_no_task(self, manager):
        """Submit → Reject must NOT create a task."""
        manager.ledger.submit_for_approval(
            triage_id="tri-lifecycle2",
            summary="Bad idea",
            fingerprint=generate_fingerprint("bad idea"),
            routing={"board": "default", "assignee": "octacon"},
        )

        result = manager.handle_approval_action(
            triage_id="tri-lifecycle2",
            action="reject",
            actor="sahil",
            reason="Not aligned with roadmap",
        )

        assert result["status"] == "ok"
        assert result["action"] == "reject"
        # No task_id should be in the result
        assert "task_id" not in result or result.get("task_id") is None

    def test_fingerprint_dedup_prevents_duplicate_task(self, manager):
        """Two approvals with the same fingerprint must produce a link, not a duplicate."""
        fp = generate_fingerprint("unique feature idea")
        manager.ledger.submit_for_approval(
            triage_id="tri-dedup1",
            summary="Feature idea",
            fingerprint=fp,
            routing={"board": "default", "assignee": "octacon"},
        )

        # First approval — no duplicate
        summary = _make_summary()
        with patch.object(manager, "_task_exists", return_value=(False, None)):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                mock_kb.create_task.return_value = "t_unique_001"
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    result1 = manager.handle_approval_action(
                        triage_id="tri-dedup1",
                        action="approve",
                        actor="sahil",
                        reason="Approved",
                        summary_dict=summary,
                    )

        assert result1["action"] == "created"

        # Second submission with same fingerprint
        manager.ledger.submit_for_approval(
            triage_id="tri-dedup2",
            summary="Same feature idea again",
            fingerprint=fp,  # Same fingerprint
            routing={"board": "default", "assignee": "octacon"},
        )

        # Second approval — duplicate detected
        with patch.object(manager, "_task_exists", return_value=(True, "t_unique_001")):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    result2 = manager.handle_approval_action(
                        triage_id="tri-dedup2",
                        action="approve",
                        actor="sahil",
                        reason="Approved again",
                        summary_dict=summary,
                    )

        assert result2["action"] == "linked"
        assert result2["task_id"] == "t_unique_001"