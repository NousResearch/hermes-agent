"""Integration tests: Idea Box flow wired to approval state machine.

End-to-end tests combining the idea_box module (capture, dedup, confirm/reject)
with the governance/approvals module (ledger, manager) to verify:
  - A captured idea cannot auto-create a Kanban task (AC1)
  - Only explicit approval creates a task (AC1)
  - Deduplication works across the full flow (AC3)
  - Audit trail is complete (AC4)
  - Reject/Amend flows work end-to-end (AC4)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from idea_box.models import IdeaCard, IdeaStatus, SourceRef, DedupResult
from idea_box.dedup import DedupChecker
from idea_box.flow import IdeaBoxFlow, FlowResult
from governance.approvals.ledger import ApprovalLedger, generate_fingerprint
from governance.approvals.manager import ApprovalWorkflowManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_source(**kwargs) -> SourceRef:
    defaults = dict(
        platform="discord",
        channel_id="111222333",
        message_id="444555666",
        user_id="999888777",
        user_name="sahil",
        channel_name="idea-box",
        raw_text="/idea Build a new analytics dashboard",
    )
    defaults.update(kwargs)
    return SourceRef(**defaults)


def _make_checker(is_dup: bool = False) -> MagicMock:
    checker = MagicMock(spec=DedupChecker)
    checker.check.return_value = DedupResult(
        is_duplicate=is_dup,
        matches=[{"source": "kanban", "title": "existing task", "ref_id": "t_001", "score": 0.8}] if is_dup else [],
        checked_sources=["kanban", "session_search", "mnemosyne"],
    )
    return checker


@pytest.fixture
def tmp_ledger_dir(tmp_path):
    from governance.approvals import ledger as ledger_mod
    ledger_mod.LEDGER_DIR = tmp_path
    ledger_mod.STATE_FILE = tmp_path / "approval_state.json"
    ledger_mod.AUDIT_FILE = tmp_path / "approval_audit.jsonl"
    return tmp_path


@pytest.fixture
def approval_manager(tmp_ledger_dir):
    return ApprovalWorkflowManager(default_board="default")


# ---------------------------------------------------------------------------
# AC1: No task created without explicit approval
# ---------------------------------------------------------------------------

class TestNoAutoCreation:
    def test_capture_does_not_submit_for_approval(self, approval_manager, tmp_ledger_dir):
        """Capturing an idea should not create an approval request."""
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        result = flow.capture("Build analytics dashboard", _make_source())

        # The approval ledger should have no entries
        from governance.approvals import ledger as ledger_mod
        if ledger_mod.STATE_FILE.exists():
            state = json.loads(ledger_mod.STATE_FILE.read_text(encoding="utf-8"))
            assert len(state) == 0, "No approval entries should exist after capture"
        # FlowResult should be present/presentation, not confirmed
        assert result.action in ("present", "duplicate")
        assert result.kanban_task_id is None

    def test_capture_does_not_create_kanban_task(self, approval_manager, tmp_ledger_dir):
        """Capture must not trigger any Kanban task creation."""
        mock_create = MagicMock()
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        flow.capture("Build a feature", _make_source())
        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# AC2: Approval creates task with triage metadata and dedup reference
# ---------------------------------------------------------------------------

class TestApprovalCreatesTask:
    def test_full_flow_capture_then_approve(self, approval_manager, tmp_ledger_dir):
        """Full flow: capture → submit for approval → approve → create task."""
        # 1. Capture the idea
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        capture_result = flow.capture("Build analytics dashboard", _make_source())
        assert capture_result.action == "present"

        # 2. Submit for approval
        fingerprint = generate_fingerprint(capture_result.card.summary)
        submit_result = approval_manager.ledger.submit_for_approval(
            triage_id="idea-001",
            summary=capture_result.card.summary,
            fingerprint=fingerprint,
            routing={"board": "default", "assignee": "octacon"},
            metadata={
                "source": capture_result.card.source.to_dict(),
                "tags": capture_result.card.tags,
            },
        )
        assert submit_result["status"] == "ok"

        # 3. Approve
        summary_dict = {
            "title": capture_result.card.summary,
            "classification": {"category": "feature"},
            "confidence": "high",
            "effort": "m",
            "recommendation": "proceed",
            "routing": {"specialist": "octacon"},
            "reasoning": "User-requested analytics dashboard",
            "source": {
                "url": f"discord://{capture_result.card.source.channel_id}/{capture_result.card.source.message_id}",
                "source_type": capture_result.card.source.platform,
                "title": capture_result.card.summary[:80],
                "provenance": {
                    "submitted_by": capture_result.card.source.user_name,
                    "submitted_at": capture_result.card.created_at,
                },
                "content_snippet": capture_result.card.summary,
            },
        }

        with patch.object(approval_manager, "_task_exists", return_value=(False, None)):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                mock_kb.create_task.return_value = "t_integ_001"
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    approve_result = approval_manager.handle_approval_action(
                        triage_id="idea-001",
                        action="approve",
                        actor="sahil",
                        reason="Approved via Discord button",
                        summary_dict=summary_dict,
                    )

        assert approve_result["status"] == "ok"
        assert approve_result["action"] == "created"
        assert approve_result["task_id"] == "t_integ_001"

        # Verify task was created with triage=True equivalent (assignee from routing)
        mock_kb.create_task.assert_called_once()

    def test_task_body_includes_source_provenance(self, approval_manager, tmp_ledger_dir):
        """The Kanban task body includes source provenance from the idea card."""
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        src = _make_source(user_name="sahil_discord", message_id="msg_789")
        capture_result = flow.capture("Real-time dashboard", src)

        fingerprint = generate_fingerprint(capture_result.card.summary)
        approval_manager.ledger.submit_for_approval(
            triage_id="idea-prov",
            summary=capture_result.card.summary,
            fingerprint=fingerprint,
            routing={"board": "default", "assignee": "octacon"},
            metadata={"source": capture_result.card.source.to_dict()},
        )

        summary_dict = {
            "title": capture_result.card.summary,
            "source": {
                "url": f"discord://{src.channel_id}/{src.message_id}",
                "source_type": "discord",
                "provenance": {
                    "submitted_by": src.user_name,
                },
            },
        }

        body = approval_manager._format_body(summary_dict)
        assert "sahil_discord" in body
        assert "discord" in body.lower()


# ---------------------------------------------------------------------------
# AC3: Duplicate detection prevents multiple tasks
# ---------------------------------------------------------------------------

class TestDuplicateDetectionIntegration:
    def test_duplicate_idea_links_to_existing(self, approval_manager, tmp_ledger_dir):
        """When the same idea is submitted twice, dedup detects it and
        approval links to the existing task instead of creating a new one."""
        # First idea: novel, approved, task created
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        capture1 = flow.capture("Build analytics dashboard", _make_source())
        assert capture1.action == "present"

        # Submit and approve first idea
        fp1 = generate_fingerprint(capture1.card.summary)
        approval_manager.ledger.submit_for_approval(
            triage_id="idea-dup-1",
            summary=capture1.card.summary,
            fingerprint=fp1,
            routing={"board": "default", "assignee": "octacon"},
        )

        summary_dict = {"title": capture1.card.summary}
        with patch.object(approval_manager, "_task_exists", return_value=(False, None)):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                mock_kb.create_task.return_value = "t_first_001"
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    result1 = approval_manager.handle_approval_action(
                        triage_id="idea-dup-1",
                        action="approve",
                        actor="sahil",
                        reason="Approved first",
                        summary_dict=summary_dict,
                    )

        assert result1["action"] == "created"

        # Second submission: same fingerprint → duplicate detected
        approval_manager.ledger.submit_for_approval(
            triage_id="idea-dup-2",
            summary="Same analytics dashboard idea",
            fingerprint=fp1,  # Same fingerprint!
            routing={"board": "default", "assignee": "octacon"},
        )

        with patch.object(approval_manager, "_task_exists", return_value=(True, "t_first_001")):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    result2 = approval_manager.handle_approval_action(
                        triage_id="idea-dup-2",
                        action="approve",
                        actor="sahil",
                        reason="Approved duplicate",
                        summary_dict=summary_dict,
                    )

        assert result2["action"] == "linked"
        assert result2["task_id"] == "t_first_001"
        mock_kb.create_task.assert_not_called()


# ---------------------------------------------------------------------------
# AC4: Audit trail is complete
# ---------------------------------------------------------------------------

class TestAuditTrailIntegration:
    def test_approve_produces_complete_audit_trail(self, approval_manager, tmp_ledger_dir):
        """Full approval lifecycle must produce SUBMIT + APPROVE audit entries."""
        from governance.approvals import ledger as ledger_mod

        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        capture = flow.capture("Build feature X", _make_source())

        fp = generate_fingerprint(capture.card.summary)
        approval_manager.ledger.submit_for_approval(
            triage_id="idea-audit",
            summary=capture.card.summary,
            fingerprint=fp,
            routing={"board": "default", "assignee": "octacon"},
            metadata={"source": capture.card.source.to_dict()},
        )

        with patch.object(approval_manager, "_task_exists", return_value=(False, None)):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                mock_kb.create_task.return_value = "t_audit_001"
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    approval_manager.handle_approval_action(
                        triage_id="idea-audit",
                        action="approve",
                        actor="sahil",
                        reason="Good idea",
                        summary_dict={"title": capture.card.summary},
                    )

        # Verify audit trail
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        submit_entry = json.loads(lines[0])
        approve_entry = json.loads(lines[1])
        assert submit_entry["action"] == "SUBMIT"
        assert approve_entry["action"] == "APPROVE"
        assert approve_entry["actor"] == "sahil"
        assert approve_entry["triage_id"] == "idea-audit"

    def test_reject_produces_audit_without_task(self, approval_manager, tmp_ledger_dir):
        """Reject must produce an audit entry and no Kanban task."""
        from governance.approvals import ledger as ledger_mod

        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        capture = flow.capture("Bad idea", _make_source())

        fp = generate_fingerprint(capture.card.summary)
        approval_manager.ledger.submit_for_approval(
            triage_id="idea-rej",
            summary=capture.card.summary,
            fingerprint=fp,
            routing={"board": "default", "assignee": "octacon"},
        )

        result = approval_manager.handle_approval_action(
            triage_id="idea-rej",
            action="reject",
            actor="sahil",
            reason="Out of scope",
        )

        assert result["status"] == "ok"
        assert result["action"] == "reject"

        # Verify audit trail
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        actions = [json.loads(l)["action"] for l in lines]
        assert "SUBMIT" in actions
        assert "REJECT" in actions
        assert "APPROVE" not in actions

    def test_amend_then_approve_produces_full_trail(self, approval_manager, tmp_ledger_dir):
        """Submit → Amend → Approve must produce all 3 audit entries."""
        from governance.approvals import ledger as ledger_mod

        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        capture = flow.capture("Needs refinement", _make_source())

        fp = generate_fingerprint(capture.card.summary)
        approval_manager.ledger.submit_for_approval(
            triage_id="idea-amend",
            summary=capture.card.summary,
            fingerprint=fp,
            routing={"board": "default", "assignee": "octacon"},
        )

        # Amend
        approval_manager.handle_approval_action(
            triage_id="idea-amend",
            action="amend",
            actor="sahil",
            reason="Need more detail on scope",
        )

        # Approve
        with patch.object(approval_manager, "_task_exists", return_value=(False, None)):
            with patch("governance.approvals.manager.kanban_db") as mock_kb:
                mock_kb.create_task.return_value = "t_amend_001"
                with patch("governance.approvals.manager._board_compat") as mock_compat:
                    mock_compat.resolve_board_db_str.return_value = ":memory:"
                    approval_manager.handle_approval_action(
                        triage_id="idea-amend",
                        action="approve",
                        actor="sahil",
                        reason="Refined and approved",
                        summary_dict={"title": capture.card.summary},
                    )

        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        actions = [json.loads(l)["action"] for l in lines]
        assert actions == ["SUBMIT", "AMEND", "APPROVE"]


# ---------------------------------------------------------------------------
# Data contract conformance
# ---------------------------------------------------------------------------

class TestDataContractConformance:
    def test_approval_state_has_required_fields(self, approval_manager, tmp_ledger_dir):
        """A submitted request must have all required fields per the data contract."""
        fp = generate_fingerprint("contract test idea")
        approval_manager.ledger.submit_for_approval(
            triage_id="tri-contract",
            summary="Contract test idea",
            fingerprint=fp,
            routing={"board": "default", "assignee": "octacon"},
            metadata={"source_url": "https://discord.com/channels/123/456/789"},
        )

        req = approval_manager.ledger.get_request("tri-contract")
        assert req is not None
        # Required fields from the data contract
        assert "triage_id" in req
        assert "summary" in req
        assert "fingerprint" in req
        assert "routing" in req
        assert "status" in req
        assert "created_at" in req
        assert "updated_at" in req
        assert req["status"] == "PENDING"

    def test_summary_dict_includes_triage_metadata(self, approval_manager, tmp_ledger_dir):
        """When approving, the summary_dict must carry triage metadata."""
        summary_dict = {
            "classification": {"category": "feature"},
            "confidence": "high",
            "effort": "s",
            "recommendation": "proceed",
            "routing": {"specialist": "octacon"},
            "reasoning": "High-value user request",
            "source": {
                "url": "https://discord.com/channels/123/456/789",
                "source_type": "discord",
                "provenance": {
                    "submitted_by": "sahil",
                    "submitted_at": "2026-08-11T12:00:00Z",
                },
            },
        }

        fp = generate_fingerprint("metadata test")
        approval_manager.ledger.submit_for_approval(
            triage_id="tri-meta",
            summary="Metadata test",
            fingerprint=fp,
            routing={"board": "default", "assignee": "octacon"},
        )

        body = approval_manager._format_body(summary_dict)
        # Verify triage metadata fields appear in the body
        assert "feature" in body
        assert "high" in body
        assert "proceed" in body
        assert "octacon" in body
        assert "sahil" in body
        assert "discord" in body.lower()