"""Unit tests for governance.approvals.ledger — ApprovalLedger state machine.

Covers every state transition (submit, approve, reject, amend) and edge cases
(duplicate submit, invalid transitions, fingerprint generation, audit log).

Acceptance criteria mapped:
  - AC1: No task created unless Approve button pressed → tested via PENDING→APPROVED
  - AC4: Audit log records Approve, Reject, Amend events with actor, timestamp, reason
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from governance.approvals.ledger import ApprovalLedger, generate_fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_ledger(tmp_path: Path) -> ApprovalLedger:
    """Return an ApprovalLedger that writes state/audit to tmp_path."""
    state_file = tmp_path / "approval_state.json"
    audit_file = tmp_path / "approval_audit.jsonl"
    with patch.object(ApprovalLedger, "__init__", lambda self: None):
        ledger = ApprovalLedger.__new__(ApprovalLedger)
        # Manually set attributes that __init__ would set
        from governance.approvals import ledger as ledger_mod
        # Override module-level paths
        ledger_mod.STATE_FILE = state_file
        ledger_mod.AUDIT_FILE = audit_file
        ledger_mod.LEDGER_DIR = tmp_path
        # Now manually initialise
        ledger.state = {}
        if state_file.exists():
            try:
                ledger.state = json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                ledger.state = {}
        return ledger


def _submit_basic(ledger: ApprovalLedger, triage_id: str = "tri-001") -> dict:
    """Submit a basic request for approval."""
    return ledger.submit_for_approval(
        triage_id=triage_id,
        summary="Test idea for analytics dashboard",
        fingerprint=generate_fingerprint("analytics dashboard idea"),
        routing={"board": "default", "assignee": "octacon"},
    )


# ---------------------------------------------------------------------------
# generate_fingerprint
# ---------------------------------------------------------------------------

class TestGenerateFingerprint:
    def test_deterministic(self):
        """Same input always produces the same fingerprint."""
        fp1 = generate_fingerprint("Build a new feature")
        fp2 = generate_fingerprint("Build a new feature")
        assert fp1 == fp2

    def test_different_inputs(self):
        """Different inputs produce different fingerprints."""
        fp1 = generate_fingerprint("Feature A")
        fp2 = generate_fingerprint("Feature B")
        assert fp1 != fp2

    def test_case_insensitive(self):
        """Fingerprint is case-insensitive (normalised)."""
        fp1 = generate_fingerprint("Build Analytics")
        fp2 = generate_fingerprint("build analytics")
        assert fp1 == fp2

    def test_whitespace_normalised(self):
        """Leading/trailing whitespace is stripped."""
        fp1 = generate_fingerprint("  Build Analytics  ")
        fp2 = generate_fingerprint("Build Analytics")
        assert fp1 == fp2

    def test_sha256_length(self):
        """Fingerprint is a 64-char hex SHA-256 hash."""
        fp = generate_fingerprint("test")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# submit_for_approval
# ---------------------------------------------------------------------------

class TestSubmitForApproval:
    def test_submit_creates_pending_entry(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        result = _submit_basic(ledger)
        assert result["status"] == "ok"
        assert result["triage_id"] == "tri-001"

    def test_submit_stores_entry_in_state(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-100")
        req = ledger.get_request("tri-100")
        assert req is not None
        assert req["status"] == "PENDING"
        assert req["triage_id"] == "tri-100"
        assert req["fingerprint"] == generate_fingerprint("analytics dashboard idea")

    def test_submit_preserves_routing(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        routing = {"board": "engineering", "assignee": "octacon-backend"}
        ledger.submit_for_approval("tri-r1", "summary", "fp1", routing)
        req = ledger.get_request("tri-r1")
        assert req["routing"]["board"] == "engineering"
        assert req["routing"]["assignee"] == "octacon-backend"

    def test_submit_stores_metadata(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        metadata = {"source_url": "https://example.com", "channel": "idea-box"}
        ledger.submit_for_approval("tri-m1", "summary", "fp1", {}, metadata=metadata)
        req = ledger.get_request("tri-m1")
        assert req["metadata"]["source_url"] == "https://example.com"

    def test_submit_writes_audit_log(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-a1")
        from governance.approvals import ledger as ledger_mod
        audit_file = ledger_mod.AUDIT_FILE
        lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry["triage_id"] == "tri-a1"
        assert entry["action"] == "SUBMIT"
        assert "timestamp" in entry

    def test_duplicate_submit_rejected(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-dup")
        result = _submit_basic(ledger, "tri-dup")
        assert result["status"] == "error"
        assert "already exists" in result["message"]

    def test_submit_persists_state_to_disk(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-p1")
        from governance.approvals import ledger as ledger_mod
        state_file = ledger_mod.STATE_FILE
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "tri-p1" in data
        assert data["tri-p1"]["status"] == "PENDING"


# ---------------------------------------------------------------------------
# process_action — APPROVE
# ---------------------------------------------------------------------------

class TestApprove:
    def test_approve_pending(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-ap1")
        ok, msg = ledger.process_action("tri-ap1", "approve", "sahil", "Looks good")
        assert ok is True
        assert "APPROVED" in msg

    def test_approve_sets_approved_fields(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-ap2")
        ledger.process_action("tri-ap2", "approve", "sahil", "Approved")
        req = ledger.get_request("tri-ap2")
        assert req["status"] == "APPROVED"
        assert req["approved_by"] == "sahil"
        assert "approved_at" in req

    def test_approve_writes_audit(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-audit1")
        ledger.process_action("tri-audit1", "approve", "sahil", "Nice idea")
        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[-1])
        assert entry["action"] == "APPROVE"
        assert entry["actor"] == "sahil"
        assert entry["reason"] == "Nice idea"
        assert entry["triage_id"] == "tri-audit1"

    def test_approve_amending(self, tmp_path):
        """An AMENDING request can also be approved."""
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-am1")
        ledger.process_action("tri-am1", "amend", "sahil", "Tweak routing")
        ok, msg = ledger.process_action("tri-am1", "approve", "sahil", "Tweaked, approved")
        assert ok is True
        assert ledger.get_request("tri-am1")["status"] == "APPROVED"

    def test_cannot_approve_already_approved(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-ap3")
        ledger.process_action("tri-ap3", "approve", "sahil", "OK")
        ok, msg = ledger.process_action("tri-ap3", "approve", "sahil", "Double approve")
        assert ok is False
        assert "Cannot approve" in msg

    def test_cannot_approve_rejected(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-rej1")
        ledger.process_action("tri-rej1", "reject", "sahil", "Nope")
        ok, msg = ledger.process_action("tri-rej1", "approve", "sahil", "Reconsider")
        assert ok is False


# ---------------------------------------------------------------------------
# process_action — REJECT
# ---------------------------------------------------------------------------

class TestReject:
    def test_reject_pending(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-rej2")
        ok, msg = ledger.process_action("tri-rej2", "reject", "sahil", "Out of scope")
        assert ok is True
        assert "REJECTED" in msg

    def test_reject_sets_rejected_fields(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-rej3")
        ledger.process_action("tri-rej3", "reject", "sahil", "Not needed")
        req = ledger.get_request("tri-rej3")
        assert req["status"] == "REJECTED"
        assert req["rejected_by"] == "sahil"
        assert "rejected_at" in req

    def test_reject_writes_audit(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-rej4")
        ledger.process_action("tri-rej4", "reject", "sahil", "Duplicate")
        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[-1])
        assert entry["action"] == "REJECT"
        assert entry["actor"] == "sahil"
        assert entry["reason"] == "Duplicate"

    def test_reject_amending(self, tmp_path):
        """An AMENDING request can also be rejected."""
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-rej5")
        ledger.process_action("tri-rej5", "amend", "sahil", "Needs work")
        ok, msg = ledger.process_action("tri-rej5", "reject", "sahil", "Actually no")
        assert ok is True
        assert ledger.get_request("tri-rej5")["status"] == "REJECTED"

    def test_cannot_reject_already_approved(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-nr1")
        ledger.process_action("tri-nr1", "approve", "sahil", "OK")
        ok, msg = ledger.process_action("tri-nr1", "reject", "sahil", "Wait no")
        assert ok is False


# ---------------------------------------------------------------------------
# process_action — AMEND
# ---------------------------------------------------------------------------

class TestAmend:
    def test_amend_pending(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-am2")
        ok, msg = ledger.process_action("tri-am2", "amend", "sahil", "Change routing")
        assert ok is True
        assert "AMENDING" in msg

    def test_amend_sets_amended_fields(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-am3")
        ledger.process_action("tri-am3", "amend", "sahil", "Adjust")
        req = ledger.get_request("tri-am3")
        assert req["status"] == "AMENDING"
        assert req["amended_by"] == "sahil"
        assert "amended_at" in req

    def test_amend_writes_audit(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-am4")
        ledger.process_action("tri-am4", "amend", "sahil", "Needs changes")
        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[-1])
        assert entry["action"] == "AMEND"
        assert entry["actor"] == "sahil"
        assert entry["reason"] == "Needs changes"

    def test_amend_amending(self, tmp_path):
        """Can amend a request that's already AMENDING (re-amend)."""
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-am5")
        ledger.process_action("tri-am5", "amend", "sahil", "First edit")
        ok, msg = ledger.process_action("tri-am5", "amend", "sahil", "Second edit")
        assert ok is True

    def test_cannot_amend_approved(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-na1")
        ledger.process_action("tri-na1", "approve", "sahil", "OK")
        ok, msg = ledger.process_action("tri-na1", "amend", "sahil", "Wait")
        assert ok is False

    def test_cannot_amend_rejected(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-na2")
        ledger.process_action("tri-na2", "reject", "sahil", "Nope")
        ok, msg = ledger.process_action("tri-na2", "amend", "sahil", "Wait")
        assert ok is False


# ---------------------------------------------------------------------------
# process_action — invalid actions and missing IDs
# ---------------------------------------------------------------------------

class TestInvalidActions:
    def test_unknown_action(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-ia1")
        ok, msg = ledger.process_action("tri-ia1", "escalate", "sahil", "Up")
        assert ok is False
        assert "Unknown action" in msg

    def test_missing_triage_id(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        ok, msg = ledger.process_action("nonexistent-id", "approve", "sahil", "OK")
        assert ok is False
        assert "not found" in msg


# ---------------------------------------------------------------------------
# Audit log completeness (AC4)
# ---------------------------------------------------------------------------

class TestAuditLogCompleteness:
    def test_audit_has_actor_timestamp_reason(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-aud1")
        ledger.process_action("tri-aud1", "approve", "sahil_discord", "Good idea - proceed")
        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2  # SUBMIT + APPROVE
        approve_entry = json.loads(lines[1])
        assert approve_entry["triage_id"] == "tri-aud1"
        assert approve_entry["action"] == "APPROVE"
        assert approve_entry["actor"] == "sahil_discord"
        assert approve_entry["reason"] == "Good idea - proceed"
        assert "timestamp" in approve_entry
        # Timestamp should be valid ISO format
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(approve_entry["timestamp"])
        assert dt.tzinfo is not None

    def test_full_lifecycle_audit_trail(self, tmp_path):
        """Submit → Amend → Approve produces 3 audit entries."""
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-lc1")
        ledger.process_action("tri-lc1", "amend", "sahil", "Tweak")
        ledger.process_action("tri-lc1", "approve", "sahil", "OK now")
        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        actions = [json.loads(line)["action"] for line in lines]
        assert actions == ["SUBMIT", "AMEND", "APPROVE"]

    def test_reject_lifecycle_audit_trail(self, tmp_path):
        """Submit → Reject produces 2 audit entries."""
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-lc2")
        ledger.process_action("tri-lc2", "reject", "sahil", "Not needed")
        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        actions = [json.loads(line)["action"] for line in lines]
        assert actions == ["SUBMIT", "REJECT"]

    def test_reject_without_reason_still_audited(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-no-reason")
        ledger.process_action("tri-no-reason", "reject", "sahil")
        from governance.approvals import ledger as ledger_mod
        lines = ledger_mod.AUDIT_FILE.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[-1])
        assert entry["action"] == "REJECT"
        assert entry["reason"] is None


# ---------------------------------------------------------------------------
# get_request
# ---------------------------------------------------------------------------

class TestGetRequest:
    def test_returns_none_for_unknown(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        assert ledger.get_request("nonexistent") is None

    def test_returns_entry_after_submit(self, tmp_path):
        ledger = _fresh_ledger(tmp_path)
        _submit_basic(ledger, "tri-get1")
        req = ledger.get_request("tri-get1")
        assert req is not None
        assert req["triage_id"] == "tri-get1"
        assert req["fingerprint"] == generate_fingerprint("analytics dashboard idea")


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_state_survives_reload(self, tmp_path):
        """State written to disk survives ledger reinitialisation."""
        ledger1 = _fresh_ledger(tmp_path)
        _submit_basic(ledger1, "tri-ps1")
        ledger1.process_action("tri-ps1", "approve", "sahil", "OK")

        # Reload from same disk
        ledger2 = _fresh_ledger(tmp_path)
        # Force reload by reading from the state file
        from governance.approvals import ledger as ledger_mod
        state_file = ledger_mod.STATE_FILE
        ledger2.state = json.loads(state_file.read_text(encoding="utf-8"))
        req = ledger2.get_request("tri-ps1")
        assert req is not None
        assert req["status"] == "APPROVED"