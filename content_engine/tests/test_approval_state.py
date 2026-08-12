"""Tests for content_engine/approval_state.py (approval ledger + transition guard).

Covers the guarantees ported from the scratch approval state machine:
- only explicit transitions change state; illegal ones raise
- decisions are append-only and carry actor/comment/version/timestamp
- actor allowlist is enforced
- terminal states refuse further transitions
- version increments per amend/resubmit cycle
"""

import sqlite3

import pytest

from approval_state import ApprovalError, ApprovalLedger


@pytest.fixture()
def ledger(tmp_path):
    db = tmp_path / "content_engine.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE drafts (
            id TEXT PRIMARY KEY, brand TEXT, platform TEXT,
            status TEXT NOT NULL DEFAULT 'draft',
            approved_at TEXT, rejected_at TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO drafts (id, brand, platform, status) VALUES (?, ?, ?, 'draft')",
        ("draft-1", "sahil_twitter", "twitter"),
    )
    conn.commit()
    conn.close()
    return ApprovalLedger(db_path=str(db))


def _status(db_path: str, draft_id: str) -> str:
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT status FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    conn.close()
    return row[0]


def test_approve_transitions_and_is_audited(ledger):
    assert ledger.decide("draft-1", "approve", actor="sahil", comment="lgtm")
    assert _status(ledger.db_path, "draft-1") == "approved"
    history = ledger.history("draft-1")
    assert len(history) == 1
    assert history[0]["action"] == "approve"
    assert history[0]["actor"] == "sahil"
    assert history[0]["comment"] == "lgtm"
    assert history[0]["version"] == 1
    assert history[0]["decided_at"]


def test_reject_transitions_and_is_audited(ledger):
    assert ledger.decide("draft-1", "reject", actor="sahil")
    assert _status(ledger.db_path, "draft-1") == "rejected"
    assert ledger.history("draft-1")[0]["action"] == "reject"


def test_terminal_state_refuses_further_transitions(ledger):
    ledger.decide("draft-1", "approve", actor="sahil")
    with pytest.raises(ApprovalError, match="cannot"):
        ledger.decide("draft-1", "reject", actor="sahil")
    with pytest.raises(ApprovalError, match="cannot"):
        ledger.decide("draft-1", "amend", actor="sahil")


def test_amend_then_resubmit_increments_version(ledger):
    ledger.decide("draft-1", "amend", actor="sahil", comment="tone is off")
    assert _status(ledger.db_path, "draft-1") == "amended"
    assert ledger.decide("draft-1", "resubmit", actor="ceecee", comment="fixed")
    assert _status(ledger.db_path, "draft-1") == "draft"
    assert ledger.current_version("draft-1") == 2
    actions = [d["action"] for d in ledger.history("draft-1")]
    assert actions == ["amend_request", "resubmit"]


def test_actor_allowlist_enforced(tmp_path):
    db = tmp_path / "allow.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE drafts (
            id TEXT PRIMARY KEY, brand TEXT, platform TEXT,
            status TEXT NOT NULL DEFAULT 'draft',
            approved_at TEXT, rejected_at TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO drafts (id, brand, platform, status) VALUES ('d-1', 'b', 'p', 'draft')"
    )
    conn.commit()
    conn.close()
    ledger = ApprovalLedger(db_path=str(db), approver_ids=["sahil"])
    with pytest.raises(ApprovalError, match="not an allowed approver"):
        ledger.decide("d-1", "approve", actor="intruder")
    assert _status(str(db), "d-1") == "draft"
    assert ledger.decide("d-1", "approve", actor="sahil")
    assert _status(str(db), "d-1") == "approved"


def test_unknown_action_raises(ledger):
    with pytest.raises(ApprovalError, match="unknown approval action"):
        ledger.decide("draft-1", "publish-now", actor="sahil")


def test_missing_draft_returns_false(ledger):
    assert ledger.decide("no-such-draft", "approve", actor="sahil") is False


def test_approve_after_amend_request_is_allowed(ledger):
    # The live vocabulary allows approve from 'amended' (reviewer approves the
    # amended draft directly without an explicit resubmit).
    ledger.decide("draft-1", "amend", actor="sahil")
    assert ledger.decide("draft-1", "approve", actor="sahil")
    assert _status(ledger.db_path, "draft-1") == "approved"
