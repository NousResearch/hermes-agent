"""Unit tests for content_gate (G03 approval-gated enforcement).

Uses a temp SQLite DB via monkeypatched config.DB_PATH so the live DB is
never touched. Tests cover:
- gate blocks publishing when draft is not approved
- full flow: register -> approve -> gate_publish returns True
- rejected draft is not published
- dry-run mode doesn't mutate DB or send cards
- pending item registration / approval / rejection
"""
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add content_engine dir to sys.path so imports work when run from repo root.
_CE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CE_DIR))

# Point content_gate at a temp DB before importing it.
import config as ce_config


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Isolate the gate to a temp SQLite file."""
    db_file = tmp_path / "content_engine.db"
    monkeypatch.setattr(ce_config, "DB_PATH", db_file)
    # Ensure the content_gate module uses the patched DB_PATH.
    # content_gate imports DB_PATH at module level, so we need to re-import
    # or patch the module attribute directly.
    import content_gate
    monkeypatch.setattr(content_gate, "DB_PATH", db_file)
    # Also patch database module used by content_gate for draft-side checks
    import database
    monkeypatch.setattr(database, "DB_PATH", db_file)
    return db_file


@pytest.fixture(autouse=True)
def _no_dry_run(monkeypatch):
    """Ensure dry-run is off unless the test explicitly enables it."""
    monkeypatch.delenv("CONTENT_GATE_DRY_RUN", raising=False)


# ── Draft-side gate tests ─────────────────────────────────────────────────

def test_gate_publish_blocks_unapproved_draft(tmp_db):
    """A draft with status='draft' must not pass the gate."""
    import database
    database.init_db()
    database.insert_draft(
        draft_id="d1", brand="sahil_twitter", platform="twitter",
        pillar="build", topic="test", title="t", body_text="hello",
    )
    import content_gate
    assert content_gate.gate_publish("d1") is False


def test_gate_publish_allows_approved_draft(tmp_db):
    """An approved draft passes the gate."""
    import database
    database.init_db()
    database.insert_draft(
        draft_id="d2", brand="sahil_twitter", platform="twitter",
        pillar="build", topic="test", title="t", body_text="hello",
    )
    database.approve_draft("d2")
    import content_gate
    assert content_gate.gate_publish("d2") is True


def test_gate_publish_blocks_rejected_draft(tmp_db):
    """A rejected draft must not pass the gate."""
    import database
    database.init_db()
    database.insert_draft(
        draft_id="d3", brand="sahil_twitter", platform="twitter",
        pillar="build", topic="test", title="t", body_text="hello",
    )
    database.reject_draft("d3")
    import content_gate
    assert content_gate.gate_publish("d3") is False


def test_gate_publish_blocks_missing_draft(tmp_db):
    """Unknown draft id -> gate blocks."""
    import content_gate
    assert content_gate.gate_publish("nonexistent") is False


# ── Pending-item gate tests ───────────────────────────────────────────────

def test_register_for_approval_stages_pending_item(tmp_db):
    """register_for_approval inserts a row with approved=0 and returns id."""
    import content_gate
    pid = content_gate.register_for_approval(
        body_text="test suggestion",
        brand="sahil_twitter",
        platform="twitter",
        source="x_scout",
        send_card=False,
    )
    assert pid.startswith("gate_x_sc")

    pending = content_gate.get_pending()
    assert len(pending) == 1
    assert pending[0]["id"] == pid
    assert pending[0]["approved"] == 0
    assert pending[0]["source"] == "x_scout"


def test_process_approval_approve(tmp_db):
    """process_approval('approve') flips approved to 1."""
    import content_gate
    pid = content_gate.register_for_approval(
        body_text="x", brand="b", platform="p", source="test", send_card=False,
    )
    ok = content_gate.process_approval(pid, "approve", decided_by="sahil")
    assert ok is True
    assert content_gate.is_pending_approved(pid) is True
    # No longer in pending list
    assert content_gate.get_pending() == []


def test_process_approval_reject(tmp_db):
    """process_approval('reject') flips approved to -1."""
    import content_gate
    pid = content_gate.register_for_approval(
        body_text="x", brand="b", platform="p", source="test", send_card=False,
    )
    ok = content_gate.process_approval(pid, "reject", decided_by="sahil")
    assert ok is True
    assert content_gate.is_pending_approved(pid) is False
    assert content_gate.get_pending() == []


def test_process_approval_amend_leaves_pending(tmp_db):
    """process_approval('amend') records decision but keeps approved=0."""
    import content_gate
    pid = content_gate.register_for_approval(
        body_text="x", brand="b", platform="p", source="test", send_card=False,
    )
    ok = content_gate.process_approval(pid, "amend", note="needs rewrite")
    assert ok is True
    assert content_gate.is_pending_approved(pid) is False
    assert len(content_gate.get_pending()) == 1


def test_process_approval_unknown_id_returns_false(tmp_db):
    """Deciding on a non-existent id returns False."""
    import content_gate
    assert content_gate.process_approval("gate_none_xxx", "approve") is False


def test_process_approval_invalid_action_raises(tmp_db):
    """Unknown action raises ValueError."""
    import content_gate
    with pytest.raises(ValueError):
        content_gate.process_approval("gate_x_y", "explode")


def test_get_pending_filtered_by_source(tmp_db):
    """get_pending(source=...) returns only matching source items."""
    import content_gate
    content_gate.register_for_approval(
        body_text="a", brand="b1", platform="p", source="x_scout", send_card=False,
    )
    content_gate.register_for_approval(
        body_text="b", brand="b2", platform="p", source="repurpose", send_card=False,
    )
    xs = content_gate.get_pending(source="x_scout")
    assert len(xs) == 1
    assert xs[0]["source"] == "x_scout"


# ── Dry-run tests ─────────────────────────────────────────────────────────

def test_dry_run_register_does_not_mutate_db(tmp_db, monkeypatch):
    """Dry-run register returns synthetic id and writes nothing."""
    monkeypatch.setenv("CONTENT_GATE_DRY_RUN", "1")
    import content_gate
    pid = content_gate.register_for_approval(
        body_text="x", brand="b", platform="p", source="test", send_card=True,
    )
    assert pid.startswith("gate_")
    # DB file should not exist (init_gate_db was skipped)
    assert not tmp_db.exists()


def test_dry_run_gate_publish_always_false(tmp_db, monkeypatch):
    """Dry-run gate_publish always blocks, even for approved drafts."""
    import database
    database.init_db()
    database.insert_draft(
        draft_id="d4", brand="sahil_twitter", platform="twitter",
        pillar="build", topic="test", title="t", body_text="hello",
    )
    database.approve_draft("d4")
    monkeypatch.setenv("CONTENT_GATE_DRY_RUN", "1")
    import content_gate
    assert content_gate.gate_publish("d4") is False


def test_dry_run_process_approval_does_not_mutate(tmp_db, monkeypatch):
    """Dry-run process_approval logs intent but writes nothing."""
    import content_gate
    pid = content_gate.register_for_approval(
        body_text="x", brand="b", platform="p", source="test", send_card=False,
    )
    monkeypatch.setenv("CONTENT_GATE_DRY_RUN", "1")
    ok = content_gate.process_approval(pid, "approve")
    assert ok is True
    # Row still pending
    assert content_gate.is_pending_approved(pid) is False
    assert len(content_gate.get_pending()) == 1


# ── Discord card test (mocked _post) ──────────────────────────────────────

def test_register_for_approval_sends_card(tmp_db):
    """register_for_approval with send_card=True calls discord_digest._post."""
    import content_gate
    with patch("discord_digest._post", return_value=True) as mock_post:
        pid = content_gate.register_for_approval(
            body_text="hello world",
            brand="sahil_twitter",
            platform="twitter",
            source="x_scout",
            title="Test Title",
            send_card=True,
        )
        assert mock_post.called
        call_args = mock_post.call_args
        content = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("content", "")
        assert pid in content
        assert "hello world" in content
        assert "!approve" in content
        assert "!reject" in content
