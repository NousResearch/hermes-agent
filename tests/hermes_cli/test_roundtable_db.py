"""Tests for the Roundtable DB layer (hermes_cli.roundtable_db)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import roundtable_db as rdb


@pytest.fixture
def roundtable_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a fresh roundtable DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_ROUNDTABLE_DB", str(home / "roundtable.db"))
    # Clear the init cache so each test starts fresh
    rdb._INITIALIZED_PATHS.clear()
    return home


@pytest.fixture
def db_conn(roundtable_home):
    """A connected roundtable DB."""
    conn = rdb.connect()
    yield conn
    conn.close()


PARTICIPANTS = [
    {"profile": "alice", "role": "Engineer", "perspective": "Technical", "display_name": "Alice"},
    {"profile": "bob", "role": "Designer", "perspective": "UX", "display_name": "Bob"},
    {"profile": "carol", "role": "PM", "perspective": "Business", "display_name": "Carol"},
]


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------


def test_connect_creates_tables(db_conn):
    rows = db_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert {"discussions", "participants", "speeches", "findings", "convergence_history"} <= names


def test_connect_is_idempotent(db_conn):
    # Insert data
    rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    # Reconnect
    rdb._INITIALIZED_PATHS.clear()
    conn2 = rdb.connect()
    try:
        discs = rdb.list_discussions(conn2)
        assert len(discs) == 1
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# Discussion CRUD
# ---------------------------------------------------------------------------


def test_create_discussion(db_conn):
    disc = rdb.create_discussion(
        db_conn,
        topic="Database selection",
        participants=PARTICIPANTS,
        context="We need a new DB",
        max_rounds=3,
        created_by="coordinator",
    )
    assert disc.id.startswith("rt_")
    assert len(disc.id) == 11  # rt_ + 8 hex
    assert disc.topic == "Database selection"
    assert disc.context == "We need a new DB"
    assert disc.status == "active"
    assert disc.max_rounds == 3
    assert disc.current_round == 0
    assert disc.speech_order == "fixed"


def test_create_discussion_registers_participants(db_conn):
    disc = rdb.create_discussion(
        db_conn, topic="test", participants=PARTICIPANTS
    )
    parts = rdb.get_participants(db_conn, disc.id)
    assert len(parts) == 3
    assert parts[0].participant == "alice"
    assert parts[0].role == "Engineer"
    assert parts[0].display_name == "Alice"
    assert parts[0].is_active is True


def test_create_discussion_validates_speech_order(db_conn):
    with pytest.raises(ValueError, match="Invalid speech_order"):
        rdb.create_discussion(
            db_conn, topic="test", participants=PARTICIPANTS,
            speech_order="invalid"
        )


def test_create_discussion_requires_participants(db_conn):
    with pytest.raises(ValueError, match="At least one participant"):
        rdb.create_discussion(db_conn, topic="test", participants=[])


def test_create_discussion_validates_max_rounds(db_conn):
    with pytest.raises(ValueError, match="max_rounds"):
        rdb.create_discussion(
            db_conn, topic="test", participants=PARTICIPANTS, max_rounds=0
        )


def test_get_discussion(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    fetched = rdb.get_discussion(db_conn, disc.id)
    assert fetched is not None
    assert fetched.id == disc.id
    assert fetched.topic == "test"


def test_get_discussion_not_found(db_conn):
    assert rdb.get_discussion(db_conn, "rt_nonexistent") is None


def test_list_discussions(db_conn):
    for i in range(3):
        rdb.create_discussion(
            db_conn, topic=f"topic {i}", participants=PARTICIPANTS
        )
    discs = rdb.list_discussions(db_conn)
    assert len(discs) == 3


def test_list_discussions_filter_status(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    rdb.create_discussion(db_conn, topic="test2", participants=PARTICIPANTS)
    rdb.conclude_discussion(db_conn, disc.id)

    active = rdb.list_discussions(db_conn, status="active")
    concluded = rdb.list_discussions(db_conn, status="concluded")
    assert len(active) == 1
    assert len(concluded) == 1


def test_conclude_discussion(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    ok = rdb.conclude_discussion(
        db_conn, disc.id, conclusion="We chose PostgreSQL", convergence_score=0.9
    )
    assert ok is True
    fetched = rdb.get_discussion(db_conn, disc.id)
    assert fetched.status == "concluded"
    assert fetched.conclusion == "We chose PostgreSQL"
    assert fetched.convergence_score == 0.9
    assert fetched.concluded_at is not None


def test_conclude_already_concluded(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    rdb.conclude_discussion(db_conn, disc.id)
    ok = rdb.conclude_discussion(db_conn, disc.id)
    assert ok is False


def test_cancel_discussion(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    ok = rdb.cancel_discussion(db_conn, disc.id)
    assert ok is True
    fetched = rdb.get_discussion(db_conn, disc.id)
    assert fetched.status == "cancelled"


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------


def test_get_active_participant_names(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    names = rdb.get_active_participant_names(db_conn, disc.id)
    assert names == ["alice", "bob", "carol"]


# ---------------------------------------------------------------------------
# Speeches
# ---------------------------------------------------------------------------


def test_add_speech_in_round_0(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    speech = rdb.add_speech(db_conn, disc.id, "alice", "Hello everyone!")
    assert speech.id > 0
    assert speech.round == 0
    assert speech.participant == "alice"
    assert speech.content == "Hello everyone!"


def test_speech_round_advances_when_all_spoke(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)

    # Round 0: all 3 speak
    rdb.add_speech(db_conn, disc.id, "alice", "Alice's opening")
    rdb.add_speech(db_conn, disc.id, "bob", "Bob's opening")
    rdb.add_speech(db_conn, disc.id, "carol", "Carol's opening")

    # Check round advanced
    fetched = rdb.get_discussion(db_conn, disc.id)
    assert fetched.current_round == 1

    # Round 1
    rdb.add_speech(db_conn, disc.id, "alice", "Round 1 from Alice")
    fetched = rdb.get_discussion(db_conn, disc.id)
    assert fetched.current_round == 1  # Still round 1 until all spoke


def test_speech_auto_conclude_on_max_rounds(db_conn):
    disc = rdb.create_discussion(
        db_conn, topic="test", participants=PARTICIPANTS, max_rounds=1
    )

    # Round 0
    rdb.add_speech(db_conn, disc.id, "alice", "s1")
    rdb.add_speech(db_conn, disc.id, "bob", "s2")
    rdb.add_speech(db_conn, disc.id, "carol", "s3")

    # After round 0 completes, round advances to 1
    fetched = rdb.get_discussion(db_conn, disc.id)
    assert fetched.current_round == 1

    # Round 1 (max_rounds=1)
    rdb.add_speech(db_conn, disc.id, "alice", "r1s1")
    rdb.add_speech(db_conn, disc.id, "bob", "r1s2")
    rdb.add_speech(db_conn, disc.id, "carol", "r1s3")

    # Should auto-conclude
    fetched = rdb.get_discussion(db_conn, disc.id)
    assert fetched.status == "concluded"


def test_speech_with_reply_to(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    s1 = rdb.add_speech(db_conn, disc.id, "alice", "Original point")
    s2 = rdb.add_speech(db_conn, disc.id, "bob", "Responding", reply_to=s1.id)
    assert s2.reply_to == s1.id


def test_speech_reply_to_invalid(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    with pytest.raises(ValueError, match="reply_to speech"):
        rdb.add_speech(db_conn, disc.id, "alice", "test", reply_to=999)


def test_speech_on_concluded_discussion(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    rdb.conclude_discussion(db_conn, disc.id)
    with pytest.raises(ValueError, match="concluded"):
        rdb.add_speech(db_conn, disc.id, "alice", "Too late")


def test_get_speeches(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    rdb.add_speech(db_conn, disc.id, "alice", "s1")
    rdb.add_speech(db_conn, disc.id, "bob", "s2")
    rdb.add_speech(db_conn, disc.id, "carol", "s3")  # completes round 0
    rdb.add_speech(db_conn, disc.id, "alice", "r1s1")

    all_speeches = rdb.get_speeches(db_conn, disc.id)
    assert len(all_speeches) == 4

    round0 = rdb.get_speeches(db_conn, disc.id, since_round=0)
    assert len(round0) == 4

    round1 = rdb.get_speeches(db_conn, disc.id, since_round=1)
    assert len(round1) == 1

    alice_only = rdb.get_speeches(db_conn, disc.id, participant="alice")
    assert len(alice_only) == 2


def test_get_speech_count(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    rdb.add_speech(db_conn, disc.id, "alice", "s1")
    rdb.add_speech(db_conn, disc.id, "bob", "s2")
    assert rdb.get_speech_count(db_conn, disc.id) == 2


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


def test_add_finding(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    fid = rdb.add_finding(
        db_conn, disc.id, "consensus", "We all agree on X", 1, [1, 2]
    )
    assert fid > 0

    findings = rdb.get_findings(db_conn, disc.id)
    assert len(findings) == 1
    assert findings[0].type == "consensus"
    assert findings[0].content == "We all agree on X"
    assert findings[0].related_speeches == [1, 2]


def test_add_finding_invalid_type(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    with pytest.raises(ValueError, match="Invalid finding type"):
        rdb.add_finding(db_conn, disc.id, "invalid", "test", 1)


def test_get_findings_filter_type(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    rdb.add_finding(db_conn, disc.id, "consensus", "agree", 1)
    rdb.add_finding(db_conn, disc.id, "disagreement", "disagree", 1)
    rdb.add_finding(db_conn, disc.id, "new_point", "new idea", 1)

    consensus = rdb.get_findings(db_conn, disc.id, finding_type="consensus")
    assert len(consensus) == 1
    all_findings = rdb.get_findings(db_conn, disc.id)
    assert len(all_findings) == 3


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def test_record_and_get_convergence(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    rdb.record_convergence(db_conn, disc.id, 1, 0.67, 2, 1, 1)
    rdb.record_convergence(db_conn, disc.id, 2, 0.85, 3, 0, 0)

    history = rdb.get_convergence_history(db_conn, disc.id)
    assert len(history) == 2
    assert history[0].round == 1
    assert history[0].score == 0.67
    assert history[1].round == 2
    assert history[1].score == 0.85


def test_convergence_upsert(db_conn):
    disc = rdb.create_discussion(db_conn, topic="test", participants=PARTICIPANTS)
    rdb.record_convergence(db_conn, disc.id, 1, 0.5, 1, 1, 0)
    rdb.record_convergence(db_conn, disc.id, 1, 0.8, 2, 0, 0)  # same round, should replace

    history = rdb.get_convergence_history(db_conn, disc.id)
    assert len(history) == 1
    assert history[0].score == 0.8
