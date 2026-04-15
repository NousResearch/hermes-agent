"""Tests for explicit-timestamp kwargs on SessionDB.create_session and append_message."""
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB


def test_append_message_uses_explicit_timestamp(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "20260415_083111_deadbeef"
    db.create_session(sid, source="test", model="test-model")
    fixed_ts = 1_700_000_000.0
    msg_id = db.append_message(sid, role="user", content="hi", timestamp=fixed_ts)
    rows = db.get_messages(sid)
    assert len(rows) == 1
    assert rows[0]["timestamp"] == pytest.approx(fixed_ts, abs=1e-6)


def test_append_message_defaults_to_now_when_timestamp_omitted(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "20260415_083111_cafebabe"
    db.create_session(sid, source="test")
    before = time.time()
    db.append_message(sid, role="user", content="hi")
    after = time.time()
    ts = db.get_messages(sid)[0]["timestamp"]
    assert before <= ts <= after


def test_create_session_uses_explicit_started_at(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "20260415_083111_01234567"
    fixed_ts = 1_700_000_000.0
    db.create_session(sid, source="test", started_at=fixed_ts)
    row = db.get_session(sid)
    assert row["started_at"] == pytest.approx(fixed_ts, abs=1e-6)


def test_create_session_defaults_to_now_when_started_at_omitted(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "20260415_083111_89abcdef"
    before = time.time()
    db.create_session(sid, source="test")
    after = time.time()
    row = db.get_session(sid)
    assert before <= row["started_at"] <= after
