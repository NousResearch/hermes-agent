"""Tests for the previous-session bridge feature."""
from datetime import datetime, timezone

import pytest

from gateway.session import SessionEntry


def test_session_entry_carries_previous_session_id():
    entry = SessionEntry(
        session_key="signal:dm:user-abc",
        session_id="20260531_120000_aaaa",
        created_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        previous_session_id="20260530_150000_bbbb",
    )
    assert entry.previous_session_id == "20260530_150000_bbbb"


def test_session_entry_round_trip_preserves_previous_session_id():
    entry = SessionEntry(
        session_key="k",
        session_id="new",
        created_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        previous_session_id="old",
    )
    restored = SessionEntry.from_dict(entry.to_dict())
    assert restored.previous_session_id == "old"


def test_session_entry_default_previous_session_id_is_none():
    entry = SessionEntry(
        session_key="k",
        session_id="s",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    assert entry.previous_session_id is None
