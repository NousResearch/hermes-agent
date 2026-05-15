"""Tests for SessionEntry.active_agent and SessionStore.set/get_active_agent.

Phase 3 of the multi-agent gateway refactor: each gateway session
remembers which agent (Hermes profile) it is currently bound to.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user1",
        user_name="alice",
    )


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    cfg = GatewayConfig()
    return SessionStore(sessions_dir=tmp_path, config=cfg)


class TestSessionEntryActiveAgent:
    def test_default_value(self):
        now = datetime.now()
        e = SessionEntry(
            session_key="k", session_id="s1", created_at=now, updated_at=now
        )
        assert e.active_agent == "default"

    def test_to_dict_includes_active_agent(self):
        now = datetime.now()
        e = SessionEntry(
            session_key="k",
            session_id="s1",
            created_at=now,
            updated_at=now,
            active_agent="coder",
        )
        d = e.to_dict()
        assert d["active_agent"] == "coder"

    def test_from_dict_legacy_session_uses_default(self):
        now = datetime.now()
        # Simulate a session persisted before the field existed.
        d = {
            "session_key": "k",
            "session_id": "s1",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        e = SessionEntry.from_dict(d)
        assert e.active_agent == "default"

    def test_from_dict_round_trip(self):
        now = datetime.now()
        original = SessionEntry(
            session_key="k",
            session_id="s1",
            created_at=now,
            updated_at=now,
            active_agent="coder",
        )
        restored = SessionEntry.from_dict(original.to_dict())
        assert restored.active_agent == "coder"

    def test_from_dict_null_falls_back_to_default(self):
        now = datetime.now()
        d = {
            "session_key": "k",
            "session_id": "s1",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "active_agent": None,
        }
        e = SessionEntry.from_dict(d)
        assert e.active_agent == "default"


class TestSessionStoreSetActiveAgent:
    def test_set_unknown_session_returns_false(self, store):
        assert store.set_active_agent("nonexistent", "coder") is False

    def test_set_known_session(self, store):
        source = _make_source()
        entry = store.get_or_create_session(source)
        ok = store.set_active_agent(entry.session_key, "coder")
        assert ok is True
        # Round-trip via the public getter
        assert store.get_active_agent(entry.session_key) == "coder"

    def test_default_for_new_session(self, store):
        source = _make_source()
        entry = store.get_or_create_session(source)
        assert store.get_active_agent(entry.session_key) == "default"

    def test_get_unknown_session_returns_default(self, store):
        assert store.get_active_agent("nope") == "default"

    def test_empty_and_whitespace_rejected(self, store):
        source = _make_source()
        entry = store.get_or_create_session(source)
        assert store.set_active_agent(entry.session_key, "") is False
        assert store.set_active_agent(entry.session_key, "   ") is False
        # Original value preserved
        assert store.get_active_agent(entry.session_key) == "default"

    def test_value_is_stripped(self, store):
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.set_active_agent(entry.session_key, "  coder  ")
        assert store.get_active_agent(entry.session_key) == "coder"


class TestPersistence:
    def test_set_is_persisted_to_disk(self, tmp_path):
        cfg = GatewayConfig()
        # Round 1: set the value
        store1 = SessionStore(tmp_path, cfg)
        source = _make_source()
        entry = store1.get_or_create_session(source)
        store1.set_active_agent(entry.session_key, "coder")

        # Round 2: fresh store reads from disk
        store2 = SessionStore(tmp_path, cfg)
        assert store2.get_active_agent(entry.session_key) == "coder"

    def test_persisted_json_contains_active_agent(self, tmp_path):
        cfg = GatewayConfig()
        store = SessionStore(tmp_path, cfg)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.set_active_agent(entry.session_key, "coder")

        sessions_file = tmp_path / "sessions.json"
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
        assert data[entry.session_key]["active_agent"] == "coder"

    def test_legacy_sessions_json_reads_as_default(self, tmp_path):
        """A sessions.json written before this field existed must still load."""
        cfg = GatewayConfig()
        now = datetime.now()
        legacy = {
            "agent:main:telegram:dm:12345": {
                "session_key": "agent:main:telegram:dm:12345",
                "session_id": "old",
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
        }
        (tmp_path / "sessions.json").write_text(
            json.dumps(legacy), encoding="utf-8"
        )
        store = SessionStore(tmp_path, cfg)
        assert (
            store.get_active_agent("agent:main:telegram:dm:12345") == "default"
        )
