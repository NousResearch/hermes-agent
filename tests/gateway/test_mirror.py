"""Tests for gateway/mirror.py — session mirroring."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import gateway.mirror as mirror_mod
from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
    _append_to_jsonl,
    build_mirror_message,
    mirror_to_agent_history_entry,
)


def _setup_sessions(tmp_path, sessions_data):
    """Helper to write a fake sessions.json and patch module-level paths."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_file = sessions_dir / "sessions.json"
    index_file.write_text(json.dumps(sessions_data))
    return sessions_dir, index_file


class TestFindSessionId:
    def test_finds_matching_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "agent:main:telegram:dm": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_abc"

    def test_returns_most_recent(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "old": {
                "session_id": "sess_old",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "new": {
                "session_id": "sess_new",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_new"

    def test_thread_id_disambiguates_same_chat(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "topic_a": {
                "session_id": "sess_topic_a",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "10"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "topic_b": {
                "session_id": "sess_topic_b",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "11"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "-1001", thread_id="10")

        assert result == "sess_topic_a"

    def test_user_id_disambiguates_same_group_chat(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "bob": {
                "session_id": "sess_bob",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "bob"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "-1001", user_id="alice")

        assert result == "sess_alice"

    def test_ambiguous_same_group_chat_without_user_id_returns_none(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "bob": {
                "session_id": "sess_bob",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "bob"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "-1001")

        assert result is None

    def test_no_match_returns_none(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "sess": {
                "session_id": "sess_1",
                "origin": {"platform": "discord", "chat_id": "999"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "12345")

        assert result is None

    def test_missing_sessions_file(self, tmp_path):
        with patch.object(mirror_mod, "_SESSIONS_INDEX", tmp_path / "nope.json"):
            result = _find_session_id("telegram", "12345")

        assert result is None

    def test_platform_case_insensitive(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "s1": {
                "session_id": "sess_1",
                "origin": {"platform": "Telegram", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "123")

        assert result == "sess_1"


class TestAppendToJsonl:
    def test_appends_message(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir):
            _append_to_jsonl("sess_1", {"role": "assistant", "content": "Hello"})

        transcript = sessions_dir / "sess_1.jsonl"
        lines = transcript.read_text().strip().splitlines()
        assert len(lines) == 1
        msg = json.loads(lines[0])
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello"

    def test_appends_multiple_messages(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir):
            _append_to_jsonl("sess_1", {"role": "assistant", "content": "msg1"})
            _append_to_jsonl("sess_1", {"role": "assistant", "content": "msg2"})

        transcript = sessions_dir / "sess_1.jsonl"
        lines = transcript.read_text().strip().splitlines()
        assert len(lines) == 2


class TestMirrorToSession:
    def test_successful_mirror(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "s1": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite"):
            result = mirror_to_session("telegram", "12345", "Hello!", source_label="cli")

        assert result is True

        # Check JSONL was written
        transcript = sessions_dir / "sess_abc.jsonl"
        assert transcript.exists()
        msg = json.loads(transcript.read_text().strip())
        assert msg["content"] == "Hello!"
        assert msg["role"] == "delivery"
        assert msg["event_type"] == "delivery_mirror"
        assert msg["mirror"] is True
        assert msg["mirror_source"] == "cli"
        assert msg["delivery"]["source"]["label"] == "cli"
        assert msg["delivery"]["target"]["platform"] == "telegram"
        assert msg["delivery"]["target"]["chat_id"] == "12345"
        assert msg["delivery"]["target"]["session_id"] == "sess_abc"

    def test_successful_mirror_uses_thread_id(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "topic_a": {
                "session_id": "sess_topic_a",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "10"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "topic_b": {
                "session_id": "sess_topic_b",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "11"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite"):
            result = mirror_to_session("telegram", "-1001", "Hello topic!", source_label="cron", thread_id="10")

        assert result is True
        assert (sessions_dir / "sess_topic_a.jsonl").exists()
        assert not (sessions_dir / "sess_topic_b.jsonl").exists()

    def test_successful_mirror_uses_user_id_for_group_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "bob": {
                "session_id": "sess_bob",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "bob"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite"):
            result = mirror_to_session(
                "telegram",
                "-1001",
                "Hello group!",
                source_label="cli",
                user_id="alice",
            )

        assert result is True
        assert (sessions_dir / "sess_alice.jsonl").exists()
        assert not (sessions_dir / "sess_bob.jsonl").exists()

    def test_no_matching_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {})

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = mirror_to_session("telegram", "99999", "Hello!")

        assert result is False

    def test_error_returns_false(self, tmp_path):
        with patch("gateway.mirror._find_session_id", side_effect=Exception("boom")):
            result = mirror_to_session("telegram", "123", "msg")

        assert result is False


class TestMirrorHistoryEntry:
    def test_delivery_event_becomes_labelled_user_context_not_assistant_reply(self):
        msg = build_mirror_message(
            "telegram",
            "12345",
            "GCB failure summary",
            session_id="sess_target",
            source_label="telegram",
            source_chat_id="-1001",
            source_chat_name="Editors Chat",
            source_user_id="u-alan",
            source_user_name="Alan",
            source_session_key="agent:main:telegram:group:-1001:u-alan",
        )

        entry = mirror_to_agent_history_entry(msg)

        assert entry["role"] == "system"
        assert "External delivery" in entry["content"]
        assert "Alan" in entry["content"]
        assert "Editors Chat" in entry["content"]
        assert "not a reply authored by this session" in entry["content"]
        assert "GCB failure summary" in entry["content"]

    def test_legacy_assistant_mirror_also_becomes_labelled_user_context(self):
        legacy = {
            "role": "assistant",
            "content": "old mirror text",
            "mirror": True,
            "mirror_source": "telegram",
        }

        entry = mirror_to_agent_history_entry(legacy)

        assert entry["role"] == "system"
        assert "External delivery" in entry["content"]
        assert "old mirror text" in entry["content"]


class TestAppendToSqlite:
    def test_connection_is_closed_after_use(self, tmp_path):
        """Verify _append_to_sqlite closes the SessionDB connection."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()

        with patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.append_message.assert_called_once()
        mock_db.close.assert_called_once()

    def test_connection_closed_even_on_error(self, tmp_path):
        """Verify connection is closed even when append_message raises."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()
        mock_db.append_message.side_effect = Exception("db error")

        with patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.close.assert_called_once()
