"""Tests for gateway/mirror.py — session mirroring."""

import json
from unittest.mock import patch, MagicMock

import gateway.mirror as mirror_mod
from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
    _append_to_transcript,
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
        sessions_dir, _ = _setup_sessions(tmp_path, {
            "agent:main:telegram:dm": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_abc"

    def test_returns_most_recent(self, tmp_path):
        sessions_dir, _ = _setup_sessions(tmp_path, {
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

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_new"

    def test_thread_id_disambiguates_same_chat(self, tmp_path):
        sessions_dir, _ = _setup_sessions(tmp_path, {
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

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir):
            result = _find_session_id("telegram", "-1001", thread_id="10")

        assert result == "sess_topic_a"

    def test_no_match_returns_none(self, tmp_path):
        sessions_dir, _ = _setup_sessions(tmp_path, {
            "sess": {
                "session_id": "sess_1",
                "origin": {"platform": "discord", "chat_id": "999"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir):
            result = _find_session_id("telegram", "12345")

        assert result is None

    def test_missing_sessions_file(self, tmp_path):
        with patch.object(mirror_mod, "_SESSIONS_DIR", tmp_path / "sessions"):
            result = _find_session_id("telegram", "12345")

        assert result is None

    def test_platform_case_insensitive(self, tmp_path):
        sessions_dir, _ = _setup_sessions(tmp_path, {
            "s1": {
                "session_id": "sess_1",
                "origin": {"platform": "Telegram", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir):
            result = _find_session_id("telegram", "123")

        assert result == "sess_1"


class TestAppendToTranscript:
    def test_appends_message(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        mock_db = MagicMock()

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_transcript("sess_1", {"role": "assistant", "content": "Hello"})

        transcript = sessions_dir / "sess_1.jsonl"
        assert not transcript.exists()
        mock_db.append_message.assert_called_once()

    def test_reuses_supplied_db_without_reopening(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        mock_db = MagicMock()

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch("hermes_state.SessionDB") as session_db_cls:
            _append_to_transcript("sess_1", {"role": "assistant", "content": "Hello"}, db=mock_db)

        session_db_cls.assert_not_called()
        mock_db.append_message.assert_called_once()
        mock_db.close.assert_not_called()

    def test_appends_multiple_messages(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        mock_db = MagicMock()

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_transcript("sess_1", {"role": "assistant", "content": "msg1"})
            _append_to_transcript("sess_1", {"role": "assistant", "content": "msg2"})

        transcript = sessions_dir / "sess_1.jsonl"
        assert not transcript.exists()
        assert mock_db.append_message.call_count == 2


class TestMirrorToSession:
    def test_successful_mirror(self, tmp_path):
        sessions_dir, _ = _setup_sessions(tmp_path, {
            "s1": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch("hermes_state.SessionDB", return_value=MagicMock()):
            result = mirror_to_session("telegram", "12345", "Hello!", source_label="cli")

        assert result is True

        transcript = sessions_dir / "sess_abc.jsonl"
        assert not transcript.exists()

    def test_successful_mirror_uses_thread_id(self, tmp_path):
        sessions_dir, _ = _setup_sessions(tmp_path, {
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
             patch("hermes_state.SessionDB", return_value=MagicMock()):
            result = mirror_to_session("telegram", "-1001", "Hello topic!", source_label="cron", thread_id="10")

        assert result is True
        assert not (sessions_dir / "sess_topic_a.jsonl").exists()
        assert not (sessions_dir / "sess_topic_b.jsonl").exists()

    def test_successful_mirror_reuses_supplied_db(self, tmp_path):
        sessions_dir, _ = _setup_sessions(tmp_path, {
            "s1": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })
        mock_db = MagicMock()

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch("hermes_state.SessionDB") as session_db_cls:
            result = mirror_to_session("telegram", "12345", "Hello!", source_label="cli", db=mock_db)

        assert result is True
        session_db_cls.assert_not_called()
        mock_db.append_message.assert_called_once()
        mock_db.close.assert_not_called()

    def test_no_matching_session(self, tmp_path):
        sessions_dir, _ = _setup_sessions(tmp_path, {})

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir):
            result = mirror_to_session("telegram", "99999", "Hello!")

        assert result is False

    def test_error_returns_false(self, tmp_path):
        with patch("gateway.mirror._find_session_id", side_effect=Exception("boom")):
            result = mirror_to_session("telegram", "123", "msg")

        assert result is False


class TestMirrorDbLifecycle:
    def test_connection_is_closed_after_use(self, tmp_path):
        """Verify _append_to_transcript closes the SessionDB connection."""
        sessions_dir = tmp_path / "sessions"
        mock_db = MagicMock()

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_transcript("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.append_message.assert_called_once()
        mock_db.close.assert_called_once()

    def test_connection_closed_even_on_error(self, tmp_path):
        """Verify connection is closed even when append_message raises."""
        sessions_dir = tmp_path / "sessions"
        mock_db = MagicMock()
        mock_db.append_message.side_effect = Exception("db error")

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_transcript("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.close.assert_called_once()
