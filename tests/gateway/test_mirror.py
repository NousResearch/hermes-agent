"""Tests for gateway/mirror.py — session mirroring."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import gateway.mirror as mirror_mod
from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
    _append_to_jsonl,
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
        assert msg["role"] == "assistant"
        assert msg["mirror"] is True
        assert msg["mirror_source"] == "cli"

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
