"""Tests for gateway/mirror.py — session mirroring."""

import importlib
import json
from unittest.mock import patch

import gateway.mirror as mirror_mod
from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
)


def _setup_sessions(tmp_path, sessions_data):
    """Helper to write a fake sessions.json and patch module-level paths."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_file = sessions_dir / "sessions.json"
    index_file.write_text(json.dumps(sessions_data), encoding="utf-8")
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


class TestMirrorToSession:
    def test_unknown_destination_is_queued_without_mutating_transcripts(self, tmp_path, monkeypatch):
        from agent.outbound_context import connection, destination_key
        from hermes_state import SessionDB
        monkeypatch.setenv('HERMES_HOME', str(tmp_path))
        db = SessionDB(db_path=tmp_path / 'state.db')
        try:
            db.create_session('existing', source='telegram')
            assert mirror_to_session('telegram', '-1001', 'Hello group!', user_id='alice')
            with connection() as queue:
                target, recipient, content = queue.execute('SELECT session,recipient,content FROM deliveries').fetchone()
            assert target == destination_key('telegram', '-1001') and recipient == 'alice'
            assert json.loads(content)['message'] == 'Hello group!'
            assert db.get_messages('existing') == []
            assert not (tmp_path / 'sessions/sessions.json').exists()
        finally:
            db.close()

    def test_queue_write_failure_reports_false(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv('HERMES_HOME', str(tmp_path))
        (tmp_path / 'outbound-context.db').mkdir()
        assert not mirror_to_session('telegram', '123', 'Hello!')
        assert 'Mirror failed' in caplog.text


class TestSessionsIndexProfileScoping:
    """#112844: the fallback index must follow the active profile, not the launch one."""

    @staticmethod
    def _write_index(home, session_id):
        d = home / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        (d / "sessions.json").write_text(json.dumps({
            "agent:main:telegram:dm": {
                "session_id": session_id,
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        }), encoding="utf-8")

    def test_fallback_follows_active_profile_home(self, tmp_path, monkeypatch):
        """A profile switched in after import must be read, not the launch profile's index.

        The module captures ``sessions.json`` under the home that was live at import. Under the
        multiplexed gateway one process serves every profile, so a lookup for another profile
        must not resolve against the launch profile's index and return its session id.
        """
        launch_home, active_home = tmp_path / "launch", tmp_path / "active"
        self._write_index(launch_home, "sess_launch")
        self._write_index(active_home, "sess_active")

        # Re-import the module with the launch home live: this is the import-time capture.
        monkeypatch.setenv("HERMES_HOME", str(launch_home))
        importlib.reload(mirror_mod)
        try:
            # A request for a different profile is now served by the same process.
            monkeypatch.setenv("HERMES_HOME", str(active_home))
            assert mirror_mod._find_session_id("telegram", "12345") == "sess_active"
        finally:
            monkeypatch.undo()
            importlib.reload(mirror_mod)

    def test_patched_constant_still_wins(self, tmp_path, monkeypatch):
        """Tests that patch ``_SESSIONS_INDEX`` keep overriding the live home."""
        patched = tmp_path / "patched"
        self._write_index(patched, "sess_patched")

        monkeypatch.setattr(mirror_mod, "_SESSIONS_INDEX", patched / "sessions" / "sessions.json")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "elsewhere"))

        assert mirror_mod._find_session_id("telegram", "12345") == "sess_patched"
