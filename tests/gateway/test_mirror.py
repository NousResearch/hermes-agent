"""Tests for gateway/mirror.py — session mirroring."""

import json
from unittest.mock import patch, MagicMock

import gateway.mirror as mirror_mod
from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
)


def _setup_sessions(tmp_path, sessions_data):
    """Helper to write a fake sessions.json."""
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

        with patch.object(
            mirror_mod, "_gateway_sessions_dir", return_value=sessions_dir
        ):
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

        with patch.object(
            mirror_mod, "_gateway_sessions_dir", return_value=sessions_dir
        ):
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

        with patch.object(
            mirror_mod, "_gateway_sessions_dir", return_value=sessions_dir
        ):
            result = _find_session_id("telegram", "-1001", thread_id="10")

        assert result == "sess_topic_a"

    def test_legacy_index_ignores_secondary_profile_scope(self, tmp_path, monkeypatch):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        gateway_home = tmp_path / "gateway"
        secondary_home = gateway_home / "profiles" / "secondary"
        sessions_dir, _ = _setup_sessions(gateway_home, {
            "secondary-chat": {
                "session_id": "sess_secondary",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })
        secondary_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(gateway_home))
        empty_db = MagicMock()
        empty_db.find_session_by_origin.return_value = None

        token = set_hermes_home_override(secondary_home)
        try:
            with patch(
                "gateway.mirror._gateway_session_db", return_value=empty_db
            ):
                result = _find_session_id("telegram", "12345")
        finally:
            reset_hermes_home_override(token)

        assert sessions_dir == mirror_mod._gateway_sessions_dir()
        assert result == "sess_secondary"
        empty_db.close.assert_called_once()


class TestMirrorToSession:

    def test_mirrors_to_gateway_store_inside_secondary_profile_scope(
        self, tmp_path, monkeypatch
    ):
        """A routed profile scope must not redirect the gateway transcript DB."""
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from hermes_state import SessionDB

        gateway_home = tmp_path / "gateway"
        secondary_home = gateway_home / "profiles" / "secondary"
        gateway_home.mkdir()
        secondary_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(gateway_home))

        gateway_db = SessionDB(db_path=gateway_home / "state.db")
        gateway_db.create_session(
            "gw-secondary-chat",
            "telegram",
            user_id="user-1",
            session_key="agent:secondary:telegram:dm",
            chat_id="chat-1",
            chat_type="dm",
        )
        gateway_db.close()

        token = set_hermes_home_override(secondary_home)
        try:
            mirrored = mirror_to_session(
                "telegram",
                "chat-1",
                "secondary profile cron brief",
                source_label="cron:test-job",
                user_id="user-1",
                role="user",
            )
        finally:
            reset_hermes_home_override(token)

        assert mirrored is True
        gateway_db = SessionDB(db_path=gateway_home / "state.db")
        try:
            assert [
                (message["role"], message["content"])
                for message in gateway_db.get_messages("gw-secondary-chat")
            ] == [("user", "secondary profile cron brief")]
        finally:
            gateway_db.close()
        assert not (secondary_home / "state.db").exists()


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

        with patch.object(
            mirror_mod, "_gateway_sessions_dir", return_value=sessions_dir
        ), \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            result = mirror_to_session(
                "telegram",
                "-1001",
                "Hello group!",
                source_label="cli",
                user_id="alice",
            )

        assert result is True
        mock_sqlite.assert_called_once()
        assert mock_sqlite.call_args[0][0] == "sess_alice"

    def test_no_matching_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {})

        with patch.object(
            mirror_mod, "_gateway_sessions_dir", return_value=sessions_dir
        ):
            result = mirror_to_session("telegram", "99999", "Hello!")

        assert result is False


class TestAppendToSqlite:
    def test_connection_is_closed_after_use(self, tmp_path):
        """Verify _append_to_sqlite closes the SessionDB connection."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()

        with patch("gateway.mirror._gateway_session_db", return_value=mock_db):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.append_message.assert_called_once()
        mock_db.close.assert_called_once()

