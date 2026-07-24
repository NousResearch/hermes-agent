"""Tests for gateway/mirror.py — session mirroring."""

import json
from unittest.mock import patch, MagicMock

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
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            result = mirror_to_session("telegram", "12345", "Hello!", source_label="cli")

        assert result is True

        # Check SQLite writer was called with the mirror message
        mock_sqlite.assert_called_once()
        call_args = mock_sqlite.call_args
        assert call_args[0][0] == "sess_abc"
        msg = call_args[0][1]
        assert msg["content"] == "Hello!"
        assert msg["role"] == "assistant"
        assert msg["mirror"] is True
        assert msg["mirror_source"] == "cli"

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
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            result = mirror_to_session("telegram", "-1001", "Hello topic!", source_label="cron", thread_id="10")

        assert result is True
        mock_sqlite.assert_called_once()
        assert mock_sqlite.call_args[0][0] == "sess_topic_a"

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

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = mirror_to_session("telegram", "99999", "Hello!")

        assert result is False

    def test_no_seed_by_default(self, tmp_path):
        """Without seed_if_missing, a missing session is a no-op (cron path)."""
        sessions_dir, index_file = _setup_sessions(tmp_path, {})

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._seed_session_id") as mock_seed:
            result = mirror_to_session("discord", "99999", "Hello!", thread_id="t1")

        assert result is False
        mock_seed.assert_not_called()

    def test_seed_if_missing_creates_and_mirrors(self, tmp_path):
        """seed_if_missing seeds a session then writes the mirror into it."""
        sessions_dir, index_file = _setup_sessions(tmp_path, {})

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._seed_session_id", return_value="sess_seeded") as mock_seed, \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            result = mirror_to_session(
                "discord",
                "99999",
                "Hello!",
                thread_id="t1",
                user_id="u1",
                seed_if_missing=True,
            )

        assert result is True
        mock_seed.assert_called_once_with("discord", "99999", thread_id="t1", user_id="u1")
        mock_sqlite.assert_called_once()
        assert mock_sqlite.call_args[0][0] == "sess_seeded"

    def test_seed_failure_returns_false(self, tmp_path):
        """If seeding can't produce a session, fall back to the no-op."""
        sessions_dir, index_file = _setup_sessions(tmp_path, {})

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._seed_session_id", return_value=None), \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            result = mirror_to_session(
                "discord", "99999", "Hello!", thread_id="t1", seed_if_missing=True,
            )

        assert result is False
        mock_sqlite.assert_not_called()

    def test_seed_session_id_uses_session_store(self, tmp_path):
        """_seed_session_id reuses SessionStore/SessionSource for a thread send,
        keyed the way the Discord adapter keys an inbound thread session
        (chat_id == thread id) so a later @mention resolves to it (#53414)."""
        sessions_dir, _ = _setup_sessions(tmp_path, {})
        fake_entry = MagicMock(session_id="sess_brand_new")
        fake_store = MagicMock()
        fake_store.get_or_create_session.return_value = fake_entry

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch("gateway.session.SessionStore", return_value=fake_store) as mock_store_cls, \
             patch("gateway.config.load_gateway_config", return_value=MagicMock()):
            result = mirror_mod._seed_session_id(
                "discord", "999parent", thread_id="555thread", user_id="u1",
            )

        assert result == "sess_brand_new"
        fake_store.get_or_create_session.assert_called_once()
        source = fake_store.get_or_create_session.call_args[0][0]
        # Must match the inbound adapter shape: thread id as chat_id, parent kept.
        assert source.chat_id == "555thread"
        assert source.thread_id == "555thread"
        assert source.parent_chat_id == "999parent"
        assert source.chat_type == "thread"

    def test_seeded_thread_key_matches_inbound_adapter_key(self, tmp_path):
        """A send to discord:<parent>:<thread> must seed a session whose key
        equals the one the Discord adapter builds for an inbound thread
        message, so a later @mention in that thread reuses it (#53414)."""
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key

        sessions_dir, _ = _setup_sessions(tmp_path, {})
        fake_store = MagicMock()
        fake_store.get_or_create_session.return_value = MagicMock(session_id="s")

        # Capture the SessionSource _seed_session_id actually builds for the
        # send target "discord:999parent:555thread".
        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch("gateway.session.SessionStore", return_value=fake_store), \
             patch("gateway.config.load_gateway_config", return_value=MagicMock()):
            mirror_mod._seed_session_id(
                "discord", "999parent", thread_id="555thread", user_id="bot",
            )
        seed_source = fake_store.get_or_create_session.call_args[0][0]

        # SessionSource as plugins/platforms/discord/adapter.py builds for an
        # inbound thread message (chat_id == thread id), possibly a different
        # participant — thread sessions are shared, so the user must not matter.
        inbound_source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="555thread",
            chat_type="thread",
            thread_id="555thread",
            parent_chat_id="999parent",
            user_id="some_human",
        )

        assert build_session_key(seed_source) == build_session_key(inbound_source)
        with patch("gateway.mirror._find_session_id", side_effect=Exception("boom")):
            result = mirror_to_session("telegram", "123", "msg")

        assert result is False


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
