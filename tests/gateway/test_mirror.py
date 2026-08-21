"""Tests for gateway/mirror.py — session mirroring."""

import json
from unittest.mock import patch, MagicMock

import pytest

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


class TestMirrorToSession:


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

    def test_mirror_row_is_tagged_with_relay_provenance(self, tmp_path):
        """A relayed message must be identifiable without trusting its role.

        The mirror keeps role="assistant" for strict-alternation replay, so
        the provenance has to travel on the presentation-only display fields
        or a reader cannot tell relayed text from real model output.
        """
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001"},
                "updated_at": "2026-01-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite") as mock_sqlite:
            assert mirror_to_session(
                "telegram", "-1001", "Relayed text", source_label="cron",
            ) is True

        message = mock_sqlite.call_args[0][1]
        assert message["role"] == "assistant"
        assert message["display_kind"] == mirror_mod.MIRROR_DISPLAY_KIND
        assert message["display_metadata"]["mirror"] is True
        assert message["display_metadata"]["mirror_source"] == "cron"
        assert message["display_metadata"]["platform"] == "telegram"
        assert message["display_metadata"]["chat_id"] == "-1001"


class TestAppendToSqlite:
    def test_connection_is_closed_after_use(self, tmp_path):
        """Verify _append_to_sqlite closes the SessionDB connection."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()

        with patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.append_message.assert_called_once()
        mock_db.close.assert_called_once()

    def test_display_provenance_reaches_append_message(self, tmp_path):
        """The provenance must survive the SQLite boundary, not be dropped."""
        from gateway.mirror import _append_to_sqlite, MIRROR_DISPLAY_KIND
        mock_db = MagicMock()

        with patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_sqlite("sess_1", {
                "role": "assistant",
                "content": "hello",
                "display_kind": MIRROR_DISPLAY_KIND,
                "display_metadata": {"mirror": True, "mirror_source": "cli"},
            })

        kwargs = mock_db.append_message.call_args.kwargs
        assert kwargs["display_kind"] == MIRROR_DISPLAY_KIND
        assert kwargs["display_metadata"] == {"mirror": True, "mirror_source": "cli"}


class TestMirrorProvenanceRoundTrip:
    """The provenance must survive a real SQLite write/read, end to end.

    The previous behaviour dropped the mirror flags at the SessionDB boundary,
    which is exactly why a relayed message became indistinguishable from model
    output once it was read back.
    """

    def test_relayed_message_reads_back_as_relay(self, tmp_path, monkeypatch):
        from hermes_state import SessionDB
        import mcp_serve

        db = SessionDB(db_path=tmp_path / "state.db")
        session_id = "20260820_120000_relay"
        db.create_session(session_id, "slack")
        db.record_gateway_session_peer(
            session_id,
            source="slack",
            session_key="agent:main:slack:dm:C1",
            chat_id="C1",
            chat_type="dm",
            display_name="livio",
            origin_json=json.dumps({"platform": "slack", "chat_id": "C1"}),
        )

        class _SharedDB:
            """Hand mirror the same connection; ignore its close()."""

            def __getattr__(self, name):
                return getattr(db, name)

            def close(self):
                pass

        monkeypatch.setattr("hermes_state.SessionDB", lambda *a, **k: _SharedDB())

        try:
            assert mirror_to_session(
                "slack", "C1", "relayed instruction", source_label="cli",
            ) is True

            stored = [m for m in db.get_messages(session_id)
                      if m.get("role") == "assistant"]
            assert len(stored) == 1
            message = stored[0]
            assert message["display_kind"] == mirror_mod.MIRROR_DISPLAY_KIND
            assert message["display_metadata"]["mirror"] is True

            # What the MCP read tool would report for this row.
            assert mcp_serve._message_origin(message) == ("relay", "cli")
        finally:
            db.close()

