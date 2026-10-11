"""Tests for gateway/mirror.py — session mirroring."""

import importlib
import json

import pytest
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


    def test_failed_sqlite_write_reports_false(self, tmp_path):
        """A mirror whose transcript write raises must not report success (#10130)."""
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "dm": {
                "session_id": "sess_dm",
                "origin": {"platform": "telegram", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            },
        })
        broken_db = MagicMock()
        broken_db.find_session_by_origin.return_value = None  # resolve via sessions.json
        broken_db.append_message.side_effect = OSError("disk full")

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("hermes_state_registry.acquire", return_value=broken_db), \
             patch("hermes_state_registry.release_or_close"):
            result = mirror_to_session("telegram", "123", "Hello!")

        assert result is False
        broken_db.append_message.assert_called_once()


class TestAppendToSqlite:
    def test_connection_is_released_after_use(self, tmp_path):
        """Verify _append_to_sqlite returns the shared SessionDB reference."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()
        released = []

        with patch("hermes_state_registry.acquire", return_value=mock_db), \
             patch(
                 "hermes_state_registry.release_or_close",
                 side_effect=lambda db: released.append(db),
             ):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.append_message.assert_called_once()
        assert released == [mock_db], (
            "the shared handle must be released exactly once after use"
        )


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


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("alice_exists", [False, True])
def test_matrix_legacy_mirror_uses_the_configured_participant_key(
    tmp_path, monkeypatch, isolated, alice_exists
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key
    from gateway import mirror

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({
        "thread_sessions_per_user": isolated,
        "multiplex_profiles": False,
    }), encoding="utf-8")
    room = "!reports:remote.test"
    source = SessionSource(
        platform=Platform.MATRIX, chat_id=room, thread_id="$root",
        chat_type="group", user_id="@bob:remote.test",
    )
    bob_key = build_session_key(source, thread_sessions_per_user=isolated)
    entries = {bob_key: {"session_id": "bob-session", "origin": source.to_dict()}}
    source.user_id = "@alice:remote.test"
    if alice_exists:
        alice_key = build_session_key(source, thread_sessions_per_user=isolated)
        entries[alice_key] = {"session_id": "alice-session", "origin": source.to_dict()}
    index = tmp_path / "sessions.json"
    index.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(mirror, "_SESSIONS_INDEX", index)
    db = SimpleNamespace(find_latest_gateway_session_for_peer=Mock(return_value=None))
    monkeypatch.setattr("hermes_state_registry.acquire", lambda: db)
    monkeypatch.setattr("hermes_state_registry.release_or_close", lambda db: None)
    append = Mock()
    monkeypatch.setattr(mirror, "_append_to_sqlite", append)

    mirrored = mirror.mirror_to_session(
        "matrix", room, "Brief", thread_id=source.thread_id,
        user_id=source.user_id, chat_type=source.chat_type,
    )

    expected_id = "alice-session" if alice_exists else "bob-session" if not isolated else None
    assert mirrored == bool(expected_id)
    assert [(args[0], args[1]["content"]) for args, kwargs in append.call_args_list] == (
        [(expected_id, "Brief")] if expected_id else []
    )
    db.find_latest_gateway_session_for_peer.assert_called_once_with(
        source="matrix", session_key=build_session_key(source, thread_sessions_per_user=isolated),
    )
