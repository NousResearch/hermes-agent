"""Tests for gateway/mirror.py — session mirroring."""

import json
from pathlib import Path
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
            "agent:main:telegram:dm:old": {
                "session_id": "sess_old",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "agent:main:telegram:dm:new": {
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
            "agent:main:telegram:topic:10": {
                "session_id": "sess_topic_a",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "10"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "agent:main:telegram:topic:11": {
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

    def test_routes_root_index_entries_to_each_owning_profile_db(self, tmp_path, monkeypatch):
        """A multiplex root index must append A→B→A through each session's owner DB."""
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_state import SessionDB
        import hermes_state_registry

        root = tmp_path / ".hermes"
        work_home = root / "profiles" / "work"
        work_home.mkdir(parents=True)
        sessions_dir = root / "sessions"
        sessions_dir.mkdir(parents=True)
        index_file = sessions_dir / "sessions.json"
        routes = {
            "agent:work:discord:dm:work-chat": {
                "session_key": "agent:work:discord:dm:work-chat",
                "session_id": "sess_work",
                "origin": {"platform": "discord", "chat_id": "work-chat", "profile": "work"},
                "updated_at": "2026-01-02T00:00:00",
            },
            "agent:main:discord:dm:default-chat": {
                "session_key": "agent:main:discord:dm:default-chat",
                "session_id": "sess_default",
                "origin": {"platform": "discord", "chat_id": "default-chat", "profile": "default"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "agent:work:discord:group:shared:alice": {
                "session_key": "agent:work:discord:group:shared:alice",
                "session_id": "sess_work",
                "origin": {"platform": "discord", "chat_id": "shared", "user_id": "alice", "profile": "work"},
                "updated_at": "2026-01-03T00:00:00",
            },
            "agent:main:discord:group:shared:bob": {
                "session_key": "agent:main:discord:group:shared:bob",
                "session_id": "sess_default",
                "origin": {"platform": "discord", "chat_id": "shared", "user_id": "bob", "profile": "default"},
                "updated_at": "2026-01-04T00:00:00",
            },
            "agent:work:discord:group:exact:carol": {
                "session_key": "agent:work:discord:group:exact:carol",
                "session_id": "sess_work",
                "origin": {"platform": "discord", "chat_id": "exact", "user_id": "carol", "profile": "work"},
                "updated_at": "2026-01-05T00:00:00",
            },
            "agent:main:discord:group:exact:carol": {
                "session_key": "agent:main:discord:group:exact:carol",
                "session_id": "sess_default",
                "origin": {"platform": "discord", "chat_id": "exact", "user_id": "carol", "profile": "default"},
                "updated_at": "2026-01-06T00:00:00",
            },
            "agent:work:discord:dm:missing": {
                "session_key": "agent:work:discord:dm:missing",
                "origin": {"platform": "discord", "chat_id": "missing", "profile": "work"},
                "updated_at": "2026-01-07T00:00:00",
            },
            "agent:work:discord:dm:disagree": {
                "session_key": "agent:work:discord:dm:disagree",
                "session_id": "sess_work",
                "origin": {"platform": "discord", "chat_id": "disagree", "profile": "default"},
                "updated_at": "2026-01-08T00:00:00",
            },
            "agent:work:discord:group:single-user": {
                "session_key": "agent:work:discord:group:single-user",
                "session_id": "sess_work",
                "origin": {
                    "platform": "discord", "chat_id": "single-user",
                    "user_id": "alice", "profile": "work",
                },
                "updated_at": "2026-01-08T01:00:00",
            },
        }
        json_route = {
            "agent:work:discord:dm:json-only": {
                "session_key": "agent:work:discord:dm:json-only",
                "session_id": "sess_json",
                "origin": {"platform": "discord", "chat_id": "json-only", "profile": "work"},
                "updated_at": "2026-01-09T00:00:00",
            }
        }
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(root))
        token = set_hermes_home_override(str(root))

        root_db = SessionDB(db_path=root / "state.db")
        work_db = SessionDB(db_path=work_home / "state.db")
        try:
            root_db.create_session(
                "sess_default", "gateway",
                session_key="agent:main:discord:dm:default-chat", profile_name="default")
            root_db.create_session(
                "sess_legacy", "discord", chat_id="legacy-chat",
                session_key="agent:main:discord:dm:legacy-chat", profile_name="default")
            root_db.create_session(
                "sess_ambiguous_legacy", "discord", user_id="mallory", chat_id="shared",
                session_key="agent:main:discord:group:shared:mallory", profile_name="default")
            root_db.create_session(
                "sess_exact_legacy", "discord", user_id="carol", chat_id="exact",
                session_key="agent:main:discord:group:exact:legacy", profile_name="default")
            root_db.create_session(
                "sess_missing_legacy", "discord", chat_id="missing",
                session_key="agent:main:discord:dm:missing-legacy", profile_name="default")
            root_db.create_session(
                "sess_json_legacy", "discord", chat_id="json-only",
                session_key="agent:main:discord:dm:json-only-legacy", profile_name="default")
            work_db.create_session(
                "sess_work", "gateway",
                session_key="agent:work:discord:dm:work-chat", profile_name="work")
            work_db.create_session(
                "sess_json", "gateway",
                session_key="agent:work:discord:dm:json-only", profile_name="work")
            for session_key, entry in routes.items():
                root_db.save_gateway_routing_entry(
                    session_key, json.dumps(entry), scope=str(sessions_dir.resolve()))
            index_file.write_text(json.dumps(json_route), encoding="utf-8")
        finally:
            root_db.close()
            work_db.close()

        try:
            with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
                 patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
                assert mirror_to_session("discord", "work-chat", "work one") is True
                assert mirror_to_session("discord", "default-chat", "default") is True
                assert mirror_to_session("discord", "work-chat", "work two") is True
                assert mirror_to_session(
                    "discord", "shared", "must not reach legacy root", user_id="mallory") is False
                assert mirror_to_session(
                    "discord", "exact", "must not choose newer owner", user_id="carol") is False
                assert mirror_to_session(
                    "discord", "missing", "must not bypass malformed route") is False
                assert mirror_to_session(
                    "discord", "disagree", "must not trust conflicting profile") is False
                assert mirror_to_session(
                    "discord", "single-user", "must not cross users", user_id="bob") is False
                assert mirror_to_session(
                    "discord", "json-only", "json route wins over legacy root") is True
                nested_token = set_hermes_home_override(str(work_home))
                try:
                    assert mirror_to_session("discord", "legacy-chat", "legacy root") is True
                finally:
                    reset_hermes_home_override(nested_token)
                with patch.object(
                    mirror_mod, "_routing_entries_from_db", side_effect=OSError("routing unavailable")
                ):
                    assert mirror_to_session(
                        "discord", "legacy-chat", "must not bypass unavailable routes") is False
                index_file.write_text("{not-json", encoding="utf-8")
                assert mirror_to_session(
                    "discord", "work-chat", "must not bypass malformed routes") is False
        finally:
            hermes_state_registry.close_all()
            reset_hermes_home_override(token)

        root_db = SessionDB(db_path=root / "state.db")
        work_db = SessionDB(db_path=work_home / "state.db")
        try:
            assert [m["content"] for m in root_db.get_messages("sess_default")] == ["default"]
            assert [m["content"] for m in root_db.get_messages("sess_legacy")] == ["legacy root"]
            assert root_db.get_messages("sess_ambiguous_legacy") == []
            assert root_db.get_messages("sess_exact_legacy") == []
            assert root_db.get_messages("sess_missing_legacy") == []
            assert root_db.get_messages("sess_json_legacy") == []
            assert [m["content"] for m in work_db.get_messages("sess_work")] == ["work one", "work two"]
            assert [m["content"] for m in work_db.get_messages("sess_json")] == [
                "json route wins over legacy root"
            ]
        finally:
            root_db.close()
            work_db.close()


    def test_successful_mirror_uses_user_id_for_group_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "agent:main:telegram:dm:alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "agent:main:telegram:dm:bob": {
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
            "agent:main:telegram:dm": {
                "session_id": "sess_dm",
                "origin": {"platform": "telegram", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            },
        })
        broken_db = MagicMock()
        broken_db.load_gateway_routing_entries.return_value = {}
        broken_db.find_session_by_origin.return_value = None  # resolve via sessions.json
        broken_db.append_message.side_effect = OSError("disk full")

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("hermes_state_registry.acquire", return_value=broken_db), \
             patch("hermes_state_registry.release_or_close"):
            result = mirror_to_session("telegram", "123", "Hello!")

        assert result is False
        broken_db.append_message.assert_called_once()

    def test_explicit_non_default_session_id_uses_route_owner(self, tmp_path, monkeypatch):
        """Cron's explicit session-id fast path must preserve profile ownership."""
        from hermes_state import SessionDB

        home = tmp_path / ".hermes"
        sessions_dir = home / "sessions"
        sessions_dir.mkdir(parents=True)
        profile_home = home / "profiles" / "stratta"
        profile_home.mkdir(parents=True)
        session_key = "agent:stratta:discord:dm:channel-1"
        entry = {
            "session_key": session_key,
            "session_id": "sess_stratta",
            "origin": {"platform": "discord", "chat_id": "channel-1", "profile": "stratta"},
            "updated_at": "2026-09-16T00:00:00",
        }

        owner_db = SessionDB(profile_home / "state.db")
        owner_db.create_session(
            "sess_stratta", "discord", session_key=session_key, profile_name="stratta",
            chat_id="channel-1", chat_type="dm",
        )
        owner_db.close()
        routing_db = SessionDB(home / "state.db")
        routing_db.create_session(
            "sess_bad_outer", "discord", session_key="agent:main:discord:dm:bad-outer",
            profile_name="default", chat_id="bad-outer", chat_type="dm",
        )
        routing_db.save_gateway_routing_entry(
            session_key, json.dumps(entry), scope=str(sessions_dir.resolve()))
        routing_db.close()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(home))
        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", sessions_dir / "sessions.json"):
            result = mirror_to_session(
                "discord", "channel-1", "Cron brief", role="user", session_id="sess_stratta")
            unresolved = mirror_to_session(
                "discord", "channel-1", "must not write", role="user", session_id="missing")

            conflicting_key = "agent:main:discord:dm:other-channel"
            conflicting_entry = {
                "session_key": conflicting_key,
                "session_id": "sess_stratta",
                "origin": {"platform": "discord", "chat_id": "other-channel", "profile": "default"},
                "updated_at": "2026-09-16T00:01:00",
            }
            routing_db = SessionDB(home / "state.db")
            routing_db.save_gateway_routing_entry(
                conflicting_key, json.dumps(conflicting_entry), scope=str(sessions_dir.resolve()))
            routing_db.close()
            conflicting = mirror_to_session(
                "discord", "channel-1", "must not cross profiles",
                role="user", session_id="sess_stratta")

            malformed_key = "agent:stratta:discord:dm:malformed"
            malformed_entry = {
                "session_key": "agent:main:discord:dm:malformed",
                "session_id": "sess_malformed",
                "origin": {"platform": "discord", "chat_id": "malformed", "profile": "stratta"},
            }
            routing_db = SessionDB(home / "state.db")
            routing_db.save_gateway_routing_entry(
                malformed_key, json.dumps(malformed_entry), scope=str(sessions_dir.resolve()))
            routing_db.close()
            malformed = mirror_to_session(
                "discord", "malformed", "must not redirect",
                role="user", session_id="sess_malformed")

            malformed_embedded_key = "agent:stratta:discord:dm:bad-embedded"
            malformed_embedded_entry = {
                "session_key": "not-an-agent-key",
                "session_id": "sess_bad_embedded",
                "origin": {
                    "platform": "discord", "chat_id": "bad-embedded", "profile": "stratta",
                },
            }
            routing_db = SessionDB(home / "state.db")
            routing_db.save_gateway_routing_entry(
                malformed_embedded_key, json.dumps(malformed_embedded_entry),
                scope=str(sessions_dir.resolve()))
            routing_db.close()
            malformed_embedded = mirror_to_session(
                "discord", "bad-embedded", "must reject malformed embedded key",
                role="user", session_id="sess_bad_embedded")

            empty_namespace_entry = {
                "session_key": "agent:",
                "session_id": "sess_empty_namespace",
                "origin": {"platform": "discord", "chat_id": "empty-namespace"},
            }
            routing_db = SessionDB(home / "state.db")
            routing_db.save_gateway_routing_entry(
                "agent:", json.dumps(empty_namespace_entry), scope=str(sessions_dir.resolve()))
            routing_db.close()
            empty_namespace = mirror_to_session(
                "discord", "empty-namespace", "must reject empty namespace",
                role="user", session_id="sess_empty_namespace")

            bad_outer_entry = {
                "session_key": "agent:main",
                "session_id": "sess_bad_outer",
                "origin": {"platform": "discord", "chat_id": "bad-outer", "profile": "default"},
            }
            routing_db = SessionDB(home / "state.db")
            routing_db.save_gateway_routing_entry(
                "agent:main", json.dumps(bad_outer_entry), scope=str(sessions_dir.resolve()))
            routing_db.close()
            bad_outer = mirror_to_session(
                "discord", "bad-outer", "must reject truncated outer key",
                role="user", session_id="sess_bad_outer")

            conflicting_json_entry = {
                "session_key": "agent:main:discord:dm:channel-1",
                "session_id": "sess_cross_source",
                "origin": {
                    "platform": "discord", "chat_id": "channel-1", "profile": "default",
                },
            }
            cross_source_db_entry = {
                "session_key": "agent:stratta:discord:dm:channel-1-cross-source",
                "session_id": "sess_cross_source",
                "origin": {
                    "platform": "discord", "chat_id": "channel-1", "profile": "stratta",
                },
            }
            routing_db = SessionDB(home / "state.db")
            routing_db.save_gateway_routing_entry(
                "agent:stratta:discord:dm:channel-1-cross-source",
                json.dumps(cross_source_db_entry), scope=str(sessions_dir.resolve()))
            routing_db.close()
            (sessions_dir / "sessions.json").write_text(
                json.dumps({"agent:main:discord:dm:channel-1": conflicting_json_entry}),
                encoding="utf-8",
            )
            cross_source_conflict = mirror_to_session(
                "discord", "channel-1", "must reject cross-source conflict",
                role="user", session_id="sess_cross_source")
            (sessions_dir / "sessions.json").unlink()

            with patch.object(
                mirror_mod, "_routing_entries_from_db", side_effect=OSError("routing unavailable")
            ):
                db_unavailable = mirror_to_session(
                    "discord", "channel-1", "must reject unavailable DB routes",
                    role="user", session_id="sess_stratta")
            with patch.object(
                mirror_mod, "_routing_entries_from_json", side_effect=OSError("routing unavailable")
            ):
                json_unavailable = mirror_to_session(
                    "discord", "channel-1", "must reject unavailable JSON routes",
                    role="user", session_id="sess_stratta")

            deleted_key = "agent:deleted:discord:dm:gone"
            deleted_entry = {
                "session_key": deleted_key,
                "session_id": "sess_deleted",
                "origin": {"platform": "discord", "chat_id": "gone", "profile": "deleted"},
            }
            routing_db = SessionDB(home / "state.db")
            routing_db.save_gateway_routing_entry(
                deleted_key, json.dumps(deleted_entry), scope=str(sessions_dir.resolve()))
            routing_db.close()
            deleted = mirror_to_session(
                "discord", "gone", "must not recreate a profile",
                role="user", session_id="sess_deleted")

        assert result is True
        assert unresolved is False
        assert conflicting is False
        assert malformed is False
        assert malformed_embedded is False
        assert empty_namespace is False
        assert bad_outer is False
        assert cross_source_conflict is False
        assert db_unavailable is False
        assert json_unavailable is False
        assert deleted is False
        owner_db = SessionDB(profile_home / "state.db")
        root_db = SessionDB(home / "state.db")
        try:
            assert [message["content"] for message in owner_db.get_messages("sess_stratta")] == [
                "Cron brief"
            ]
            assert root_db.get_messages("sess_bad_outer") == []
        finally:
            owner_db.close()
            root_db.close()


class TestAppendToSqlite:
    def test_connection_is_released_after_use(self, tmp_path):
        """Verify _append_to_sqlite returns the shared SessionDB reference."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()
        released = []

        db_path = tmp_path / "state.db"
        with patch("hermes_state_registry.acquire", return_value=mock_db) as acquire, \
             patch(
                 "hermes_state_registry.release_or_close",
                 side_effect=lambda db: released.append(db),
             ):
            _append_to_sqlite(
                "sess_1", {"role": "assistant", "content": "hello"}, db_path)

        acquire.assert_called_once_with(db_path)
        mock_db.append_message.assert_called_once()
        assert released == [mock_db], (
            "the shared handle must be released exactly once after use"
        )

