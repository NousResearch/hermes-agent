"""Tests for gateway/channel_directory.py — channel resolution and display."""

import asyncio
import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from gateway.channel_directory import (
    build_channel_directory,
    lookup_channel_type,
    resolve_channel_name,
    format_directory_for_display,
    load_directory,
    _build_from_sessions,
    _build_slack,
)


import pytest


@pytest.fixture(autouse=True)
def _isolate_channel_aliases(tmp_path_factory):
    """Point the alias overlay at a nonexistent path by default so a real
    ~/.hermes/channel_aliases.json never leaks into directory tests. Tests
    that exercise aliases patch CHANNEL_ALIASES_PATH themselves inside the
    test body, which takes precedence over this outer patch."""
    missing = tmp_path_factory.mktemp("aliases") / "none.json"
    with patch("gateway.channel_directory.CHANNEL_ALIASES_PATH", missing):
        yield


def _write_directory(tmp_path, platforms):
    """Helper to write a fake channel directory."""
    data = {"updated_at": "2026-01-01T00:00:00", "platforms": platforms}
    cache_file = tmp_path / "channel_directory.json"
    cache_file.write_text(json.dumps(data))
    return cache_file


def _seed_sessions(tmp_path, sessions_data):
    """Write sessions.json at the path _build_from_sessions expects (its
    sessions.json fallback fires in tests because there is no state.db)."""
    sessions_path = tmp_path / "sessions" / "sessions.json"
    sessions_path.parent.mkdir(parents=True, exist_ok=True)
    sessions_path.write_text(json.dumps(sessions_data))


def _dm_session(platform, chat_id, name):
    """One sessions.json payload holding a single DM origin."""
    return {f"s_{platform}_{chat_id}": {
        "origin": {"platform": platform, "chat_id": chat_id, "chat_name": name},
        "chat_type": "dm",
    }}


class TestLoadDirectory:
    def test_missing_file(self, tmp_path):
        with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "nope.json"):
            result = load_directory()
        assert result["updated_at"] is None
        assert result["platforms"] == {}

    def test_null_platforms_map_is_treated_as_empty(self, tmp_path):
        """#48303 review NIT: a cache written as ``{"platforms": null}`` must not
        crash consumers. setdefault would hand back the stored None, and every
        read path (merge, resolve, display, lookup) iterates the map."""
        cache_file = _write_directory(tmp_path, None)
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            directory = load_directory()
            assert directory["platforms"] == {}
            # Consumers go through load_directory(); none may raise.
            assert resolve_channel_name("discord", "anything") is None
            assert lookup_channel_type("discord", "100") is None
            assert "No messaging platforms" in format_directory_for_display()


class TestBuildChannelDirectoryWrites:
    def test_failed_write_preserves_previous_cache(self, tmp_path, monkeypatch):
        cache_file = _write_directory(tmp_path, {
            "telegram": [{"id": "123", "name": "Alice", "type": "dm"}]
        })
        previous = json.loads(cache_file.read_text())

        import utils

        def broken_dump(data, fp, *args, **kwargs):
            fp.write('{"updated_at":')
            fp.flush()
            raise OSError("disk full")

        # Fault the canonical writer's serializer (the seam the directory writes through), not
        # json.dump — the helper serializes to a str first, so a stdlib patch never fires.
        monkeypatch.setattr(utils, "_dump_json", broken_dump)

        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            asyncio.run(build_channel_directory({}))
            result = load_directory()

        assert result == previous

    def test_uses_adapter_list_channels_when_available(self, tmp_path):
        class AdapterWithChannels:
            async def list_channels(self):
                return [
                    {"id": "default", "name": "主对话", "type": "dm"},
                    {"id": "family_1", "name": "达拉崩吧", "type": "group"},
                    {"id": "", "name": "ignored", "type": "dm"},
                    {"id": "family_1", "name": "duplicate", "type": "group"},
                ]

        cache_file = tmp_path / "channel_directory.json"
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: AdapterWithChannels()}))

        assert directory["platforms"]["telegram"] == [
            {"id": "default", "name": "主对话", "type": "dm"},
            {"id": "family_1", "name": "达拉崩吧", "type": "group"},
        ]


class TestBuildChannelDirectoryOffload:
    def test_discord_builder_runs_off_event_loop_thread(self, tmp_path):
        from gateway.config import Platform

        cache_file = tmp_path / "channel_directory.json"
        loop_thread = threading.get_ident()
        builder_threads = []

        def fake_build_discord(_adapter):
            builder_threads.append(threading.get_ident())
            return []

        with patch("gateway.channel_directory._build_discord", side_effect=fake_build_discord), \
             patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            asyncio.run(build_channel_directory({Platform.DISCORD: object()}))

        assert builder_threads
        assert all(tid != loop_thread for tid in builder_threads)

    def test_directory_write_runs_off_event_loop_thread(self, tmp_path):
        """The persist step calls os.fsync, which blocks the loop until the write
        reaches stable storage. #60794 moved the builders off the loop; the write
        stayed on it."""
        from gateway.config import Platform

        cache_file = tmp_path / "channel_directory.json"
        loop_thread = threading.get_ident()
        write_threads = []

        def fake_write(path, data, *args, **kwargs):
            write_threads.append(threading.get_ident())

        with patch("gateway.channel_directory.atomic_json_write", side_effect=fake_write), \
             patch("gateway.channel_directory._build_discord", return_value=[]), \
             patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            asyncio.run(build_channel_directory({Platform.DISCORD: object()}))

        assert write_threads
        assert all(tid != loop_thread for tid in write_threads)


class TestResolveChannelName:
    def _setup(self, tmp_path, platforms):
        cache_file = _write_directory(tmp_path, platforms)
        return patch("gateway.channel_directory.DIRECTORY_PATH", cache_file)

    def test_exact_match(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "111", "name": "bot-home", "guild": "MyServer", "type": "channel"},
                {"id": "222", "name": "general", "guild": "MyServer", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("discord", "bot-home") == "111"
            assert resolve_channel_name("discord", "#bot-home") == "111"

    def test_case_insensitive(self, tmp_path):
        platforms = {
            "slack": [{"id": "C01", "name": "Engineering", "type": "channel"}]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("slack", "engineering") == "C01"
            assert resolve_channel_name("slack", "ENGINEERING") == "C01"


    def test_prefix_match_unambiguous(self, tmp_path):
        platforms = {
            "slack": [
                {"id": "C01", "name": "engineering-backend", "type": "channel"},
                {"id": "C02", "name": "design-team", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            # "engineering" prefix matches only one channel
            assert resolve_channel_name("slack", "engineering") == "C01"


    def test_no_match_returns_none(self, tmp_path):
        platforms = {
            "telegram": [{"id": "123", "name": "John", "type": "dm"}]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("telegram", "nonexistent") is None


class TestSessionFallbackResolution:
    """#48303: the directory file lags live sessions by up to one rebuild
    interval (5 min), so a brand-new DM is resolvable from session data on a
    directory MISS — but only for platforms the directory already lists, or
    #60574's connected-only gate would regress by resurrecting stale targets."""

    def _home(self, tmp_path, directory_platforms, sessions_payload):
        cache_file = _write_directory(tmp_path, directory_platforms)
        _seed_sessions(tmp_path, sessions_payload)
        return (
            patch("gateway.channel_directory.DIRECTORY_PATH", cache_file),
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
        )

    def test_new_discord_dm_resolves_before_directory_rebuild(self, tmp_path):
        """A: stale directory (guild channel only) + session DM -> resolves to DM id."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {"discord": [{"id": "10", "name": "general", "guild": "MyServer", "type": "channel"}]},
            _dm_session("discord", "98765", "new-dm-contact"),
        )
        with dir_patch, home_patch:
            assert resolve_channel_name("discord", "new-dm-contact") == "98765"

    def test_directory_hit_takes_precedence_over_session_entry(self, tmp_path):
        """B: same name in directory and sessions -> the directory id wins."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {"discord": [{"id": "D1", "name": "dup-name", "type": "dm"}]},
            _dm_session("discord", "S1", "dup-name"),
        )
        with dir_patch, home_patch:
            assert resolve_channel_name("discord", "dup-name") == "D1"

    def test_platform_absent_from_directory_never_resurrected(self, tmp_path):
        """C: #60574 — session origins for a platform the directory omits stay invisible."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {"telegram": [{"id": "1", "name": "home", "type": "dm"}]},
            _dm_session("discord", "777", "ghost-dm"),
        )
        with dir_patch, home_patch:
            assert resolve_channel_name("discord", "ghost-dm") is None

    def test_empty_directory_list_still_resolves_from_sessions(self, tmp_path):
        """D: DM-only bot — the platform key exists but its channel list is empty;
        key-presence (not list-truthiness) must let the session fallback through."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {"discord": []},
            _dm_session("discord", "88888", "new-dm-contact"),
        )
        with dir_patch, home_patch:
            assert resolve_channel_name("discord", "new-dm-contact") == "88888"

    @pytest.mark.parametrize("infra_platform", ["local", "api_server", "webhook"])
    def test_infrastructure_platforms_never_resolve_from_sessions(self, tmp_path, infra_platform):
        """E: _SKIP_SESSION_DISCOVERY platforms are excluded from the fallback."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {infra_platform: []},
            _dm_session(infra_platform, "sess-1", "infra-contact"),
        )
        with dir_patch, home_patch:
            assert resolve_channel_name(infra_platform, "infra-contact") is None

    def test_directory_hit_does_not_consult_session_store(self, tmp_path):
        """F: laziness — the fallback is miss-only; a hit must not read sessions."""
        cache_file = _write_directory(tmp_path, {
            "telegram": [{"id": "1", "name": "home", "type": "dm"}],
        })
        calls = []
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file), \
             patch("gateway.channel_directory._build_from_sessions",
                   side_effect=lambda plat: calls.append(plat) or []):
            assert resolve_channel_name("telegram", "home") == "1"
        assert calls == []

    def test_directory_miss_falls_back_to_session_store(self, tmp_path):
        """F: a miss on a listed, session-discoverable platform consults sessions."""
        cache_file = _write_directory(tmp_path, {
            "telegram": [{"id": "1", "name": "home", "type": "dm"}],
        })
        calls = []
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file), \
             patch("gateway.channel_directory._build_from_sessions",
                   side_effect=lambda plat: calls.append(plat) or []):
            assert resolve_channel_name("telegram", "unknown-contact") is None
        assert calls == ["telegram"]

    def test_ambiguous_directory_prefix_is_not_silently_resolved_by_session_entry(self, tmp_path):
        """G: "foo" is ambiguous in the directory ("foobar" vs "foobaz"), so
        nothing may be silently chosen — even when the session entries alone
        look unambiguous. Uniqueness is a property of the whole candidate set;
        matching sessions in isolation manufactures a false prefix hit and
        mis-addresses the send (#48303 review follow-up)."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {"discord": [
                {"id": "111", "name": "foobar", "guild": "G", "type": "channel"},
                {"id": "222", "name": "foobaz", "guild": "G", "type": "channel"},
            ]},
            _dm_session("discord", "77", "foobar-dm"),
        )
        with dir_patch, home_patch:
            assert resolve_channel_name("discord", "foo") is None

    def test_unique_session_prefix_resolves_when_directory_has_no_competitor(self, tmp_path):
        """H: the union fix must not over-correct — a session-only name whose
        prefix is unique across directory AND sessions still resolves, by full
        name and by prefix."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {"discord": [{"id": "111", "name": "general", "guild": "G", "type": "channel"}]},
            _dm_session("discord", "77", "foobar-dm"),
        )
        with dir_patch, home_patch:
            assert resolve_channel_name("discord", "foobar") == "77"
            assert resolve_channel_name("discord", "foo") == "77"


class TestBuildFromSessions:
    def _write_sessions(self, tmp_path, sessions_data):
        """Write sessions.json at the path _build_from_sessions expects."""
        sessions_path = tmp_path / "sessions" / "sessions.json"
        sessions_path.parent.mkdir(parents=True)
        sessions_path.write_text(json.dumps(sessions_data))

    def test_builds_from_sessions_json(self, tmp_path):
        self._write_sessions(tmp_path, {
            "session_1": {
                "origin": {
                    "platform": "telegram",
                    "chat_id": "12345",
                    "chat_name": "Alice",
                },
                "chat_type": "dm",
            },
            "session_2": {
                "origin": {
                    "platform": "telegram",
                    "chat_id": "67890",
                    "user_name": "Bob",
                },
                "chat_type": "group",
            },
            "session_3": {
                "origin": {
                    "platform": "discord",
                    "chat_id": "99999",
                },
            },
        })

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            entries = _build_from_sessions("telegram")

        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert "Alice" in names
        assert "Bob" in names


class TestFormatDirectoryForDisplay:

    def test_platform_with_no_channels_gets_hint(self):
        """A configured platform with zero discovered channels is shown with
        a hint instead of being hidden entirely."""
        result = format_directory_for_display({
            "simplex": [],
            "telegram": [{"id": "1", "name": "home", "type": "dm"}],
        })
        assert "Simplex:" in result
        assert "telegram:home" in result

    def test_explicit_platforms_override_disk(self, tmp_path):
        with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "nope.json"):
            result = format_directory_for_display(
                {"irc": [{"id": "#chan", "name": "#chan", "type": "channel"}]}
            )
        assert "irc:#chan" in result


class TestFormatDirectorySessionMerge:
    """#48303: send_message(action="list") reads the on-disk directory, which
    lags live sessions by up to one rebuild interval; the disk-load path merges
    session-derived entries so a brand-new DM shows up immediately — without
    ever inventing a platform key (#60574)."""

    def _home(self, tmp_path, directory_platforms, sessions_payload):
        cache_file = _write_directory(tmp_path, directory_platforms)
        _seed_sessions(tmp_path, sessions_payload)
        return (
            patch("gateway.channel_directory.DIRECTORY_PATH", cache_file),
            patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}),
        )

    def test_new_session_dm_appears_in_display(self, tmp_path):
        """G: stale directory (guild channel) + session DM -> both render, DM grouped."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {"discord": [{"id": "10", "name": "general", "guild": "MyServer", "type": "channel"}]},
            _dm_session("discord", "98765", "new-dm-contact"),
        )
        with dir_patch, home_patch:
            result = format_directory_for_display()
        assert "discord:#general" in result
        assert "discord:new-dm-contact" in result
        # The merged DM lands in the DM group, not the guild group.
        assert "Discord (DMs):" in result
        assert result.index("Discord (DMs):") < result.index("discord:new-dm-contact")

    def test_session_entry_already_in_directory_not_duplicated(self, tmp_path):
        """H: merge is id-deduped against the directory."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {"discord": [{"id": "98765", "name": "already-known", "type": "dm"}]},
            _dm_session("discord", "98765", "already-known"),
        )
        with dir_patch, home_patch:
            result = format_directory_for_display()
        assert result.count("discord:already-known") == 1

    def test_display_never_invents_a_platform_key(self, tmp_path):
        """I: sessions for a platform absent from the directory must not surface it."""
        dir_patch, home_patch = self._home(
            tmp_path,
            {"telegram": [{"id": "1", "name": "home", "type": "dm"}]},
            _dm_session("whatsapp", "1500@g.us", "stale-group"),
        )
        with dir_patch, home_patch:
            result = format_directory_for_display()
        assert "whatsapp" not in result
        assert "stale-group" not in result

    def test_explicit_override_does_not_consult_sessions(self, tmp_path):
        """J: an explicit ``platforms`` dict is the caller's view — no merge."""
        calls = []
        with patch("gateway.channel_directory._build_from_sessions",
                   side_effect=lambda plat: calls.append(plat) or []):
            format_directory_for_display(
                {"telegram": [{"id": "1", "name": "home", "type": "dm"}]}
            )
        assert calls == []


class TestLookupChannelType:
    def _setup(self, tmp_path, platforms):
        cache_file = _write_directory(tmp_path, platforms)
        return patch("gateway.channel_directory.DIRECTORY_PATH", cache_file)

    def test_forum_channel(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "100", "name": "ideas", "guild": "Server1", "type": "forum"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert lookup_channel_type("discord", "100") == "forum"


    def test_unknown_chat_id_returns_none(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "200", "name": "general", "guild": "Server1", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert lookup_channel_type("discord", "999") is None


def _make_slack_adapter(team_clients):
    """Build a stand-in for SlackAdapter exposing only ``_team_clients``."""
    return SimpleNamespace(_team_clients=team_clients)


def _make_slack_client(pages):
    """Build an AsyncWebClient mock whose ``users_conversations`` returns pages."""
    client = MagicMock()
    client.users_conversations = AsyncMock(side_effect=pages)
    return client


class TestBuildSlack:
    """_build_slack actually calls users.conversations on each workspace client."""

    def test_no_team_clients_falls_back_to_sessions(self, tmp_path):
        sessions_path = tmp_path / "sessions" / "sessions.json"
        sessions_path.parent.mkdir(parents=True)
        sessions_path.write_text(json.dumps({
            "s1": {"origin": {"platform": "slack", "chat_id": "D123", "chat_name": "Alice"}},
        }))

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            entries = asyncio.run(_build_slack(_make_slack_adapter({})))

        assert len(entries) == 1
        assert entries[0]["id"] == "D123"

    def test_lists_channels_from_users_conversations(self, tmp_path):
        client = _make_slack_client([
            {
                "ok": True,
                "channels": [
                    {"id": "C0B0QV5434G", "name": "engineering", "is_private": False},
                    {"id": "G123ABCDEF", "name": "secret-chat", "is_private": True},
                ],
                "response_metadata": {},
            },
        ])
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            entries = asyncio.run(_build_slack(_make_slack_adapter({"T1": client})))

        ids = {e["id"] for e in entries}
        assert ids == {"C0B0QV5434G", "G123ABCDEF"}
        types = {e["id"]: e["type"] for e in entries}
        assert types["C0B0QV5434G"] == "channel"
        assert types["G123ABCDEF"] == "private"
        client.users_conversations.assert_awaited_once()

    def test_paginates_via_response_metadata_cursor(self, tmp_path):
        client = _make_slack_client([
            {
                "ok": True,
                "channels": [{"id": "C001", "name": "first", "is_private": False}],
                "response_metadata": {"next_cursor": "cur1"},
            },
            {
                "ok": True,
                "channels": [{"id": "C002", "name": "second", "is_private": False}],
                "response_metadata": {"next_cursor": ""},
            },
        ])
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            entries = asyncio.run(_build_slack(_make_slack_adapter({"T1": client})))

        assert {e["id"] for e in entries} == {"C001", "C002"}
        assert client.users_conversations.await_count == 2

    def test_thread_ids_use_base_conversation_and_dedupe_info_calls(self, tmp_path, monkeypatch):
        client = _make_slack_client([{"ok": True, "channels": [], "response_metadata": {}}])
        client.conversations_info = AsyncMock(side_effect=[
            {"ok": True, "channel": {"name": "engineering"}},
            {"ok": True, "channel": {"name": "support"}},
        ])
        monkeypatch.setattr(
            "gateway.channel_directory._build_from_sessions",
            lambda platform: [
                {"id": "C001:111", "name": "C001:111", "type": "channel"},
                {"id": "C001:222", "name": "C001:222", "type": "channel"},
                {"id": "C002:333", "name": "C002:333", "type": "channel"},
            ],
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            entries = asyncio.run(_build_slack(_make_slack_adapter({"T1": client})))

        assert {entry["name"] for entry in entries} == {"engineering", "support"}
        assert client.conversations_info.await_count == 2
        assert [call.kwargs["channel"] for call in client.conversations_info.await_args_list] == ["C001", "C002"]


class TestChannelAliases:
    """The user-maintained alias overlay (channel_aliases.json) gives durable
    friendly names that survive the timed directory rebuild."""

    def _setup_aliases(self, tmp_path, aliases):
        alias_file = tmp_path / "channel_aliases.json"
        alias_file.write_text(json.dumps(aliases))
        return patch("gateway.channel_directory.CHANNEL_ALIASES_PATH", alias_file)


    def test_alias_injects_undiscovered_group(self, tmp_path):
        """A group named in the alias file but not yet seen in any session is
        still addressable by name (pre-naming before first traffic)."""
        cache_file = _write_directory(tmp_path, {"whatsapp": []})
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file), \
             self._setup_aliases(tmp_path, {"whatsapp": {"999@g.us": "marketing"}}):
            assert resolve_channel_name("whatsapp", "marketing") == "999@g.us"
            entries = load_directory()["platforms"]["whatsapp"]
            injected = [e for e in entries if e["id"] == "999@g.us"]
            assert injected and injected[0]["type"] == "group"


    def test_alias_persists_through_rebuild(self, tmp_path, monkeypatch):
        """build_channel_directory must bake aliases into the written file so
        they survive the periodic regeneration, not just live reads."""
        cache_file = tmp_path / "channel_directory.json"
        monkeypatch.setattr("gateway.channel_directory._build_from_sessions",
                            lambda plat: [{"id": "120363@g.us", "name": "120363",
                                           "type": "group", "thread_id": None}]
                            if plat == "whatsapp" else [])
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file), \
             self._setup_aliases(tmp_path, {"whatsapp": {"120363@g.us": "general"}}):
            asyncio.run(build_channel_directory({}))
            on_disk = json.loads(cache_file.read_text())
        names = [e["name"] for e in on_disk["platforms"]["whatsapp"]
                 if e["id"] == "120363@g.us"]
        assert names == ["general"]

