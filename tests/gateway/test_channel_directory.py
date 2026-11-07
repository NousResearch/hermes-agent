"""Tests for gateway/channel_directory.py — channel resolution and display."""

import asyncio
import json
import os
import threading
from pathlib import Path
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


class TestLoadDirectory:
    def test_missing_file(self, tmp_path):
        with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "nope.json"):
            result = load_directory()
        assert result["updated_at"] is None
        assert result["platforms"] == {}


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


class TestMultiplexServedProfileRefresh:
    """#141023: a multiplexed gateway must rebuild each served profile's OWN
    channel_directory.json — from that profile's adapters, inside that profile's runtime
    scope — so name-based targets resolve in that profile and not only in the launch home."""

    @staticmethod
    def _runner(multiplex, homes, profile_adapters):
        return SimpleNamespace(
            config=SimpleNamespace(multiplex_profiles=multiplex, _runtime_profile_homes=homes),
            _profile_adapters=profile_adapters,
        )

    @staticmethod
    def _record_builds(monkeypatch):
        """Replace the builder with a recorder of (home in scope, adapters handed to it)."""
        from hermes_constants import get_hermes_home

        calls = []

        async def _build(adapters):
            calls.append((Path(get_hermes_home()).name, adapters))
            return {"updated_at": "2026-01-01T00:00:00", "platforms": {}}

        monkeypatch.setattr("gateway.channel_directory.build_channel_directory", _build)
        # Keep the real runtime scope but skip profile secret hydration (off-loop .env work).
        monkeypatch.setattr("gateway.run._load_profile_secret_scope", lambda _home: {})
        return calls

    def test_single_profile_gateway_is_untouched(self, tmp_path, monkeypatch):
        """No served secondaries, no scoped build: the launch build stays the profile's own."""
        from gateway.run import _refresh_served_profile_channel_directories

        calls = self._record_builds(monkeypatch)
        runner = self._runner(False, [("default", tmp_path)], {"seo": {Platform.TELEGRAM: object()}})

        assert asyncio.run(_refresh_served_profile_channel_directories(runner)) == 0
        assert calls == []

    def test_each_served_profile_is_built_in_its_own_scope(self, tmp_path, monkeypatch):
        """One build per served profile, under that profile's home, with that profile's adapters —
        and never a second build of the launch home."""
        from gateway.run import _refresh_served_profile_channel_directories

        for name in ("launch", "seo", "mubeen"):
            (tmp_path / name).mkdir()
        seo_adapter, mubeen_adapter = object(), object()
        calls = self._record_builds(monkeypatch)
        runner = self._runner(
            True,
            [("default", tmp_path / "launch"), ("seo", tmp_path / "seo"), ("mubeen", tmp_path / "mubeen")],
            {"seo": {Platform.TELEGRAM: seo_adapter}, "mubeen": {Platform.TELEGRAM: mubeen_adapter}},
        )

        rebuilt = asyncio.run(_refresh_served_profile_channel_directories(runner))

        assert rebuilt == 2
        assert [home for home, _adapters in calls] == ["seo", "mubeen"]
        assert [adapters for _home, adapters in calls] == [
            {Platform.TELEGRAM: seo_adapter}, {Platform.TELEGRAM: mubeen_adapter}]

    def test_profile_without_its_own_adapters_fails_closed(self, tmp_path, monkeypatch):
        """A served profile whose adapter map is still the empty startup placeholder gets an EMPTY
        map, never the launch profile's. ``authz_mixin._adapters_for_profile`` gives a named profile
        ``{}`` precisely so it cannot deliver through the default bot; a directory advertising the
        launch profile's channels would resolve name-based targets (``slack:#ops``, cron
        ``deliver: slack:<name>``) to channels that profile cannot reach."""
        from gateway.run import _refresh_served_profile_channel_directories

        (tmp_path / "launch").mkdir()
        (tmp_path / "seo").mkdir()
        calls = self._record_builds(monkeypatch)
        launch_adapters = {Platform.TELEGRAM: object()}
        runner = self._runner(True, [("default", tmp_path / "launch"), ("seo", tmp_path / "seo")],
                              {"seo": {}})

        assert asyncio.run(_refresh_served_profile_channel_directories(runner)) == 1
        assert calls == [("seo", {})]
        assert launch_adapters not in [adapters for _home, adapters in calls]

    def test_real_build_writes_into_the_profile_home(self, tmp_path, monkeypatch):
        """End to end through the real builder: the scope is what puts the file in the profile's
        home (``_directory_path()`` -> ``get_hermes_home()``), which the pre-fix code never did."""
        from gateway.run import _refresh_served_profile_channel_directories

        profile_home = tmp_path / "seo"
        profile_home.mkdir()
        monkeypatch.setattr("gateway.run._load_profile_secret_scope", lambda _home: {})

        class _Adapter:
            async def list_channels(self):
                return [{"id": "C0NEW", "name": "just-joined", "type": "channel"}]

        runner = self._runner(True, [("seo", profile_home)], {"seo": {Platform.SLACK: _Adapter()}})

        assert asyncio.run(_refresh_served_profile_channel_directories(runner)) == 1

        written = json.loads((profile_home / "channel_directory.json").read_text())
        assert [entry["name"] for entry in written["platforms"]["slack"]] == ["just-joined"]
        assert written["updated_at"]

    @pytest.mark.asyncio
    async def test_startup_call_site_rebuilds_each_served_profile(self, tmp_path, monkeypatch):
        """The startup pass must actually REACH the per-profile rebuild.

        The helper takes the runner alone. A second argument at this call site raises before the
        body runs, and the surrounding ``_log_suppressed`` swallows the ``TypeError``: startup
        then logs a misleading warning and no served profile is rebuilt until the 5-minute
        housekeeping tick. Every other test here calls the helper directly, so the call site
        itself was uncovered.
        """
        from gateway.run import GatewayRunner

        for name in ("launch", "seo", "mubeen"):
            (tmp_path / name).mkdir()
        seo_adapter, mubeen_adapter = object(), object()
        calls = self._record_builds(monkeypatch)

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = SimpleNamespace(
            multiplex_profiles=True,
            _runtime_profile_homes=[
                ("default", tmp_path / "launch"), ("seo", tmp_path / "seo"),
                ("mubeen", tmp_path / "mubeen")],
        )
        runner._profile_adapters = {
            "seo": {Platform.TELEGRAM: seo_adapter},
            "mubeen": {Platform.TELEGRAM: mubeen_adapter},
        }
        runner.adapters = {}
        runner._heartbeat_watch = {}
        runner._background_tasks = set()
        runner._ensure_hosted_room_worker = AsyncMock()
        runner._hosted_room_worker_watcher = AsyncMock()
        runner._spawn_supervised = lambda *args: None
        runner._start_loop_heartbeat_task = lambda: None
        runner.hooks = SimpleNamespace(loaded_hooks=[], emit=AsyncMock())
        runner._send_update_notification = AsyncMock(return_value=True)
        runner.session_store = SimpleNamespace(list_sessions=list)
        runner._run_in_executor_with_context = asyncio.to_thread

        await runner._start_post_connect_services(0)
        task = getattr(runner, "_heartbeat_poll_task", None)
        try:
            # The launch build happens first (unscoped), then one build per served profile, each
            # in ITS OWN home — the startup path used to raise here and rebuild neither.
            assert [home for home, _adapters in calls if home in ("seo", "mubeen")] == ["seo", "mubeen"]
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

