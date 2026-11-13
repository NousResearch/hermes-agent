"""Configured Telegram DM topics stay reusable: across adapter rebuilds/reconnects and in the channel directory."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hermes_agent.gateway.config import Platform, PlatformConfig
from tests.gateway.conftest import _ensure_telegram_mock

_ensure_telegram_mock()
sys.modules.pop("plugins.platforms.telegram.adapter", None)

from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_reconnect_reuses_dm_topics_created_by_this_process():
    """A same-process reconnect re-runs setup on the config it was built with: the topics it created
    on the first run must be reused, not created again."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 222, "topics": [{"name": "WEBSITE"}, {"name": "CONTENT"}]}]}))
    ids = iter((58890, 58892, 1, 2))
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.side_effect = lambda **kw: SimpleNamespace(message_thread_id=next(ids))
    adapter._persist_dm_topic_thread_id = lambda *a, **k: None
    adapter._reload_dm_topics_from_config = lambda: None  # no config.yaml re-read: only the in-memory state

    await adapter._setup_dm_topics()
    await adapter._setup_dm_topics()

    assert adapter._bot.create_forum_topic.await_count == 2
    assert adapter._dm_topics == {"222:WEBSITE": 58890, "222:CONTENT": 58892}


def test_channel_directory_lists_configured_topics_without_sessions(tmp_path):
    from hermes_agent.gateway.channel_directory import build_channel_directory, resolve_channel_name

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 591994976, "topics": [{"name": "General", "thread_id": 306210},
                                                         {"name": "Reports", "thread_id": 306214}, {"name": "Draft"}]},
                      {"chat_id": 777, "topics": [{"name": "Reports", "thread_id": 9}]}],
        "group_topics": [{"chat_id": -1001, "topics": [{"name": "Ops", "thread_id": 5}]}]}))
    session_entry = {"id": "591994976:306210", "name": "Tester / General", "type": "dm", "thread_id": "306210"}

    with patch("hermes_agent.gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"), \
         patch("hermes_agent.gateway.channel_directory._build_from_sessions", return_value=[session_entry]):
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: adapter}))
        entries = {e["id"]: e["name"] for e in directory["platforms"]["telegram"]}
        assert entries == {"591994976:306210": "Tester / General", "591994976:306214": "591994976 / Reports",
                           "777:9": "777 / Reports", "-1001:5": "Ops"}
        assert resolve_channel_name("telegram", "Ops") == "-1001:5"
        assert resolve_channel_name("telegram", "777 / Reports") == "777:9"


def test_topic_names_differing_only_by_hash_are_qualified(tmp_path):
    """``Reports`` and ``#Reports`` resolve as the same name, so both must be chat-qualified."""
    from hermes_agent.gateway.channel_directory import build_channel_directory, resolve_channel_name

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 111, "topics": [{"name": "Reports", "thread_id": 1}]},
                      {"chat_id": 222, "topics": [{"name": "#Reports", "thread_id": 2}]}]}))
    with patch("hermes_agent.gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"), \
         patch("hermes_agent.gateway.channel_directory._build_from_sessions", return_value=[]):
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: adapter}))
        for entry in directory["platforms"]["telegram"]:
            assert resolve_channel_name("telegram", entry["name"]) == entry["id"]


def test_unnamed_group_topic_is_listed(tmp_path):
    from hermes_agent.gateway.channel_directory import build_channel_directory, resolve_channel_name

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "group_topics": [{"chat_id": -1001, "topics": [{"thread_id": 5, "skill": "arxiv"}]}]}))
    with patch("hermes_agent.gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"), \
         patch("hermes_agent.gateway.channel_directory._build_from_sessions", return_value=[]):
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: adapter}))
        (entry,) = directory["platforms"]["telegram"]
        assert entry["id"] == "-1001:5"
        assert resolve_channel_name("telegram", entry["name"]) == "-1001:5"


@pytest.mark.asyncio
async def test_topic_setup_does_not_mutate_shared_config_cache(tmp_path, monkeypatch):
    """Setup writes created thread_ids into its own copy; the process-wide config cache only learns
    them through the persist path."""
    from hermes_cli.config import load_config_readonly

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  telegram:\n    extra:\n      dm_topics:\n"
        "      - chat_id: 222\n        topics:\n        - name: WEBSITE\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 222, "topics": [{"name": "WEBSITE"}]}]}))
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.return_value = SimpleNamespace(message_thread_id=58890)
    adapter._persist_dm_topic_thread_id = lambda *a, **k: None  # the save fails / is skipped

    await adapter._setup_dm_topics()

    assert adapter._dm_topics == {"222:WEBSITE": 58890}
    topics = load_config_readonly()["platforms"]["telegram"]["extra"]["dm_topics"][0]["topics"]
    assert "thread_id" not in topics[0]


@pytest.mark.asyncio
async def test_reconnect_after_failed_save_reuses_the_created_topic(tmp_path, monkeypatch):
    """Reload stays enabled and the thread_id save fails: the reconnect's fresh config copy lacks the
    id, so setup must reuse the one this adapter created instead of creating a duplicate topic."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  telegram:\n    extra:\n      dm_topics:\n"
        "      - chat_id: 222\n        topics:\n        - name: WEBSITE\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 222, "topics": [{"name": "WEBSITE"}]}]}))
    ids = iter((58890, 58891))
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.side_effect = lambda **kw: SimpleNamespace(message_thread_id=next(ids))
    adapter._persist_dm_topic_thread_id = lambda *a, **k: None  # the save fails

    await adapter._setup_dm_topics()
    await adapter._setup_dm_topics()

    assert adapter._bot.create_forum_topic.await_count == 1
    assert adapter._dm_topics == {"222:WEBSITE": 58890}


@pytest.mark.asyncio
async def test_reconnect_after_failed_save_keeps_the_topic_listed(tmp_path, monkeypatch):
    """The reused unsaved id must stay a channel-directory target, not only a cache entry."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  telegram:\n    extra:\n      dm_topics:\n"
        "      - chat_id: 222\n        topics:\n        - name: WEBSITE\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 222, "topics": [{"name": "WEBSITE"}]}]}))
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.return_value = SimpleNamespace(message_thread_id=58890)
    adapter._persist_dm_topic_thread_id = lambda *a, **k: None  # the save fails

    await adapter._setup_dm_topics()
    await adapter._setup_dm_topics()

    assert [e["id"] for e in adapter.configured_channels()] == ["222:58890"]


def test_configured_topic_name_never_resolves_to_a_discovered_chat(tmp_path):
    """A discovered DM named ``Reports`` (chat 111) and a configured ``Reports`` topic (222:9): every
    listed label resolves to its own entry, and the bare shared name resolves to neither."""
    from hermes_agent.gateway.channel_directory import build_channel_directory, resolve_channel_name

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 222, "topics": [{"name": "Reports", "thread_id": 9}]}]}))
    discovered = [{"id": "111", "name": "Reports", "type": "dm", "thread_id": None}]
    with patch("hermes_agent.gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"), \
         patch("hermes_agent.gateway.channel_directory._build_from_sessions", return_value=discovered):
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: adapter}))
        entries = directory["platforms"]["telegram"]
        assert {e["id"] for e in entries} == {"111", "222:9"}
        for entry in entries:
            assert resolve_channel_name("telegram", entry["name"]) == entry["id"]
        assert resolve_channel_name("telegram", "Reports") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("config_yaml", [
    'telegram:\n  dm_topics:\n  - chat_id: "222"\n    topics:\n    - name: WEBSITE\n',
    'gateway:\n  platforms:\n    telegram:\n      extra:\n        dm_topics:\n'
    '        - chat_id: "222"\n          topics:\n          - name: WEBSITE\n',
    'platforms:\n  telegram:\n    extra:\n      dm_topics:\n      - chat_id: 222\n        topics:\n        - name: WEBSITE\n',
], ids=["top_level_telegram", "gateway_platforms", "platforms_telegram_extra"])
async def test_dm_topics_outside_platforms_telegram_extra_are_set_up(tmp_path, monkeypatch, config_yaml):
    """Every dm_topics shape the gateway loader accepts (top-level ``telegram:``, ``gateway.platforms``,
    ``platforms.telegram.extra``) is set up, and a rebuild from the same boot-time config reuses the thread_id the first adapter saved instead of creating it again."""
    from hermes_agent.gateway.config import load_gateway_config

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(config_yaml)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    boot_config = load_gateway_config().platforms[Platform.TELEGRAM]
    adapters = []
    for _ in range(2):  # boot, then a rebuild from the boot-time snapshot
        adapter = TelegramAdapter(boot_config)
        adapter._bot = AsyncMock()
        adapter._bot.create_forum_topic.return_value = SimpleNamespace(message_thread_id=58890)
        await adapter._setup_dm_topics()
        adapters.append(adapter)

    assert [a._bot.create_forum_topic.await_count for a in adapters] == [1, 0]
    assert adapters[1]._dm_topic_chat_ids == {"222"}
    assert [e["id"] for e in adapters[1].configured_channels()] == ["222:58890"]


@pytest.mark.asyncio
async def test_saved_thread_id_wins_over_the_boot_snapshot(tmp_path, monkeypatch):
    """A topic re-created since boot (``force_create`` saves the new id): setup on the boot-time config
    must use the saved id, not the stale one the snapshot carries."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  telegram:\n    extra:\n      dm_topics:\n"
        "      - chat_id: 222\n        topics:\n        - name: WEBSITE\n          thread_id: 200\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 222, "topics": [{"name": "WEBSITE", "thread_id": 100}]}]}))
    adapter._bot = AsyncMock()

    await adapter._setup_dm_topics()

    adapter._bot.create_forum_topic.assert_not_awaited()
    assert adapter._dm_topics == {"222:WEBSITE": 200}
