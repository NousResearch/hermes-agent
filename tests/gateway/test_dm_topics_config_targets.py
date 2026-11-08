"""Configured Telegram DM topics stay reusable: across adapter rebuilds/reconnects and in the channel directory."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
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
    from gateway.channel_directory import build_channel_directory, resolve_channel_name

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 591994976, "topics": [{"name": "General", "thread_id": 306210},
                                                         {"name": "Reports", "thread_id": 306214}, {"name": "Draft"}]},
                      {"chat_id": 777, "topics": [{"name": "Reports", "thread_id": 9}]}],
        "group_topics": [{"chat_id": -1001, "topics": [{"name": "Ops", "thread_id": 5}]}]}))
    session_entry = {"id": "591994976:306210", "name": "Tester / General", "type": "dm", "thread_id": "306210"}

    with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"), \
         patch("gateway.channel_directory._build_from_sessions", return_value=[session_entry]):
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: adapter}))
        entries = {e["id"]: e["name"] for e in directory["platforms"]["telegram"]}
        assert entries == {"591994976:306210": "Tester / General", "591994976:306214": "591994976 / Reports",
                           "777:9": "777 / Reports", "-1001:5": "Ops"}
        assert resolve_channel_name("telegram", "Ops") == "-1001:5"
        assert resolve_channel_name("telegram", "777 / Reports") == "777:9"

def test_topic_names_differing_only_by_hash_are_qualified(tmp_path):
    """``Reports`` and ``#Reports`` resolve as the same name, so both must be chat-qualified."""
    from gateway.channel_directory import build_channel_directory, resolve_channel_name

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={
        "dm_topics": [{"chat_id": 111, "topics": [{"name": "Reports", "thread_id": 1}]},
                      {"chat_id": 222, "topics": [{"name": "#Reports", "thread_id": 2}]}]}))
    with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "channel_directory.json"), \
         patch("gateway.channel_directory._build_from_sessions", return_value=[]):
        directory = asyncio.run(build_channel_directory({Platform.TELEGRAM: adapter}))
        for entry in directory["platforms"]["telegram"]:
            assert resolve_channel_name("telegram", entry["name"]) == entry["id"]
