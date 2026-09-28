"""Discord approval prompts can opt into owner mentions."""

import os
from types import SimpleNamespace

import pytest

from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    _apply_yaml_config,
)


class _FakeChannel:
    def __init__(self):
        self.sent_kwargs = None

    async def send(self, **kwargs):
        self.sent_kwargs = kwargs
        return SimpleNamespace(id=12345)


class _FakeClient:
    def __init__(self, channel):
        self.channel = channel

    def get_channel(self, channel_id):
        return self.channel


def _message(user_id, *, bot=False):
    return SimpleNamespace(author=SimpleNamespace(id=user_id, bot=bot))


def _thread(messages, starter_message=None):
    import discord

    class FakeThread(discord.Thread):
        parent = None

        def __init__(self):
            self._messages = messages
            self.sent_kwargs = None

        def history(self, *, limit=None):
            async def generate():
                for message in self._messages:
                    yield message
            return generate()

        async def send(self, **kwargs):
            self.sent_kwargs = kwargs
            return SimpleNamespace(id=12345)

    FakeThread.starter_message = starter_message
    return FakeThread()


@pytest.mark.asyncio
async def test_exec_approval_mentions_allowed_users_when_enabled(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    channel = _FakeChannel()
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"222", "111", "alice"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)

    result = await adapter.send_exec_approval(
        chat_id="99",
        command="make check",
        session_key="session-1",
        description="dangerous command",
    )

    assert result.success is True
    # Mentions are prepended to the (always present) content mirror.
    assert channel.sent_kwargs["content"].startswith("<@111> <@222>\n")
    assert "make check" in channel.sent_kwargs["content"]
    assert "allowed_mentions" in channel.sent_kwargs


@pytest.mark.asyncio
async def test_exec_approval_participants_scope_mentions_thread_participants(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS_SCOPE", "participants")
    channel = _thread([_message("999", bot=True), _message("111"), _message("333")])
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"222", "111"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)

    result = await adapter.send_exec_approval(
        chat_id="99", command="make check", session_key="session-1", description="dangerous command",
    )

    assert result.success is True
    assert channel.sent_kwargs["content"].startswith("<@111>\n")
    assert "<@222>" not in channel.sent_kwargs["content"]


@pytest.mark.asyncio
async def test_exec_approval_participants_scope_uses_thread_starter(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS_SCOPE", "participants")
    channel = _thread([_message("999", bot=True)], starter_message=_message("222"))
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"222", "111"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)

    result = await adapter.send_exec_approval(
        chat_id="99", command="make check", session_key="session-1", description="dangerous command",
    )

    assert result.success is True
    assert channel.sent_kwargs["content"].startswith("<@222>\n")
    assert "<@111>" not in channel.sent_kwargs["content"]


@pytest.mark.asyncio
async def test_exec_approval_participants_scope_combines_history_and_thread_starter(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS_SCOPE", "participants")
    channel = _thread([_message("111")], starter_message=_message("222"))
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"222", "111"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)

    result = await adapter.send_exec_approval(
        chat_id="99", command="make check", session_key="session-1", description="dangerous command",
    )

    assert result.success is True
    assert channel.sent_kwargs["content"].startswith("<@111> <@222>\n")


@pytest.mark.asyncio
async def test_exec_approval_participants_scope_falls_back_to_all_when_nobody_matches(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS_SCOPE", "participants")
    channel = _thread([_message("999", bot=True), _message("888")])
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"222", "111"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)

    result = await adapter.send_exec_approval(
        chat_id="99", command="make check", session_key="session-1", description="dangerous command",
    )

    assert result.success is True
    assert channel.sent_kwargs["content"].startswith("<@111> <@222>\n")


def test_yaml_config_seeds_approval_mentions_scope(monkeypatch):
    monkeypatch.delenv("DISCORD_APPROVAL_MENTIONS_SCOPE", raising=False)

    seeded = _apply_yaml_config({}, {"approval_mentions_scope": "Participants"})

    assert os.environ["DISCORD_APPROVAL_MENTIONS_SCOPE"] == "participants"
    assert seeded["approval_mentions_scope"] == "participants"


def test_yaml_config_seeds_websocket_health_with_primary_precedence(monkeypatch):
    for key in (
        "HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS",
        "HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)

    seeded = _apply_yaml_config(
        {},
        {
            "websocket_liveness_interval_seconds": 11,
            "liveness_interval_seconds": 99,
            "websocket_liveness_failure_threshold": 2,
            "websocket_heartbeat_ack_max_age_seconds": 75,
            "websocket_max_latency_seconds": 30,
        },
    )

    assert os.environ["HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS"] == "11"
    assert os.environ["HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD"] == "2"
    assert seeded == {
        "websocket_liveness_interval_seconds": 11,
        "websocket_liveness_failure_threshold": 2,
        "websocket_heartbeat_ack_max_age_seconds": 75,
        "websocket_max_latency_seconds": 30,
    }


