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


def _msg(uid, bot=False):
    return SimpleNamespace(author=SimpleNamespace(id=uid, bot=bot))


def _make_thread(messages, starter_msg=None, parent_chan=None):
    """Build a minimal object that passes ``isinstance(x, discord.Thread)``.

    Subclassing without calling ``discord.Thread.__init__`` sidesteps the
    gateway-state plumbing; class attributes shadow the ``starter_message`` /
    ``parent`` properties.
    """
    import discord

    class _FakeThread(discord.Thread):
        starter_message = starter_msg
        parent = parent_chan

        def __init__(self):
            self._msgs = list(messages)
            self.sent_kwargs = None

        def history(self, *, limit=None, oldest_first=False):
            msgs = self._msgs

            async def _gen():
                for m in msgs:
                    yield m

            return _gen()

        async def send(self, **kwargs):
            self.sent_kwargs = kwargs
            return SimpleNamespace(id=12345)

    return _FakeThread()


def _make_adapter(channel):
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"222", "111"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)
    return adapter


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
    assert channel.sent_kwargs["embed"].title.endswith("Command Approval Required")


@pytest.mark.asyncio
async def test_exec_approval_participants_scope_mentions_thread_participants(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS_SCOPE", "participants")
    thread = _make_thread([_msg(999, bot=True), _msg(111), _msg(333)])
    adapter = _make_adapter(thread)

    result = await adapter.send_exec_approval(
        chat_id="99",
        command="make check",
        session_key="session-1",
        description="dangerous command",
    )

    assert result.success is True
    # Only the approver active in this thread is pinged; 333 is not an
    # approver, 999 is a bot, and 222 is working elsewhere.
    assert thread.sent_kwargs["content"].startswith("<@111>\n")
    assert "<@222>" not in thread.sent_kwargs["content"]


@pytest.mark.asyncio
async def test_exec_approval_participants_scope_falls_back_to_all(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS_SCOPE", "participants")
    thread = _make_thread([_msg(999, bot=True)])
    adapter = _make_adapter(thread)

    result = await adapter.send_exec_approval(
        chat_id="99",
        command="make check",
        session_key="session-1",
        description="dangerous command",
    )

    assert result.success is True
    assert thread.sent_kwargs["content"].startswith("<@111> <@222>\n")


@pytest.mark.asyncio
async def test_exec_approval_participants_scope_uses_starter_message_author(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS_SCOPE", "participants")
    # Fresh auto-thread: no human messages inside yet, requester authored the
    # parent-channel message the thread hangs off.
    thread = _make_thread([_msg(999, bot=True)], starter_msg=_msg(222))
    adapter = _make_adapter(thread)

    result = await adapter.send_exec_approval(
        chat_id="99",
        command="make check",
        session_key="session-1",
        description="dangerous command",
    )

    assert result.success is True
    assert thread.sent_kwargs["content"].startswith("<@222>\n")
    assert "<@111>" not in thread.sent_kwargs["content"]


@pytest.mark.asyncio
async def test_exec_approval_participants_scope_non_thread_mentions_all(monkeypatch):
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS", "true")
    monkeypatch.setenv("DISCORD_APPROVAL_MENTIONS_SCOPE", "participants")
    channel = _FakeChannel()
    adapter = _make_adapter(channel)

    result = await adapter.send_exec_approval(
        chat_id="99",
        command="make check",
        session_key="session-1",
        description="dangerous command",
    )

    assert result.success is True
    assert channel.sent_kwargs["content"].startswith("<@111> <@222>\n")


def test_yaml_config_bridges_approval_mentions_scope(monkeypatch):
    monkeypatch.delenv("DISCORD_APPROVAL_MENTIONS_SCOPE", raising=False)

    _apply_yaml_config({}, {"approval_mentions_scope": "Participants"})

    assert os.environ["DISCORD_APPROVAL_MENTIONS_SCOPE"] == "participants"


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


