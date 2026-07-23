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
async def test_exec_approval_does_not_mention_by_default(monkeypatch):
    monkeypatch.delenv("DISCORD_APPROVAL_MENTIONS", raising=False)
    channel = _FakeChannel()
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient(channel)
    adapter._allowed_user_ids = {"111"}
    adapter._allowed_role_ids = set()
    adapter.config = SimpleNamespace(extra=None)

    result = await adapter.send_exec_approval(
        chat_id="99",
        command="make check",
        session_key="session-1",
    )

    assert result.success is True
    # Content mirror is always present (embed-invisibility fix), but no
    # mention markup and no allowed_mentions override.
    assert "<@" not in channel.sent_kwargs["content"]
    assert "allowed_mentions" not in channel.sent_kwargs


def test_yaml_config_bridges_approval_mentions_to_env(monkeypatch):
    monkeypatch.delenv("DISCORD_APPROVAL_MENTIONS", raising=False)

    _apply_yaml_config(
        {"discord": {"approval_mentions": True}},
        {"approval_mentions": True},
    )

    assert os.environ["DISCORD_APPROVAL_MENTIONS"] == "true"
