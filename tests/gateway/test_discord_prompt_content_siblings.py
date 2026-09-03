"""Sibling coverage for the embed-invisibility fix (send_exec_approval got it
in the same PR): slash confirm, clarify, and update prompts must also mirror
their payload into plain message content, since embeds don't render on some
Discord clients (web/mobile)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as discord_adapter
from plugins.platforms.discord.adapter import DiscordAdapter


def _capture_channel(adapter):
    sent = {}

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    return sent


@pytest.mark.asyncio
async def test_slash_confirm_mirrors_message_into_content():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_slash_confirm(
        chat_id="555",
        title="Reset session?",
        message="This will clear the current conversation history.",
        session_key="discord:555",
        confirm_id="c1",
    )

    assert result.success is True
    assert sent["view"] is not None
    assert sent["embed"] is not None
    assert "Reset session?" in sent["content"]
    assert "clear the current conversation history" in sent["content"]


@pytest.mark.asyncio
async def test_clarify_with_choices_mirrors_question_into_content():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_clarify(
        chat_id="555",
        question="Which environment should I deploy to?",
        choices=["staging", "production"],
        clarify_id="cl1",
        session_key="discord:555",
    )

    assert result.success is True
    assert sent["view"] is not None
    assert "Hermes needs your input" in sent["content"]
    assert "Which environment should I deploy to?" in sent["content"]
    assert "Pick one below" in sent["content"]


@pytest.mark.asyncio
async def test_clarify_mentions_only_the_target_user(monkeypatch):
    class FakeAllowedMentions:
        def __init__(self, *, users, roles, everyone, replied_user):
            self.users = users
            self.roles = roles
            self.everyone = everyone
            self.replied_user = replied_user

    monkeypatch.setattr(discord_adapter.discord, "AllowedMentions", FakeAllowedMentions)
    monkeypatch.setattr(
        discord_adapter.discord,
        "Object",
        lambda *, id: SimpleNamespace(id=id),
    )
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_clarify(
        chat_id="555",
        question="Choose one, not <@999>.",
        choices=["first", "second"],
        clarify_id="cl2",
        session_key="discord:555",
        metadata={"user_id": "123"},
    )

    assert result.success is True
    assert sent["content"].startswith("<@123>\n")
    assert [user.id for user in sent["allowed_mentions"].users] == [123]
    assert sent["allowed_mentions"].everyone is False
    assert sent["allowed_mentions"].roles is False


@pytest.mark.asyncio
async def test_clarify_without_target_user_stays_unmentioned():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_clarify(
        chat_id="555",
        question="Which environment should I deploy to?",
        choices=None,
        clarify_id="cl3",
        session_key="discord:555",
    )

    assert result.success is True
    assert sent["content"].startswith("❓ **Hermes needs your input**")
    assert "allowed_mentions" not in sent


