"""Sibling coverage for the embed-invisibility fix (send_exec_approval got it
in the same PR): slash confirm, clarify, and update prompts must also mirror
their payload into plain message content, since embeds don't render on some
Discord clients (web/mobile)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
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
    assert "Which environment should I deploy to?" in sent["content"]




@pytest.mark.asyncio
async def test_clarify_drops_duplicate_title_header_when_configured():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***", extra={"prompt_header_in_content": False}))
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
    assert sent["embed"] is not None
    assert sent["embed"].title == "❓ Hermes needs your input"
    # Duplicate title header is suppressed from content
    assert "Hermes needs your input" not in sent["content"]
    # The actual question, choices hint, and buttons view are fully preserved
    assert sent["content"].startswith("Which environment should I deploy to?")
    assert "Pick one below, or click ✏️ Other" in sent["content"]


@pytest.mark.asyncio
async def test_clarify_preserves_title_header_by_default():
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
    assert "❓ **Hermes needs your input**" in sent["content"]
    assert "Which environment should I deploy to?" in sent["content"]
    assert "Pick one below, or click ✏️ Other" in sent["content"]


@pytest.mark.asyncio
async def test_slash_confirm_drops_duplicate_title_header_when_configured():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***", extra={"prompt_header_in_content": False}))
    sent = _capture_channel(adapter)

    result = await adapter.send_slash_confirm(
        chat_id="555",
        title="Reset session?",
        message="This will clear the current conversation history.",
        session_key="discord:555",
        confirm_id="c1",
    )

    assert result.success is True
    assert sent["embed"].title == "Reset session?"
    assert "Reset session?" not in sent["content"]
    assert sent["content"] == "This will clear the current conversation history."


@pytest.mark.asyncio
async def test_exec_approval_drops_duplicate_title_header_when_configured():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***", extra={"prompt_header_in_content": False}))
    sent = _capture_channel(adapter)

    command = "python scripts/deploy.py --env prod --force"
    result = await adapter.send_exec_approval(
        chat_id="555",
        command=command,
        session_key="discord:555",
        description="script execution via -c flag",
    )

    assert result.success is True
    assert sent["embed"] is not None
    assert sent["embed"].title == "⚠️ Hermes wants to run a command that needs your OK"
    # Title header is suppressed from content
    assert "Hermes wants to run a command that needs your OK" not in sent["content"]
    # Question, command, reason, and buttons are fully preserved
    assert "Do you want Hermes to run this command?" in sent["content"]
    assert command in sent["content"]
    assert "script execution via -c flag" in sent["content"]
