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
async def test_exec_approval_prompt_uses_visible_content_with_command_and_reason():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    command = "python scripts/deploy.py --env prod --force"
    result = await adapter.send_exec_approval(
        chat_id="555",
        command=command,
        session_key="discord:555",
        description="script execution via -c flag",
    )

    assert result.success is True
    assert sent["view"] is not None
    assert sent["embed"] is not None

    prompt_text = sent["content"]
    assert "Hermes wants to run a command that needs your OK" in prompt_text
    assert "Do you want Hermes to run this command?" in prompt_text
    assert "Requested command" in prompt_text
    assert command in prompt_text
    assert "Why it was flagged" in prompt_text
    assert "script execution via -c flag" in prompt_text
    assert "it will NOT run" in prompt_text

    # Content is the canonical, accessible approval payload (embeds may not render, #33681);
    # the embed is a header-only card so the command and reason appear exactly once.
    embed = sent["embed"]
    embed_text = (embed.description or "") + "".join(field.value or "" for field in embed.fields)
    assert command not in embed_text
    assert "script execution via -c flag" not in embed_text


@pytest.mark.asyncio
async def test_exec_approval_content_stays_within_discord_cap_for_long_reason():
    """The reason shares the 2000-char content cap with the command preview."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_exec_approval(
        chat_id="555", command="x" * 5000, session_key="discord:555", description="r" * 5000)

    assert result.success is True
    assert len(sent["content"]) <= adapter.MAX_MESSAGE_LENGTH
    assert "xxx" in sent["content"]  # the command preview is not starved to zero



def _embed_text(embed):
    # Fields are attribute objects on discord.py and dicts on the test stub.
    return (embed.description or "") + "".join(
        str(getattr(f, "value", None) or (f.get("value") if isinstance(f, dict) else "") or "")
        for f in embed.fields)


@pytest.mark.asyncio
async def test_slash_confirm_embed_is_header_only_card():
    """Same rule as the approval prompt: content carries the message once, the embed is a header card."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    await adapter.send_slash_confirm(
        chat_id="555", title="Reset session?", message="This will clear the conversation history.",
        session_key="discord:555", confirm_id="c1")

    assert "clear the conversation history" in sent["content"]
    assert "clear the conversation history" not in _embed_text(sent["embed"])


@pytest.mark.asyncio
async def test_clarify_embed_is_header_only_card():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    await adapter.send_clarify(
        chat_id="555", question="Which environment should I deploy to?", choices=["staging", "prod"],
        clarify_id="cl1", session_key="discord:555")

    assert "Which environment should I deploy to?" in sent["content"]
    assert "Pick one below" in sent["content"]
    assert "Which environment should I deploy to?" not in _embed_text(sent["embed"])


@pytest.mark.asyncio
async def test_update_prompt_embed_is_header_only_card():
    """Same rule as the other prompt cards: content carries the payload once, the embed is a
    header card — send_update_prompt kept description=... (duplicating into the embed too)
    after its siblings were converted to header-only."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    await adapter.send_update_prompt(
        chat_id="555", prompt="Continue the update?", default="yes", session_key="discord:555")

    assert "Continue the update?" in sent["content"]
    assert "Continue the update?" not in _embed_text(sent["embed"])
