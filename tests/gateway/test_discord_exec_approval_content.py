from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.i18n import t
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
    assert command in prompt_text
    assert "script execution via -c flag" in prompt_text

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


@pytest.mark.asyncio
async def test_reason_content_past_the_old_clip_still_renders():
    """A context-first reason must not render as header-only: its decisive content can sit past
    the old 300-char clip, and the operator needs it to judge what they are approving."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    reason = ("task: nightly digest review | profile: media | model: local | " + "meta " * 60
              + "DECISIVE-INSTRUCTION: check the payload before approving")

    result = await adapter.send_exec_approval(
        chat_id="555", command="ls -la", session_key="discord:555", description=reason)

    content = sent["content"]
    assert result.success is True
    assert "DECISIVE-INSTRUCTION" in content
    assert len(content) <= adapter.MAX_MESSAGE_LENGTH


@pytest.mark.asyncio
async def test_every_cut_is_labelled_with_the_catalogued_marker():
    """A bare "..." reads as a render glitch; each cut — the reason and the command preview —
    carries the catalogued marker, and the content still fits the cap."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    await adapter.send_exec_approval(
        chat_id="555", command="c" * 5000, session_key="discord:555",
        description="r" * 1400 + "REASON-TAIL-Z")

    content = sent["content"]
    assert "REASON-TAIL-Z" not in content  # the reason was cut past its budget...
    assert content.count(t("platform.discord.prompt.truncated_marker")) >= 2  # ...and both cuts say so
    assert len(content) <= adapter.MAX_MESSAGE_LENGTH


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
    assert "Which environment should I deploy to?" not in _embed_text(sent["embed"])
