import asyncio
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
async def test_exec_approval_view_honors_discord_timeout_with_request_id(monkeypatch):
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, "read_raw_config", lambda: {
        "approvals": {"timeout": 7200, "discord_prompt_timeout": 3600},
    })
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_exec_approval(
        chat_id="555", command="ls", session_key="agent:other:discord:555",
        request_id="request-1",
    )

    assert result.success
    assert sent["view"].timeout == 3600
    assert sent["view"].request_id == "request-1"


@pytest.mark.asyncio
async def test_expired_card_cannot_resolve_a_different_pending_request(monkeypatch):
    from plugins.platforms.discord import adapter as discord_adapter
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)
    session = "agent:other:discord:555"
    await adapter.send_exec_approval(
        chat_id="555", command="ls", session_key=session,
        request_id="expired",
    )
    other = _ApprovalEntry({"request_id": "new", "command": "rm other"})
    monkeypatch.setitem(approval._gateway_queues, session, [other])
    view = sent["view"]
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))
    monkeypatch.setattr(view, "_finalize_embed", AsyncMock())
    monkeypatch.setattr(discord_adapter.discord, "Color", SimpleNamespace(dark_grey=lambda: 0))
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="tester"), response=SimpleNamespace(send_message=AsyncMock()))

    await view._resolve(interaction, "once", 1, "gateway.exec_approval.action_once")

    assert not other.event.is_set()
    assert approval._gateway_queues[session] == [other]
    # Its own request already settled elsewhere: the clicker hears so, the card is left alone.
    view._finalize_embed.assert_not_awaited()
    interaction.response.send_message.assert_awaited_once_with(
        t("platform.discord.approval.no_longer_pending"), ephemeral=True)


@pytest.mark.asyncio
async def test_unsettled_click_without_an_observer_marks_the_card_expired(monkeypatch):
    """A card with no request id has no observer, so its own no-op click stamps it, as before."""
    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord.adapter import ExecApprovalView

    view = ExecApprovalView(session_key="agent:other:discord:555", allowed_user_ids=set())
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))
    monkeypatch.setattr(view, "_finalize_embed", AsyncMock())
    monkeypatch.setattr(discord_adapter.discord, "Color", SimpleNamespace(dark_grey=lambda: 0))
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="tester"), response=SimpleNamespace(send_message=AsyncMock()))

    await view._resolve(interaction, "once", 1, "gateway.exec_approval.action_once")

    view._finalize_embed.assert_awaited_once_with(interaction, 0, t("platform.discord.approval.expired"))
    interaction.response.send_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("ends,footer_key", [
    ("after_send", "platform.discord.prompt.expired_footer"),
    ("during_send", "platform.discord.prompt.expired_footer"),
    ("before_send", "platform.discord.approval.no_longer_pending"),
])
async def test_card_retires_when_backend_request_ends(monkeypatch, ends, footer_key):
    """The card shows how its request ended, even when that happened while it was being sent."""
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    session = "agent:other:discord:555"
    entry = _ApprovalEntry({"request_id": "request-1", "command": "ls"})
    monkeypatch.setitem(approval._gateway_queues, session, [] if ends == "before_send" else [entry])
    message = SimpleNamespace(id=1234, edit=AsyncMock(), embeds=[])
    sent = {}

    async def send(**kwargs):
        sent.update(kwargs)
        message.embeds = [kwargs["embed"]]
        if ends == "during_send":
            entry.observers[0]("timeout")  # the wait times out while Discord is still sending
            await asyncio.sleep(0)
        return message

    channel = SimpleNamespace(send=send)
    adapter._client = SimpleNamespace(get_channel=lambda _id: channel)
    await adapter.send_exec_approval(chat_id="555", command="ls", session_key=session, request_id="request-1")
    view = sent["view"]
    if ends == "after_send":
        assert not view.resolved
        entry.observers[0]("timeout")
        await asyncio.sleep(0)
        await asyncio.gather(*adapter._background_tasks)
    assert view.resolved and view.is_finished()
    message.edit.assert_awaited_once()
    assert message.edit.await_args.kwargs["view"] is None
    footer = message.embeds[0].footer
    assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(footer_key)


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
