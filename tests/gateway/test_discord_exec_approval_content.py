from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, ExecApprovalView


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
    assert "embed" not in sent

    prompt_text = sent["content"]
    assert "Command Approval Required" in prompt_text
    assert "Do you want Hermes to run this command?" in prompt_text
    assert "Requested command" in prompt_text
    assert command in prompt_text
    assert "Reason" in prompt_text
    assert "script execution via -c flag" in prompt_text


@pytest.mark.asyncio
async def test_exec_approval_prompt_truncates_long_command_in_content():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    long_command = "python -c '" + ("x" * 5000) + "'"
    result = await adapter.send_exec_approval(
        chat_id="555",
        command=long_command,
        session_key="discord:555",
        description="long generated shell command",
    )

    assert result.success is True
    assert len(sent["content"]) <= adapter.MAX_MESSAGE_LENGTH
    assert "... [truncated]" in sent["content"]
    assert "long generated shell command" in sent["content"]
    assert "embed" not in sent


@pytest.mark.asyncio
async def test_exec_approval_resolution_marks_plain_content_and_disables_buttons(monkeypatch):
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda *_args: 1)
    view = ExecApprovalView(session_key="discord:555", allowed_user_ids={"1"})
    response = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())
    interaction = SimpleNamespace(
        message=SimpleNamespace(content="⚠️ **Command Approval Required**", embeds=[]),
        response=response,
        user=SimpleNamespace(id=1, display_name="Cael", roles=[]),
    )

    await view._resolve(interaction, "once", "Approved once")

    kwargs = response.edit_message.await_args.kwargs
    assert kwargs["content"].startswith("✅ Approved once by Cael")
    assert "Command Approval Required" in kwargs["content"]
    assert kwargs["view"] is view
    assert all(button.disabled for button in view.children)


@pytest.mark.asyncio
async def test_exec_approval_resolution_keeps_status_within_discord_content_limit(monkeypatch):
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda *_args: 1)
    view = ExecApprovalView(session_key="discord:555", allowed_user_ids={"1"})
    response = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())
    interaction = SimpleNamespace(
        message=SimpleNamespace(content="x" * 2000, embeds=[]),
        response=response,
        user=SimpleNamespace(id=1, display_name="Cael", roles=[]),
    )

    await view._resolve(interaction, "once", "Approved once")

    content = response.edit_message.await_args.kwargs["content"]
    assert content.startswith("✅ Approved once by Cael")
    assert len(content) <= 2000


@pytest.mark.asyncio
async def test_exec_approval_resolution_unblocks_agent_when_message_edit_fails(monkeypatch):
    resolved = []
    monkeypatch.setattr(
        "tools.approval.resolve_gateway_approval",
        lambda *args: resolved.append(args) or 1,
    )
    view = ExecApprovalView(session_key="discord:555", allowed_user_ids={"1"})
    response = SimpleNamespace(
        edit_message=AsyncMock(side_effect=RuntimeError("Discord API unavailable")),
        send_message=AsyncMock(),
    )
    interaction = SimpleNamespace(
        message=SimpleNamespace(content="⚠️ **Command Approval Required**", embeds=[]),
        response=response,
        user=SimpleNamespace(id=1, display_name="Cael", roles=[]),
    )

    await view._resolve(interaction, "once", "Approved once")

    assert resolved == [("discord:555", "once")]


@pytest.mark.asyncio
async def test_exec_approval_timeout_marks_plain_content_and_disables_buttons():
    view = ExecApprovalView(session_key="discord:555", allowed_user_ids={"1"})
    message = SimpleNamespace(
        content="⚠️ **Command Approval Required**",
        embeds=[],
        edit=AsyncMock(),
    )
    view._message = message

    await view.on_timeout()

    kwargs = message.edit.await_args.kwargs
    assert kwargs["content"].startswith("⏱ Prompt expired — no action taken")
    assert "Command Approval Required" in kwargs["content"]
    assert kwargs["view"] is view
    assert all(button.disabled for button in view.children)
