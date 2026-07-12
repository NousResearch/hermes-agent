from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
import plugins.platforms.discord.adapter as discord_adapter
from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    _remove_discord_view_item_by_label,
)


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
    assert "Command Approval Required" in prompt_text
    assert "Do you want Hermes to run this command?" in prompt_text
    assert "Requested command" in prompt_text
    assert command in prompt_text
    assert "Reason" in prompt_text
    assert "script execution via -c flag" in prompt_text


@pytest.mark.asyncio
async def test_exec_approval_hides_always_when_permanent_approval_disallowed(monkeypatch):
    captured = {}

    class _CapturedView:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(discord_adapter, "ExecApprovalView", _CapturedView)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_exec_approval(
        chat_id="555",
        command="curl http://172.16.0.2/internal",
        session_key="discord:555",
        allow_permanent=False,
    )

    assert result.success is True
    assert captured["allow_permanent"] is False


def test_discord_view_removes_permanent_approval_button_by_label():
    always = SimpleNamespace(label="Always Allow")
    view = SimpleNamespace(
        children=[
            SimpleNamespace(label="Allow Once"),
            SimpleNamespace(label="Allow Session"),
            always,
            SimpleNamespace(label="Deny"),
        ]
    )
    view.remove_item = view.children.remove

    _remove_discord_view_item_by_label(view, "Always Allow")

    assert [child.label for child in view.children] == [
        "Allow Once",
        "Allow Session",
        "Deny",
    ]


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
    assert len(sent["embed"].description) > len(sent["content"])
