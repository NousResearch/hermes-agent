import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as discord_adapter
from plugins.platforms.discord.adapter import DiscordAdapter


pytestmark = pytest.mark.skipif(
    not discord_adapter.DISCORD_AVAILABLE,
    reason="discord.py is not available",
)


def _make_adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    client = MagicMock()
    client.user = SimpleNamespace(id=222222222222222222)
    adapter._client = client
    return adapter


def test_approval_notify_mentions_reads_env_with_dedupe(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setenv(
        "DISCORD_APPROVAL_NOTIFY_MENTIONS",
        "<@111111111111111111>, <@111111111111111111> <@999999999999999999>",
    )
    monkeypatch.setenv("DISCORD_OPERATOR_MENTIONS", "<@ignored>")

    assert adapter._discord_approval_notify_mentions() == [
        "<@111111111111111111>",
        "<@999999999999999999>",
    ]


def test_approval_notify_mentions_falls_back_to_operator_mentions(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.delenv("DISCORD_APPROVAL_NOTIFY_MENTIONS", raising=False)
    monkeypatch.setenv("DISCORD_OPERATOR_MENTIONS", "<@111111111111111111>")

    assert adapter._discord_approval_notify_mentions() == ["<@111111111111111111>"]


@pytest.mark.asyncio
async def test_send_exec_approval_mentions_supervisor_and_includes_text_command_fallback(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setenv("DISCORD_APPROVAL_NOTIFY_MENTIONS", "<@***>")

    channel = MagicMock()
    channel.send = AsyncMock(return_value=SimpleNamespace(id=12345))
    client = adapter._client
    assert client is not None
    client.get_channel.return_value = channel

    result = await adapter.send_exec_approval(
        chat_id="1507587363164131349",
        command="rm -rf /tmp/nonexistent-approval-smoke",
        session_key="discord:thread:abc",
        description="approval visibility smoke",
    )

    assert result.success is True
    channel.send.assert_awaited_once()
    content = channel.send.await_args.kwargs["content"]
    assert content == "<@***> approval required"
    assert channel.send.await_args.kwargs["embed"] is not None
    assert channel.send.await_args.kwargs["view"] is not None


@pytest.mark.asyncio
async def test_send_exec_approval_mentions_allowed_supervisor_without_bot_msg(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setenv("DISCORD_APPROVAL_NOTIFY_MENTIONS", "<@777>")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")

    channel = MagicMock()
    channel.send = AsyncMock(return_value=SimpleNamespace(id=12345))
    client = adapter._client
    assert client is not None
    client.get_channel.return_value = channel

    result = await adapter.send_exec_approval(
        chat_id="1507587363164131349",
        command="rm -rf /tmp/nonexistent-approval-smoke",
        session_key="discord:thread:abc",
        description="approval visibility smoke",
    )

    assert result.success is True
    assert channel.send.await_args.kwargs["content"] == "<@777> approval required"


@pytest.mark.asyncio
async def test_send_exec_approval_keeps_content_optional_without_notify(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.delenv("DISCORD_APPROVAL_NOTIFY_MENTIONS", raising=False)
    monkeypatch.delenv("DISCORD_OPERATOR_MENTIONS", raising=False)

    channel = MagicMock()
    channel.send = AsyncMock(return_value=SimpleNamespace(id=12345))
    client = adapter._client
    assert client is not None
    client.get_channel.return_value = channel

    result = await adapter.send_exec_approval(
        chat_id="1507587363164131349",
        command="echo ok",
        session_key="discord:thread:abc",
    )

    assert result.success is True
    assert channel.send.await_args.kwargs["content"] is None
