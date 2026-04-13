from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.discord import DiscordAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_send_retries_without_reference_when_reply_target_is_system_message():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    ref_msg = SimpleNamespace(id=99)
    sent_msg = SimpleNamespace(id=1234)
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 50035): Invalid Form Body\n"
                "In message_reference: Cannot reply to a system message"
            )
        return sent_msg

    channel = SimpleNamespace(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "hello", reply_to="99")

    assert result.success is True
    assert result.message_id == "1234"
    assert channel.fetch_message.await_count == 1
    assert channel.send.await_count == 2
    assert send_calls[0]["reference"] is ref_msg
    assert send_calls[1]["reference"] is None


@pytest.mark.asyncio
async def test_send_formats_continuation_chunks_without_trailing_parenthesized_suffix():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    sent_msg = SimpleNamespace(id=1234)
    sent_chunks = []

    async def fake_send(*, content, reference=None):
        sent_chunks.append({"content": content, "reference": reference})
        return sent_msg

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    adapter.truncate_message = lambda content, max_len: ["First chunk (1/2)", "Second chunk (2/2)"]

    result = await adapter.send("555", "Long answer")

    assert result.success is True
    assert sent_chunks == [
        {"content": "First chunk", "reference": None},
        {"content": "↪ Continued 2/2\nSecond chunk", "reference": None},
    ]


@pytest.mark.asyncio
async def test_send_formats_prechunked_stream_continuation_metadata():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    sent_msg = SimpleNamespace(id=5678)
    sent_chunks = []

    async def fake_send(*, content, reference=None):
        sent_chunks.append({"content": content, "reference": reference})
        return sent_msg

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send(
        "555",
        "Pre-chunked second chunk (2/2)",
        metadata={"_discord_chunk_position": {"index": 2, "total": 2}},
    )

    assert result.success is True
    assert sent_chunks == [
        {"content": "↪ Continued 2/2\nPre-chunked second chunk", "reference": None},
    ]


@pytest.mark.asyncio
async def test_send_keeps_continuation_chunks_within_discord_limit():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    sent_msg = SimpleNamespace(id=9012)
    sent_chunks = []

    async def fake_send(*, content, reference=None):
        if len(content) > adapter.MAX_MESSAGE_LENGTH:
            raise RuntimeError(
                "400 Bad Request (error code: 50035): Invalid Form Body\n"
                "In content: Must be 2000 or fewer in length."
            )
        sent_chunks.append({"content": content, "reference": reference})
        return sent_msg

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "a" * 5000)

    assert result.success is True
    assert len(sent_chunks) >= 3
    assert all(len(chunk["content"]) <= adapter.MAX_MESSAGE_LENGTH for chunk in sent_chunks)
    assert sent_chunks[1]["content"].startswith("↪ Continued 2/")
