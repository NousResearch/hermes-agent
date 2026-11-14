"""Discord split delivery never cuts a reply off (issue #86581, #152012).

The #86581 incident delivered 60,698 chars as 31 messages. A reply that needs more than
``MAX_SPLIT_MESSAGES`` now goes out as one file (``test_reply_file_handoff.py``); when that upload
fails the gateway falls back to ``send()``, which must then deliver every chunk, in order, with no
truncation notice (Teknium's ruling on PR #153476: Discord matches Telegram). discord.py's
rate-limit buckets pace the sends.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter


MAX = DiscordAdapter.MAX_MESSAGE_LENGTH
CAP = DiscordAdapter.MAX_SPLIT_MESSAGES


def _make_adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def _huge_content(chars: int = 60_000) -> str:
    # Distinct filler — this test is about SIZE, not repetition.
    return " ".join(f"word-{i}-" + "x" * 12 for i in range(chars // 20))


class TestSendDeliversEveryChunk:
    @pytest.mark.asyncio
    async def test_send_delivers_every_chunk_past_the_cap(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        sends = []

        async def fake_send(*, content, reference=None):
            sends.append(content)
            return SimpleNamespace(id=9000 + len(sends))

        channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))
        adapter._client = SimpleNamespace(
            get_channel=lambda _cid: channel,
            fetch_channel=AsyncMock(),
        )

        content = _huge_content()
        result = await adapter.send("555", content)

        assert result.success is True
        assert len(sends) > CAP
        assert sends == adapter.truncate_message(adapter.format_message(content), MAX)


class TestForumDeliversEveryChunk:
    @pytest.mark.asyncio
    async def test_send_to_forum_delivers_every_followup_chunk(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        thread_sends = []

        async def fake_thread_send(*, content):
            thread_sends.append(content)
            return SimpleNamespace(id=8000 + len(thread_sends))

        thread_channel = SimpleNamespace(
            id=777, send=AsyncMock(side_effect=fake_thread_send)
        )
        forum_channel = SimpleNamespace(
            id=666,
            type=SimpleNamespace(value=15),
            create_thread=AsyncMock(return_value=SimpleNamespace(
                id=777,
                thread=thread_channel,
                message=SimpleNamespace(id=8000),
            )),
        )

        content = _huge_content()
        result = await adapter._send_to_forum(forum_channel, content)

        assert result.success is True
        # The starter carries chunk 1; every later chunk follows in the thread.
        chunks = adapter.truncate_message(adapter.format_message(content), MAX)
        assert forum_channel.create_thread.call_args.kwargs["content"] == chunks[0]
        assert thread_sends == chunks[1:]


class TestEditOverflowDeliversEveryChunk:
    @pytest.mark.asyncio
    async def test_edit_overflow_split_delivers_every_chunk(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        edits = []
        sends = []

        async def fake_edit(*, content):
            edits.append(content)

        async def fake_send(*, content, reference=None):
            sends.append(content)
            return SimpleNamespace(id=9000 + len(sends))

        msg = SimpleNamespace(id=42, edit=AsyncMock(side_effect=fake_edit))
        channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))

        content = _huge_content()
        result = await adapter._edit_overflow_split(channel, msg, "42", content)

        assert result.success is True
        # 1 in-place edit + every remaining chunk as a continuation send.
        chunks = adapter.truncate_message(adapter.format_message(content), MAX)
        assert edits == chunks[:1]
        assert sends == chunks[1:]
