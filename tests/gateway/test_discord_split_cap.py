"""Regression tests for the Discord split-delivery cap (issue #86581).

A degenerate turn can produce tens of thousands of characters.  Without a
ceiling, the adapter posts every 2000-char chunk back-to-back and floods the
channel — the #86581 incident delivered 60,698 chars as 31 messages.  The
cap keeps the first ``MAX_SPLIT_MESSAGES`` chunks and replaces the remainder
with a short notice.
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

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


MAX = DiscordAdapter.MAX_MESSAGE_LENGTH
CAP = DiscordAdapter.MAX_SPLIT_MESSAGES


def _make_adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def _huge_content(chars: int = 60_000) -> str:
    # Distinct filler — this test is about SIZE, not repetition.
    return " ".join(f"word-{i}-" + "x" * 12 for i in range(chars // 20))


class TestCapSplitChunks:
    def test_below_cap_unchanged(self):
        adapter = _make_adapter()
        chunks = ["a", "b", "c"]
        assert adapter._cap_split_chunks(chunks) == chunks

    def test_over_cap_keeps_n_minus_1_plus_notice(self):
        adapter = _make_adapter()
        chunks = [f"chunk-{i}-" + "z" * 100 for i in range(40)]
        capped = adapter._cap_split_chunks(chunks)
        assert len(capped) == CAP
        assert capped[0] == chunks[0]
        assert "Response truncated" in capped[-1]
        assert "delivery limit" in capped[-1]
        # The notice itself must stay under Discord's per-message cap.
        assert len(capped[-1]) <= MAX


class TestSendCap:
    @pytest.mark.asyncio
    async def test_send_caps_split_flood(self, monkeypatch, tmp_path):
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

        result = await adapter.send("555", _huge_content())

        assert result.success is True
        assert len(sends) == CAP
        assert "Response truncated" in sends[-1]


class TestForumCap:
    @pytest.mark.asyncio
    async def test_send_to_forum_caps_followup_chunks(self, monkeypatch, tmp_path):
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

        result = await adapter._send_to_forum(forum_channel, _huge_content())

        assert result.success is True
        # 1 starter message + at most (CAP - 1) follow-up chunks.
        assert len(thread_sends) <= CAP - 1
        assert "Response truncated" in thread_sends[-1]


class TestEditOverflowCap:
    @pytest.mark.asyncio
    async def test_edit_overflow_split_capped(self, monkeypatch, tmp_path):
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

        result = await adapter._edit_overflow_split(channel, msg, "42", _huge_content())

        assert result.success is True
        # 1 in-place edit + at most (CAP - 1) continuation sends.
        assert len(edits) == 1
        assert len(sends) <= CAP - 1
        assert "Response truncated" in (sends[-1] if sends else edits[-1])
