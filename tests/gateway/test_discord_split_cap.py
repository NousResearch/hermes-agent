"""Regression tests for the Discord split-delivery cap (issue #86581).

A degenerate turn can produce tens of thousands of characters.  Without a
ceiling, the adapter posts every 2000-char chunk back-to-back and floods the
channel — the #86581 incident delivered 60,698 chars as 31 messages.  The
delivery cap now routes oversized responses to one Markdown attachment instead
of flooding the channel. The pure cap helper remains a bounded fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path
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
        assert capped[-1] not in chunks  # last slot is the truncation notice
        # The notice itself must stay under Discord's per-message cap.
        assert len(capped[-1]) <= MAX


@pytest.fixture
def uploads(monkeypatch):
    # The shared SDK mock's File does not model path-based files or close().
    # Read the actual generated payload; only Discord's transport is simulated.
    from plugins.platforms.discord.adapter import discord

    captured = []

    def fake_file(path, *, filename):
        file = SimpleNamespace(
            filename=filename, data=Path(path).read_bytes(), close=MagicMock(),
        )
        captured.append(file)
        return file

    monkeypatch.setattr(discord, "File", fake_file)
    return captured


def _assert_bounded_upload(result, calls, uploads, original, upload_fails):
    assert result.success is (not upload_fails)
    assert len(uploads) == 1
    assert uploads[0].filename.endswith(".md")
    assert uploads[0].data == original.encode("utf-8")
    assert len(calls) == (2 if upload_fails else 1)
    assert len(calls) <= CAP
    assert all(0 < len(content) <= MAX for content in calls)
    if upload_fails:
        assert result.error
        assert "Response truncated" in calls[-1]
    else:
        assert "attached" in calls[0]
        assert "Response truncated" not in calls[0]


class TestSendCap:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("upload_fails", [False, True])
    async def test_send_caps_split_flood(self, monkeypatch, tmp_path, uploads, upload_fails):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        sends = []
        original = "# Résumé 🌍\n\n" + _huge_content()

        async def fake_send(*, content, reference=None, files=None):
            sends.append(content)
            if files and upload_fails:
                raise RuntimeError("upload denied")
            return SimpleNamespace(id=9000 + len(sends), attachments=files or [])

        channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))
        adapter._client = SimpleNamespace(
            get_channel=lambda _cid: channel,
            fetch_channel=AsyncMock(),
        )

        result = await adapter.send("555", original)

        _assert_bounded_upload(result, sends, uploads, original, upload_fails)
        assert sum(bool(call.kwargs.get("files")) for call in channel.send.await_args_list) == 1


class TestForumCap:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("upload_fails", [False, True])
    async def test_send_to_forum_caps_followup_chunks(self, monkeypatch, tmp_path, uploads, upload_fails):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        starters = []
        original = "# Résumé 🌍\n\n" + _huge_content()
        thread_channel = SimpleNamespace(id=777, send=AsyncMock())

        async def fake_create_thread(*, name, content, files=None):
            starters.append(content)
            if files and upload_fails:
                raise RuntimeError("upload denied")
            return SimpleNamespace(
                id=777, thread=thread_channel,
                message=SimpleNamespace(id=8000, attachments=files or []),
            )

        forum_channel = SimpleNamespace(
            id=666,
            type=SimpleNamespace(value=15),
            create_thread=AsyncMock(side_effect=fake_create_thread),
        )
        adapter._client = SimpleNamespace(get_channel=lambda _cid: forum_channel)

        result = await adapter._send_to_forum(forum_channel, original)

        _assert_bounded_upload(result, starters, uploads, original, upload_fails)
        thread_channel.send.assert_not_awaited()
        assert sum(bool(call.kwargs.get("files")) for call in forum_channel.create_thread.await_args_list) == 1


class TestEditOverflowCap:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("upload_fails", [False, True])
    async def test_edit_overflow_split_capped(self, monkeypatch, tmp_path, uploads, upload_fails):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        edits = []
        original = "# Résumé 🌍\n\n" + _huge_content()

        async def fake_edit(*, content, attachments=None):
            edits.append(content)
            if attachments and upload_fails:
                raise RuntimeError("upload denied")
            return SimpleNamespace(id=42, attachments=attachments or [])

        msg = SimpleNamespace(id=42, edit=AsyncMock(side_effect=fake_edit))
        channel = SimpleNamespace(id=555, send=AsyncMock())
        adapter._client = SimpleNamespace(get_channel=lambda _cid: channel)

        result = await adapter._edit_overflow_split(channel, msg, "42", original)

        _assert_bounded_upload(result, edits, uploads, original, upload_fails)
        channel.send.assert_not_awaited()
        assert sum(bool(call.kwargs.get("attachments")) for call in msg.edit.await_args_list) == 1
        assert not result.continuation_message_ids
        if not upload_fails:
            assert result.message_id == "42"
