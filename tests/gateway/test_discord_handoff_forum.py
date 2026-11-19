"""``DiscordAdapter.create_handoff_thread`` under a forum channel.

A forum channel only accepts posts: ``create_thread`` without a starter message is rejected and a
forum has no ``send`` for the seed-message fallback. Before the fix both paths failed (the second
silently), so ``/branch`` run inside a forum post replaced the post's session instead of opening a
sibling post.
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

FORUM_TYPE = SimpleNamespace(value=15)
TEXT_TYPE = SimpleNamespace(value=0)


def _adapter_with_parent(parent):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(get_channel=lambda _cid: parent, fetch_channel=AsyncMock())
    return adapter


@pytest.mark.asyncio
async def test_forum_parent_opens_a_post_with_a_starter_message(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    post = SimpleNamespace(id=777, send=AsyncMock())
    # discord.py returns ThreadWithMessage(thread, message) for a forum post.
    created = SimpleNamespace(thread=post, message=SimpleNamespace(id=778))
    forum = SimpleNamespace(id=100, type=FORUM_TYPE, create_thread=AsyncMock(return_value=created))
    adapter = _adapter_with_parent(forum)

    thread_id = await adapter.create_handoff_thread("100", "查看预览 #2")

    assert thread_id == "777"
    forum.create_thread.assert_awaited_once()
    kwargs = forum.create_thread.await_args.kwargs
    assert kwargs["name"] == "查看预览 #2"
    assert kwargs["content"] and "查看预览 #2" in kwargs["content"]


@pytest.mark.asyncio
async def test_forum_post_failure_returns_none_and_logs(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    forum = SimpleNamespace(
        id=100, type=FORUM_TYPE, create_thread=AsyncMock(side_effect=RuntimeError("40067 tag required")),
    )
    adapter = _adapter_with_parent(forum)

    with caplog.at_level("WARNING"):
        assert await adapter.create_handoff_thread("100", "branch") is None
    assert forum.create_thread.await_count == 1
    assert "forum post create failed" in caplog.text


@pytest.mark.asyncio
async def test_text_channel_path_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    text = SimpleNamespace(
        id=200, type=TEXT_TYPE, create_thread=AsyncMock(return_value=SimpleNamespace(id=999)), send=AsyncMock(),
    )
    adapter = _adapter_with_parent(text)

    assert await adapter.create_handoff_thread("200", "branch") == "999"
    assert "content" not in text.create_thread.await_args.kwargs
    text.send.assert_not_awaited()
