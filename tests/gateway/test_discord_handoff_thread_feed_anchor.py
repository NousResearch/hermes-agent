"""Discord handoff threads anchor on a seed message so they render in the parent feed.

A message-less ``create_thread`` only surfaces in the channel's thread-list panel;
Discord inlines a thread in the feed only when it has a starter message (#131690).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is None:
        discord_mod = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        discord_mod.Intents.default.return_value = __import__(
            "unittest.mock", fromlist=["MagicMock"]
        ).MagicMock()
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.Interaction = object
        sys.modules["discord"] = discord_mod


_ensure_discord_mock()

import discord  # noqa: E402 — mock or real
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from agent.i18n import t  # noqa: E402


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    return adapter


def _parent_with(seed_ok=True, direct_ok=True):
    """A text-channel-like parent whose seed path and direct create can each be made to fail."""
    thread = SimpleNamespace(id=555, name="Hermes — Nightly brief")
    seed_message = SimpleNamespace(id=777, create_thread=AsyncMock(return_value=thread))
    parent = SimpleNamespace(
        send=AsyncMock(return_value=seed_message) if seed_ok else AsyncMock(side_effect=RuntimeError("send rejected")),
        create_thread=(
            AsyncMock(return_value=thread) if direct_ok
            else AsyncMock(side_effect=RuntimeError("create rejected"))
        ),
    )
    return parent, seed_message, thread


@pytest.mark.asyncio
async def test_handoff_thread_is_anchored_on_a_seed_message(adapter):
    parent, seed_message, thread = _parent_with()
    adapter._client.get_channel = lambda _id: parent

    result = await adapter.create_handoff_thread("123", "Hermes — Nightly brief")

    assert result == "555"
    parent.send.assert_awaited_once_with(t("platform.discord.thread.handoff_seed", name="Hermes — Nightly brief"))
    seed_message.create_thread.assert_awaited_once_with(
        name="Hermes — Nightly brief", auto_archive_duration=1440, reason="Hermes session handoff",
    )
    parent.create_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_handoff_thread_falls_back_to_messageless_public_thread(adapter):
    parent, _seed_message, thread = _parent_with(seed_ok=False)
    adapter._client.get_channel = lambda _id: parent

    result = await adapter.create_handoff_thread("123", "Hermes — Nightly brief")

    assert result == "555"
    parent.create_thread.assert_awaited_once_with(
        name="Hermes — Nightly brief",
        auto_archive_duration=1440,
        reason="Hermes session handoff",
        type=discord.ChannelType.public_thread,
    )


@pytest.mark.asyncio
async def test_handoff_thread_returns_none_when_both_paths_fail(adapter):
    parent, _seed_message, _thread = _parent_with(seed_ok=False, direct_ok=False)
    adapter._client.get_channel = lambda _id: parent

    assert await adapter.create_handoff_thread("123", "Hermes — Nightly brief") is None
