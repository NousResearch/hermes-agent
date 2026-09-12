"""Regression test for PR #68253: ignore-no-mention drops must be visible.

``DiscordAdapter._discord_message_admission`` silently dropped messages that
mention a human (but not the bot) in a non-free-response channel under
``DISCORD_IGNORE_NO_MENTION=true`` (the default) with zero log output,
making channel responsiveness impossible to diagnose. The fix adds a DEBUG
diagnostic at the drop site.

This test pins the behavior contract: the message is still rejected
``(False, False)``, and the drop is now observable via caplog at DEBUG level.
"""

import logging
from types import SimpleNamespace

import discord  # Shared mock installed by tests/gateway/conftest.py

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

DROPPED_DIAGNOSTIC = "Ignoring non-mention message in non-free-response channel"


def _make_adapter() -> DiscordAdapter:
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def test_discord_ignore_no_mention_drop_logs_debug_and_rejects(monkeypatch, caplog):
    """A non-free-response channel drop still rejects, and now logs at DEBUG."""
    adapter = _make_adapter()
    # Admission compares against the client's own user; give it a distinct bot.
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=777))

    # Default DISCORD_IGNORE_NO_MENTION=true; no free-response channels.
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    # Let the author through the allowlist so the run reaches the mention gate.
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")

    message = SimpleNamespace(
        id="msg-ignore-no-mention-regression",
        author=SimpleNamespace(id=42, bot=False),
        type=discord.MessageType.default,
        channel=SimpleNamespace(id=999, name="general", parent_id=None),
        guild=SimpleNamespace(id=1, name="test-guild"),
        # Mentions a human, NOT the bot and NOT another bot: this is exactly
        # the path the ignore-no-mention gate drops in non-free channels.
        mentions=[SimpleNamespace(id=43, bot=False)],
        content="<@43> hello there",
    )

    with caplog.at_level(logging.DEBUG):
        admitted, role_authorized = adapter._discord_message_admission(
            message, claim=False,
        )

    # Rejection result is preserved.
    assert admitted is False
    assert role_authorized is False

    # The drop is now observable: one DEBUG record naming the channel gate.
    drop_records = [
        record for record in caplog.records
        if DROPPED_DIAGNOSTIC in record.getMessage()
    ]
    assert len(drop_records) == 1
    assert drop_records[0].levelno == logging.DEBUG
