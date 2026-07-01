"""Phase 2 (drain-window recovery): write-path wiring tests.

Proves the two activity-mark sites the reconnect backfill depends on:
- INBOUND: the top of _handle_message marks a channel active BEFORE the
  message is dispatched to the gateway — so a message arriving mid-drain
  (which the gateway then drops) still leaves its channel in the map (D-8).
- OUTBOUND: a conversational send() marks its channel active — so bot-post
  surfaces like #logs/#alerts are recoverable even when the drain-window
  message arrives after the websocket closed (the observed incident).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import sys
import os

# Reuse the e2e conftest doubles for building a wired Discord adapter.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2e"))


@pytest.fixture()
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _build_adapter(_tmp_home):
    """A Discord adapter with a fresh (tmp-home) restart-recovery state."""
    from tests.e2e.conftest import _make_discord_adapter_wired, make_fake_bot_user
    adapter, runner = _make_discord_adapter_wired()
    # Rebuild recovery state now that HERMES_HOME points at the tmp dir.
    from plugins.platforms.discord.adapter import _DiscordRestartRecoveryState
    adapter._restart_recovery = _DiscordRestartRecoveryState(persist_interval_s=0.0)
    return adapter, runner


@pytest.mark.asyncio
async def test_inbound_marks_channel_active_even_during_drain(_tmp_home):
    """D-8: an inbound message marks its channel active BEFORE dispatch, so
    even when the gateway drops it mid-drain the channel is recoverable."""
    from tests.e2e.conftest import (
        make_discord_message,
        make_fake_text_channel,
        make_fake_bot_user,
    )
    adapter, runner = _build_adapter(_tmp_home)

    bot_user = make_fake_bot_user()
    adapter._client = SimpleNamespace(user=bot_user, get_channel=lambda _id: None,
                                      fetch_channel=AsyncMock())

    # Simulate the gateway DROPPING the message (as it does during drain):
    # the handler raises / no-ops. The mark must already have happened.
    async def _dropping_handler(event):
        raise RuntimeError("gateway draining — message dropped")

    adapter.set_message_handler(_dropping_handler)
    adapter.send = AsyncMock()

    channel = make_fake_text_channel(channel_id=987654, name="cold-channel")
    # Mention the bot so the require_mention gate passes and we reach the mark.
    msg = make_discord_message(
        content=f"<@{bot_user.id}> anyone there?",
        channel=channel,
        mentions=[bot_user],
    )

    # Disable auto-threading so the handler path is simple (mark is before it
    # anyway, but this keeps the test focused).
    with patch.dict(os.environ, {"DISCORD_AUTO_THREAD": "false"}):
        # The dispatch will raise (drain drop); the mark must survive it.
        try:
            await adapter._handle_message(msg)
        except Exception:
            pass

    recent = adapter._restart_recovery.recent_channels(lookback_s=10_000)
    assert "987654" in recent, "inbound channel must be marked active before the drop"


# NOTE: the rigorous "mark happens BEFORE the gateway drop" ORDERING proof —
# with the mutation gate "move the mark below the drain-drop → 0 recovered (RED)"
# — is delivered by Phase 5 Variant B (test_restart_backfill_e2e.py), which
# builds the full `_draining` end-to-end double that actually reaches the
# gateway-dispatch boundary. A unit double here cannot reach `await
# self.handle_message()` (it returns earlier in message/attachment processing),
# so proving ordering at this layer would be weaker than the E2E gate. Phase 2
# proves only that the marks FIRE on the correct triggers (inbound / conversational
# outbound / not-on-banner); Phase 5 proves the ordering and the recovery.


@pytest.mark.asyncio
async def test_outbound_marks_bot_post_channel_active(_tmp_home):
    """The #logs/#alerts incident regression: a conversational outbound send
    marks the channel active, so a later drain-window inbound (after the socket
    closes) is recoverable via that prior outbound activity."""
    adapter, runner = _build_adapter(_tmp_home)

    sent_channel = SimpleNamespace(
        id=1480528231286181948,  # #alerts-like id
        send=AsyncMock(return_value=SimpleNamespace(id=42, channel=None)),
    )
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=99999, name="bot", bot=True),
        get_channel=lambda _id: sent_channel,
        fetch_channel=AsyncMock(return_value=sent_channel),
    )

    from plugins.platforms.discord.adapter import DiscordAdapter
    result = await DiscordAdapter.send(
        adapter,
        chat_id="1480528231286181948",
        content="✅ nightly backup complete",
    )

    assert result.success
    recent = adapter._restart_recovery.recent_channels(lookback_s=10_000)
    assert "1480528231286181948" in recent, \
        "conversational outbound must mark the bot-post channel active"


@pytest.mark.asyncio
async def test_outbound_banner_does_not_mark(_tmp_home):
    """A non-conversational banner/status send must NOT mark the channel — only
    genuine conversational sends count as activity."""
    adapter, runner = _build_adapter(_tmp_home)
    sent_channel = SimpleNamespace(
        id=1480525090331561984,  # #logs-like id
        send=AsyncMock(return_value=SimpleNamespace(id=43, channel=None)),
    )
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=99999, name="bot", bot=True),
        get_channel=lambda _id: sent_channel,
        fetch_channel=AsyncMock(return_value=sent_channel),
    )
    from plugins.platforms.discord.adapter import DiscordAdapter
    result = await DiscordAdapter.send(
        adapter,
        chat_id="1480525090331561984",
        content="status ping",
        metadata={"non_conversational": True},
    )
    assert result.success
    recent = adapter._restart_recovery.recent_channels(lookback_s=10_000)
    assert "1480525090331561984" not in recent, \
        "non-conversational banner sends must not mark the channel active"
