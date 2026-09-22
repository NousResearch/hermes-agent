"""Sibling coverage for expired slash defer handling: /thread creation must
still run when the interaction token has expired (Discord error 10062), just
skipping the ephemeral followups."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.helpers import ThreadParticipationTracker
from plugins.platforms.discord.adapter import DiscordAdapter


class _UnknownInteraction(Exception):
    status = 404
    code = 10062


def _adapter():
    a = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    a._check_slash_authorization = AsyncMock(return_value=True)
    return a


@pytest.mark.asyncio
async def test_thread_create_slash_survives_expired_defer(tmp_path):
    adapter = _adapter()
    interaction = SimpleNamespace(
        response=SimpleNamespace(defer=AsyncMock(side_effect=_UnknownInteraction("Unknown interaction"))),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    adapter._create_thread = AsyncMock(
        return_value={"success": True, "thread_id": "999", "thread_name": "t"}
    )
    # The REAL tracker, pointed at a temp file, instead of a hand-rolled stub.
    # A stub that reimplements the tracker's surface silently rots when that
    # surface changes (it used to be `mark`; the off-loop fix made it
    # `mark_async`), and the test then fails for a reason that has nothing to
    # do with expired defers.
    state = tmp_path / "discord_threads.json"
    tracker = ThreadParticipationTracker.__new__(ThreadParticipationTracker)
    tracker._state_path = lambda: state
    ThreadParticipationTracker.__init__(tracker, "discord")
    tracker._state_path = lambda: state
    adapter._threads = tracker

    await adapter._handle_thread_create_slash(interaction, name="t")

    adapter._create_thread.assert_awaited_once()
    interaction.followup.send.assert_not_awaited()
    # The thread really was recorded: the off-loop mark is awaited, not dropped.
    assert "999" in adapter._threads
