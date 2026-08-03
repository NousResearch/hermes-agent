"""Sibling coverage for expired slash defer handling: /thread creation must
still run when the interaction token has expired (Discord error 10062), just
skipping the ephemeral followups."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


class _UnknownInteraction(Exception):
    status = 404
    code = 10062


def _adapter():
    a = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    a._check_slash_authorization = AsyncMock(return_value=True)
    return a


@pytest.mark.asyncio
async def test_thread_create_slash_survives_expired_defer():
    adapter = _adapter()
    interaction = SimpleNamespace(
        response=SimpleNamespace(defer=AsyncMock(side_effect=_UnknownInteraction("Unknown interaction"))),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    adapter._create_thread = AsyncMock(
        return_value={"success": True, "thread_id": "999", "thread_name": "t"}
    )
    adapter._threads = SimpleNamespace(mark=lambda _tid: None)

    await adapter._handle_thread_create_slash(interaction, name="t")

    adapter._create_thread.assert_awaited_once()
    interaction.followup.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_thread_create_slash_normal_defer_still_follows_up():
    adapter = _adapter()
    interaction = SimpleNamespace(
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    adapter._create_thread = AsyncMock(
        return_value={"success": True, "thread_id": "999", "thread_name": "t"}
    )
    adapter._threads = SimpleNamespace(mark=lambda _tid: None)

    await adapter._handle_thread_create_slash(interaction, name="t")

    interaction.followup.send.assert_awaited()


@pytest.mark.asyncio
async def test_thread_create_slash_reraises_non_expiry_errors():
    adapter = _adapter()
    interaction = SimpleNamespace(
        response=SimpleNamespace(defer=AsyncMock(side_effect=RuntimeError("boom"))),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    adapter._create_thread = AsyncMock()

    with pytest.raises(RuntimeError):
        await adapter._handle_thread_create_slash(interaction, name="t")

    adapter._create_thread.assert_not_awaited()
