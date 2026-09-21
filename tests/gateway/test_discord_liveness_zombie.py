"""Liveness probe must never die silently between strikes (#118487).

The chronic zombie pattern — one ``socket_closed, 1/2`` strike, then the probe
goes permanently silent while the adapter stays ``_running`` — was caused by the
loop's top-of-iteration guards returning without a log line and without any
re-arm. These tests pin the contract that a probe exit is either an intentional
shutdown (clean cancel) or gets logged and re-armed.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_connect import (  # noqa: E402
    _ensure_discord_mock,
)

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

from tests.gateway.test_discord_liveness import (  # noqa: E402
    _LiveBot,
    _connect,
    _make_adapter,
    _set_websocket_health,
    _wait_until,
)


@pytest.mark.asyncio
async def test_liveness_exit_with_stale_client_re_arms_and_logs(monkeypatch, caplog):
    """A probe that wakes up to a torn-down client (``_client is None`` while
    ``_running`` is still true) must log the exit and re-arm instead of going
    zombie — the exact window the incident report captured."""
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=2)
    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)

    def factory(**kwargs):
        bot = _LiveBot(
            intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions")
        )
        bot.fetch_user = AsyncMock()
        return bot

    await _connect(adapter, monkeypatch, factory)
    first_task = adapter._liveness_task
    assert first_task is not None and not first_task.done()

    # Simulate the teardown window: disconnect() cleared the client but the
    # gateway is re-starting the adapter (running stays true).
    client = adapter._client
    adapter._client = None

    with caplog.at_level("WARNING", logger="plugins.platforms.discord.adapter"):
        await asyncio.wait_for(first_task, timeout=1.0)

    assert any(
        "Discord Gateway liveness probe exited" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    # Restore a healthy client so the re-armed probe settles instead of thrashing.
    adapter._client = client
    # The probe must re-arm itself, not leave the adapter unwatched.
    await _wait_until(
        lambda: (
            adapter._liveness_task is not None
            and adapter._liveness_task is not first_task
            and not adapter._liveness_task.done()
        ),
        message="liveness probe did not re-arm after guard exit",
    )
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_liveness_exit_during_disconnect_is_silent(monkeypatch, caplog):
    """The clean-shutdown path (``disconnect()`` sets ``_disconnecting``) keeps
    its existing behaviour: silent exit, no spurious re-arm."""
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=3)

    def factory(**kwargs):
        bot = _LiveBot(
            intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions")
        )
        bot.fetch_user = AsyncMock()
        return bot

    await _connect(adapter, monkeypatch, factory)
    first_task = adapter._liveness_task
    assert first_task is not None and not first_task.done()

    with caplog.at_level("WARNING", logger="plugins.platforms.discord.adapter"):
        adapter._disconnecting = True
        await asyncio.wait_for(first_task, timeout=1.0)
        # Give a potential (buggy) re-arm a moment to surface.
        await asyncio.sleep(0.05)

    assert not [r for r in caplog.records if "liveness probe exited" in r.getMessage()]
    assert adapter._liveness_task is first_task  # no re-arm during teardown
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_socket_closed_escalates_on_first_strike(monkeypatch, caplog):
    """A closed socket is terminal evidence: the probe must not wait for a
    second confirmation before forcing the reconnect (#118487 ask #2)."""
    adapter = _make_adapter(monkeypatch, interval=0.01, threshold=2)
    handler = AsyncMock()
    adapter.set_fatal_error_handler(handler)

    def factory(**kwargs):
        bot = _LiveBot(
            intents=kwargs["intents"], allowed_mentions=kwargs.get("allowed_mentions")
        )
        bot.fetch_user = AsyncMock()
        return bot

    await _connect(adapter, monkeypatch, factory)
    client = adapter._client
    assert client is not None
    _set_websocket_health(client, socket_open=False)

    with caplog.at_level("ERROR", logger="plugins.platforms.discord.adapter"):
        await _wait_until(
            lambda: handler.await_count > 0,
            message="fatal handler not called for a closed socket",
            timeout=3.0,
        )

    assert any(
        "forcing reconnect on first strike" in r.getMessage() for r in caplog.records
    )
    await adapter.disconnect()
