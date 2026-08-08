"""Regression tests: Discord runtime status must track gateway reconnects.

``connect()`` returns once the *first* ``on_ready`` fires.  After that,
discord.py owns the socket and reconnects internally — the adapter's
``connect()`` is never re-entered.  If the adapter only records "connected"
at ``connect()`` time, the runtime status file (and therefore the dashboard's
Discord badge) freezes at whatever the first connect wrote and never reflects
a later drop or resume.

These tests pin two things at once:

1. the three gateway lifecycle events publish the right *runtime status
   writes*, and
2. publishing them does **not** disturb adapter lifecycle state.

Point 2 is not decoration.  ``on_disconnect`` fires on every routine internal
reconnect, so routing it through ``_mark_disconnected()`` (which sets
``_running = False``) would permanently kill the WebSocket liveness watchdog —
``_start_liveness_probe()`` is only reachable from ``connect()``, which
discord.py never re-enters — and disarm the split-brain guard in
``_handle_bot_task_done``.  Symmetrically, ``_mark_connected()`` clears
``_fatal_error_*``, so a late READY racing ``_notify_liveness_fatal_error``
would erase the fatal the watchdog just raised.  The fixture therefore runs the
*real* liveness probe rather than stubbing it out, because a stub cannot
observe either regression.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_connect import (  # noqa: E402
    FakeBot,
    _ensure_discord_mock,
)

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _LiveBot(FakeBot):
    """A FakeBot whose ``start()`` stays pending like a real discord.py client.

    ``FakeBot.start()`` returns immediately, which lets the adapter's bot-task
    done-callback fire and set a spurious ``discord_gateway_task_exited`` fatal
    error — that would mask the very status writes these tests assert on.
    A real client keeps ``start()`` running for the life of the connection.
    """

    def __init__(self, *, intents, proxy=None, allowed_mentions=None, **_):
        super().__init__(intents=intents, allowed_mentions=allowed_mentions)
        self._never = asyncio.Event()
        self._closed = False

    async def start(self, token):
        if "on_ready" in self._events:
            await self._events["on_ready"]()
        await self._never.wait()

    def is_closed(self):
        return self._closed

    def is_ready(self):
        return True

    async def close(self):
        self._closed = True
        self._never.set()


@pytest.fixture
def status_writes(monkeypatch):
    """Capture every ``write_runtime_status`` call the adapter makes.

    ``_write_runtime_status_safe`` imports the function from ``gateway.status``
    at call time, so patching the module attribute intercepts the real path
    the production code takes.
    """
    import gateway.status as gateway_status

    calls: list[dict] = []

    def _capture(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(gateway_status, "write_runtime_status", _capture)
    return calls


@pytest.fixture
def connected_adapter(monkeypatch):
    """A DiscordAdapter that has completed one ``connect()`` against a live bot.

    The liveness probe is intentionally NOT stubbed: it is the component the
    ``on_disconnect`` handler must not kill, so the tests need the real task.
    It is given a short interval and a health sampler that counts calls, so a
    test can observe whether the watchdog is still sampling.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

    intents = SimpleNamespace(
        message_content=False,
        dm_messages=False,
        guild_messages=False,
        members=False,
        voice_states=False,
    )
    monkeypatch.setattr(discord_platform.Intents, "default", lambda: intents)

    created: dict = {"health_samples": 0}

    def fake_bot_factory(*, command_prefix, intents, proxy=None, allowed_mentions=None, **_):
        created["bot"] = _LiveBot(intents=intents, allowed_mentions=allowed_mentions)
        return created["bot"]

    monkeypatch.setattr(discord_platform.commands, "Bot", fake_bot_factory)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())

    # Real watchdog, fast clock, always-healthy sampler: sample count is the
    # observable "is the watchdog still alive?" signal.
    adapter._liveness_interval_seconds = 0.01
    adapter._liveness_failure_threshold = 3

    def _health(client):
        created["health_samples"] += 1
        return True, "healthy"

    monkeypatch.setattr(adapter, "_read_websocket_health", _health)

    yield adapter, created

    task = adapter._liveness_task
    if task is not None and not task.done():
        task.cancel()


def _platform_states(calls: list[dict]) -> list:
    return [c.get("platform_state") for c in calls]


async def _watchdog_is_sampling(created: dict) -> bool:
    """True when the liveness watchdog produced at least one new sample."""
    before = created["health_samples"]
    await asyncio.sleep(0.05)
    return created["health_samples"] > before


@pytest.mark.asyncio
async def test_on_ready_marks_platform_connected(connected_adapter, status_writes):
    """A gateway READY must publish a ``connected`` runtime status.

    discord.py re-fires ``on_ready`` on every internal reconnect, long after
    ``connect()`` returned, so this is the only hook that can refresh the
    dashboard's Discord state on a resumed session.
    """
    adapter, created = connected_adapter

    assert await adapter.connect() is True
    bot = created["bot"]

    status_writes.clear()
    await bot._events["on_ready"]()

    assert "connected" in _platform_states(status_writes), (
        "on_ready must mark the platform connected so a post-connect() "
        f"reconnect refreshes runtime status; saw writes={status_writes}"
    )
    assert adapter.is_connected is True


@pytest.mark.asyncio
async def test_on_resumed_marks_platform_connected(connected_adapter, status_writes):
    """A RESUMED session must publish a ``connected`` runtime status.

    ``on_resumed`` (not ``on_ready``) is what discord.py fires when it replays
    a session after a transient socket drop, so without this handler the
    dashboard stays stuck on the stale ``disconnected`` badge.
    """
    adapter, created = connected_adapter

    assert await adapter.connect() is True
    bot = created["bot"]

    assert "on_resumed" in bot._events, (
        "connect() must register an on_resumed handler; registered events: "
        f"{sorted(bot._events)}"
    )

    status_writes.clear()
    await bot._events["on_resumed"]()

    assert "connected" in _platform_states(status_writes), (
        f"on_resumed must mark the platform connected; saw writes={status_writes}"
    )
    assert adapter.is_connected is True


@pytest.mark.asyncio
async def test_on_disconnect_marks_platform_disconnected(connected_adapter, status_writes):
    """A gateway socket drop must publish a ``disconnected`` runtime status."""
    adapter, created = connected_adapter

    assert await adapter.connect() is True
    bot = created["bot"]

    assert "on_disconnect" in bot._events, (
        "connect() must register an on_disconnect handler; registered events: "
        f"{sorted(bot._events)}"
    )

    status_writes.clear()
    await bot._events["on_disconnect"]()

    assert "disconnected" in _platform_states(status_writes), (
        f"on_disconnect must mark the platform disconnected; saw writes={status_writes}"
    )


@pytest.mark.asyncio
async def test_on_disconnect_keeps_the_liveness_watchdog_alive(
    connected_adapter, status_writes
):
    """A routine reconnect must not kill the WebSocket health watchdog.

    discord.py dispatches ``on_disconnect`` on every internal reconnect
    (``ReconnectWebSocket`` session resumes and any transient ``OSError`` /
    ``ConnectionClosed``).  ``_liveness_loop`` returns as soon as ``_running``
    is False and nothing restarts it — ``_start_liveness_probe()`` is only
    reached from ``connect()``, which discord.py never re-enters.  So a handler
    that flips ``_running`` would silently disable the watchdog for the whole
    life of the adapter after the first blip.
    """
    adapter, created = connected_adapter

    assert await adapter.connect() is True
    bot = created["bot"]

    assert adapter._liveness_task is not None, "connect() must start the watchdog"
    assert await _watchdog_is_sampling(created), "watchdog is not sampling before the drop"

    await bot._events["on_disconnect"]()

    assert adapter._running is True, (
        "a routine transport drop must not clear the adapter's _running flag; "
        "that would end _liveness_loop and disarm _handle_bot_task_done"
    )
    assert adapter._liveness_task is not None and not adapter._liveness_task.done(), (
        "the liveness watchdog task must survive a transient on_disconnect"
    )
    assert await _watchdog_is_sampling(created), (
        "the liveness watchdog stopped sampling after a transient on_disconnect"
    )


@pytest.mark.asyncio
async def test_on_disconnect_keeps_split_brain_detection_armed(connected_adapter):
    """A gateway-task death after a transient drop must still be surfaced.

    ``_handle_bot_task_done`` short-circuits on ``not self._running`` (startup
    failures are owned by ``_wait_for_ready_or_bot_exit``).  If a routine
    ``on_disconnect`` clears ``_running``, a genuine post-startup websocket
    death is swallowed: no fatal error, so the runner never queues a reconnect.
    """
    adapter, created = connected_adapter

    assert await adapter.connect() is True
    bot = created["bot"]

    await bot._events["on_disconnect"]()

    fatals: list[str] = []
    adapter._set_fatal_error = lambda code, msg, *, retryable: fatals.append(code)

    dead = asyncio.get_running_loop().create_future()
    dead.set_exception(RuntimeError("gateway websocket died"))
    adapter._bot_task = dead
    adapter._handle_bot_task_done(dead)

    assert fatals == ["discord_gateway_task_exited"], (
        "split-brain detection must stay armed across a transient drop; "
        f"got fatal errors {fatals!r}"
    )


@pytest.mark.asyncio
async def test_disconnect_after_fatal_error_keeps_fatal_state(
    connected_adapter, status_writes
):
    """A drop that follows a fatal error must not overwrite the fatal badge.

    The fatal code is the more specific diagnosis; a generic ``disconnected``
    write would mask the real cause on the dashboard.
    """
    adapter, created = connected_adapter

    assert await adapter.connect() is True
    bot = created["bot"]

    adapter._set_fatal_error("boom", "synthetic fatal", retryable=False)

    status_writes.clear()
    await bot._events["on_disconnect"]()

    assert "disconnected" not in _platform_states(status_writes), (
        "a fatal error must survive a subsequent on_disconnect; "
        f"saw writes={status_writes}"
    )
    assert adapter.fatal_error_code == "boom", (
        "on_disconnect must not clear the recorded fatal error code"
    )


@pytest.mark.asyncio
async def test_late_ready_does_not_clear_a_live_fatal_error(
    connected_adapter, status_writes
):
    """A READY racing the watchdog's teardown must not erase the fatal error.

    ``_liveness_loop`` raises ``discord_websocket_health_stale`` and hands off
    to ``_notify_liveness_fatal_error``, which awaits ``client.close()`` under a
    1s timeout.  The client is still live in that window and can dispatch one
    more READY/RESUMED.  Routing that through ``_mark_connected()`` would reset
    ``_fatal_error_code``/``_fatal_error_retryable`` and re-set ``_running``,
    so ``gateway/run.py``'s ``has_fatal_error and not retryable`` gates and its
    error logging would read the wrong state for an adapter being torn down.
    """
    adapter, created = connected_adapter

    assert await adapter.connect() is True
    bot = created["bot"]

    adapter._set_fatal_error(
        "discord_websocket_health_stale", "health check failed", retryable=True
    )
    assert adapter.has_fatal_error is True

    status_writes.clear()
    await bot._events["on_ready"]()
    await bot._events["on_resumed"]()

    assert adapter.has_fatal_error is True, (
        "a late READY/RESUMED must not clear the fatal error the watchdog raised"
    )
    assert adapter.fatal_error_code == "discord_websocket_health_stale"
    assert "connected" not in _platform_states(status_writes), (
        "a live fatal error is the more specific badge; a late READY must not "
        f"overwrite it with 'connected'. saw writes={status_writes}"
    )
