"""Regression test for Discord typing-indicator 429 handling (#29671).

Before the fix in #29671, ``DiscordAdapter.send_typing``'s background loop
would die on the first exception from ``client.http.request``, including
a Discord 429 rate-limit response. After a single 429, the typing bubble
stayed dark for the rest of the turn even though the platform was happy
to accept retries after ``retry_after`` seconds.

The fix extracts ``retry_after`` (via attribute or response headers),
sleeps that long, and continues the loop. Only a non-rate-limit
exception (no extractable ``retry_after``) still terminates the loop,
matching the pre-fix exit behavior for genuinely unrecoverable errors.

These tests guard both halves: 429 keeps the loop alive across multiple
hits, and a non-429 exception still tears it down.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_discord_mock() -> None:
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is not None:
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object,
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3,
        green=1, grey=2, blurple=2, red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1, green=lambda: 2, blue=lambda: 3,
        red=lambda: 4, purple=lambda: 5,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402

from gateway.config import Platform, PlatformConfig  # noqa: E402


# Capture the real sleep BEFORE any test can monkeypatch the adapter
# module's reference to it. The fake_sleep helpers below need a way to
# yield control back to the event loop without re-entering themselves.
_REAL_SLEEP = asyncio.sleep


def _make_adapter():
    """Build a minimal DiscordAdapter without running __init__.

    Matches the pattern used by ``test_discord_race_polish.py``:
    ``send_typing`` only touches ``_client`` and ``_typing_tasks``,
    so we don't need the full constructor.
    """
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="t")
    adapter._typing_tasks = {}
    adapter._client = MagicMock()
    return adapter


class _RateLimitedExc(Exception):
    """Mimics discord.py's RateLimited shape via the duck-type fallback
    branch in ``_extract_discord_retry_after``: exposes a numeric
    ``retry_after`` attribute, which the helper reads first."""

    def __init__(self, retry_after: float, message: str = "rate limited"):
        super().__init__(message)
        self.retry_after = retry_after


class _HeaderRateLimitedExc(Exception):
    """429 surfaced via a response.headers dict, covering Retry-After
    extracted from headers when the exception itself lacks the attribute."""

    def __init__(self, retry_after_header: str):
        super().__init__("rate limited (header-only)")
        self.response = SimpleNamespace(
            headers={"Retry-After": retry_after_header},
        )


async def _drain(loops: int = 8) -> None:
    """Yield control to the event loop several times so the typing task
    can run, hit our mocked side effects, and call our fake_sleep."""
    for _ in range(loops):
        await _REAL_SLEEP(0)


@pytest.mark.asyncio
async def test_429_with_retry_after_attribute_keeps_loop_alive(monkeypatch):
    """A 429 with a numeric ``retry_after`` attribute must NOT kill the
    typing loop. The loop must sleep that long and try again; multiple
    consecutive 429s should each be honored, not collapsed into a single
    fatal failure."""
    adapter = _make_adapter()

    request_calls = 0

    async def fake_request(_route):
        nonlocal request_calls
        request_calls += 1
        # First three attempts hit 429, then we let stop_typing tear it
        # down. The exact retry_after values are what we expect the loop
        # to forward to asyncio.sleep.
        if request_calls == 1:
            raise _RateLimitedExc(retry_after=2.5)
        if request_calls == 2:
            raise _RateLimitedExc(retry_after=4.0)
        if request_calls == 3:
            raise _RateLimitedExc(retry_after=1.5)
        # Past third attempt: park forever so the loop is alive when we
        # cancel it. asyncio.CancelledError flows through cleanly.
        await _REAL_SLEEP(60)

    adapter._client.http.request = AsyncMock(side_effect=fake_request)

    sleep_durations: list[float] = []

    async def fake_sleep(seconds):
        sleep_durations.append(seconds)
        # Yield so the next loop iteration runs without burning real time.
        await _REAL_SLEEP(0)

    monkeypatch.setattr(discord_platform.asyncio, "sleep", fake_sleep)

    await adapter.send_typing("chan-123")
    await _drain()

    # All three 429s should have been seen, and each should have caused
    # the loop to sleep for the corresponding retry_after, proving the
    # loop did NOT exit on a rate-limit response.
    assert request_calls >= 3, (
        f"loop exited prematurely after only {request_calls} request(s); "
        "expected at least 3 attempts across consecutive 429s"
    )
    assert sleep_durations[:3] == [2.5, 4.0, 1.5], (
        f"retry_after values were not forwarded correctly: {sleep_durations[:3]}"
    )

    await adapter.stop_typing("chan-123")
    assert "chan-123" not in adapter._typing_tasks


@pytest.mark.asyncio
async def test_429_via_response_header_keeps_loop_alive(monkeypatch):
    """Some 429s arrive as exceptions whose ``retry_after`` lives only in
    ``response.headers["Retry-After"]``. The helper falls back to that
    header, and the loop must still continue, not die."""
    adapter = _make_adapter()

    request_calls = 0

    async def fake_request(_route):
        nonlocal request_calls
        request_calls += 1
        if request_calls == 1:
            raise _HeaderRateLimitedExc(retry_after_header="3")
        if request_calls == 2:
            raise _HeaderRateLimitedExc(retry_after_header="7.25")
        await _REAL_SLEEP(60)

    adapter._client.http.request = AsyncMock(side_effect=fake_request)

    sleep_durations: list[float] = []

    async def fake_sleep(seconds):
        sleep_durations.append(seconds)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(discord_platform.asyncio, "sleep", fake_sleep)

    await adapter.send_typing("chan-456")
    await _drain()

    assert request_calls >= 2, (
        f"loop exited after {request_calls} request(s) despite header-based 429"
    )
    assert sleep_durations[:2] == [3.0, 7.25], (
        f"Retry-After header not honored: {sleep_durations[:2]}"
    )

    await adapter.stop_typing("chan-456")
    assert "chan-456" not in adapter._typing_tasks


@pytest.mark.asyncio
async def test_non_rate_limit_exception_still_exits_loop(monkeypatch):
    """The fix only changes behavior for exceptions with an extractable
    ``retry_after``. A plain exception (no rate-limit signal) must still
    terminate the loop, matching the pre-fix behavior. Otherwise we'd
    spin forever on a genuinely unrecoverable error."""
    adapter = _make_adapter()

    request_calls = 0

    async def fake_request(_route):
        nonlocal request_calls
        request_calls += 1
        # Plain exception with no retry_after attr and no response.headers,
        # so the helper returns None and the loop should return.
        raise RuntimeError("connection refused")

    adapter._client.http.request = AsyncMock(side_effect=fake_request)

    async def fake_sleep(seconds):
        # If the loop ever gets here, it means it tried to retry the
        # non-rate-limit exception. That's the regression we're guarding
        # against in the opposite direction. We still yield so we don't
        # deadlock the test, but a later assertion will catch the miss.
        await _REAL_SLEEP(0)

    monkeypatch.setattr(discord_platform.asyncio, "sleep", fake_sleep)

    await adapter.send_typing("chan-789")

    # Wait until the loop self-terminates and clears the entry from
    # _typing_tasks (the finally clause does this). Cap the wait so a
    # regression that loops forever fails fast instead of hanging.
    for _ in range(20):
        if "chan-789" not in adapter._typing_tasks:
            break
        await _REAL_SLEEP(0)

    assert request_calls == 1, (
        f"expected exactly 1 request before exit, got {request_calls}; "
        "loop is retrying non-rate-limit exceptions"
    )
    assert "chan-789" not in adapter._typing_tasks, (
        "loop did not clean up _typing_tasks on non-rate-limit exit"
    )
