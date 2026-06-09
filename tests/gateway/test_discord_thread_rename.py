"""Tests for Discord thread auto-rename after auto-title.

Mirrors the Telegram topic-rename behaviour: when Hermes auto-titles a
session, the Discord thread is silently renamed to match the generated title.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


# Ensure discord module is mocked in test environment (discord.py is not
# installed during tests).  Must include exception classes so the adapter's
# except discord.Forbidden / discord.HTTPException handlers don't crash.
def _ensure_discord_mock() -> None:
    """Ensure a discord mock is in sys.modules with the exception classes
    the adapter needs.  Must run before any DiscordAdapter import."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    disc = sys.modules.get("discord")
    # If the real discord module is installed, leave it alone.
    if disc is not None and getattr(disc, "__file__", None) is not None:
        return

    # Build a fresh mock.  Keep the minimal set of attributes other
    # test files (test_discord_send.py) depend on so we don't break
    # them when our mock replaces theirs.
    mock = MagicMock()
    mock.Forbidden = type("Forbidden", (Exception,), {})
    mock.HTTPException = type("HTTPException", (Exception,), {})
    mock.Intents = MagicMock()
    mock.Intents.default.return_value = MagicMock()
    mock.Client = MagicMock
    mock.File = MagicMock
    mock.DMChannel = type("DMChannel", (), {})
    mock.Thread = type("Thread", (), {})
    mock.ForumChannel = type("ForumChannel", (), {})
    mock.Embed = MagicMock
    mock.ui = SimpleNamespace(
        View=object,
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    mock.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3,
        green=1, grey=2, blurple=2, red=3,
    )
    mock.Color = SimpleNamespace(
        orange=lambda: 1, green=lambda: 2, blue=lambda: 3,
        red=lambda: 4, purple=lambda: 5,
    )
    mock.Interaction = object
    mock.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules["discord"] = mock
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()


# ── Discord adapter rename_thread tests ────────────────────────────────

def _make_discord_adapter() -> Any:
    """Return a DiscordAdapter with a mocked _client."""
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_rename_thread_success() -> None:
    """rename_thread fetches the channel and calls edit(name=...)."""
    adapter = _make_discord_adapter()
    client = adapter._client
    assert client is not None
    mock_thread = MagicMock()
    mock_thread.edit = AsyncMock()
    client.get_channel.return_value = mock_thread

    result = await adapter.rename_thread("12345", "My Cool Thread")

    assert result is True
    client.get_channel.assert_called_once_with(12345)
    mock_thread.edit.assert_awaited_once_with(name="My Cool Thread")


@pytest.mark.asyncio
async def test_rename_thread_fetches_when_get_channel_returns_none() -> None:
    """When get_channel returns None, fall back to fetch_channel."""
    adapter = _make_discord_adapter()
    client = adapter._client
    assert client is not None
    mock_thread = MagicMock()
    mock_thread.edit = AsyncMock()
    client.get_channel.return_value = None
    client.fetch_channel = AsyncMock(return_value=mock_thread)

    result = await adapter.rename_thread("12345", "Fetched Thread")

    assert result is True
    client.fetch_channel.assert_awaited_once_with(12345)
    mock_thread.edit.assert_awaited_once_with(name="Fetched Thread")


@pytest.mark.asyncio
async def test_rename_thread_not_found() -> None:
    """Returns False when the channel cannot be resolved."""
    adapter = _make_discord_adapter()
    client = adapter._client
    assert client is not None
    client.get_channel.return_value = None
    client.fetch_channel = AsyncMock(return_value=None)

    result = await adapter.rename_thread("12345", "Ghost Thread")

    assert result is False


@pytest.mark.asyncio
async def test_rename_thread_client_not_connected() -> None:
    """Returns False when self._client is None."""
    adapter = _make_discord_adapter()
    adapter._client = None

    result = await adapter.rename_thread("12345", "No Client")

    assert result is False


@pytest.mark.asyncio
async def test_rename_thread_empty_name() -> None:
    """Returns False when thread_id or name is empty."""
    adapter = _make_discord_adapter()

    assert await adapter.rename_thread("", "Name") is False
    assert await adapter.rename_thread("123", "") is False
    assert await adapter.rename_thread("123", "   ") is False


@pytest.mark.asyncio
async def test_rename_thread_truncates_long_names() -> None:
    """Names longer than 100 chars are truncated with ellipsis."""
    adapter = _make_discord_adapter()
    client = adapter._client
    assert client is not None
    mock_thread = MagicMock()
    mock_thread.edit = AsyncMock()
    client.get_channel.return_value = mock_thread

    long_name = "A" * 120
    result = await adapter.rename_thread("12345", long_name)

    assert result is True
    # Should be ≤100 chars and end with "..."
    called_name = mock_thread.edit.call_args.kwargs["name"]
    assert len(called_name) <= 100
    assert called_name.endswith("...")


@pytest.mark.asyncio
async def test_rename_thread_survives_edit_error() -> None:
    """Returns False when thread.edit raises."""
    adapter = _make_discord_adapter()
    client = adapter._client
    assert client is not None
    mock_thread = MagicMock()
    mock_thread.edit = AsyncMock(side_effect=RuntimeError("no perms"))
    client.get_channel.return_value = mock_thread

    result = await adapter.rename_thread("12345", "Forbidden")

    assert result is False


# ── GatewayRunner Discord rename tests ──────────────────────────────────

def _make_discord_source(*, thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="12345",
        chat_id="987654",
        user_name="tester",
        chat_type="text",
        thread_id=thread_id,
    )


def _make_runner() -> Any:
    """Minimal GatewayRunner for Discord rename tests."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    return runner


# ── _is_discord_thread_lane ────────────────────────────────────────────

def test_is_discord_thread_lane_true() -> None:
    """Discord source with thread_id returns True."""
    runner = _make_runner()
    source = _make_discord_source(thread_id="42")
    assert runner._is_discord_thread_lane(source) is True


def test_is_discord_thread_lane_wrong_platform() -> None:
    """Telegram source must not match."""
    runner = _make_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="12345",
        chat_id="987654",
        user_name="tester",
        chat_type="dm",
        thread_id="42",
    )
    assert runner._is_discord_thread_lane(source) is False


def test_is_discord_thread_lane_no_thread_id() -> None:
    """Discord source without thread_id returns False."""
    runner = _make_runner()
    source = _make_discord_source(thread_id=None)
    assert runner._is_discord_thread_lane(source) is False


def test_is_discord_thread_lane_empty_thread_id() -> None:
    """Discord source with empty-string thread_id returns False."""
    runner = _make_runner()
    source = _make_discord_source(thread_id="")
    assert runner._is_discord_thread_lane(source) is False


# ── _sanitize_discord_thread_title ─────────────────────────────────────

def test_sanitize_normal_title() -> None:
    runner = _make_runner()
    result = runner._sanitize_discord_thread_title("  Build   Discord   UX  ")
    assert result == "Build Discord UX"


def test_sanitize_long_title_truncated() -> None:
    runner = _make_runner()
    long_title = "A" * 120
    result = runner._sanitize_discord_thread_title(long_title)
    assert len(result) <= 100
    assert result.endswith("...")


def test_sanitize_empty_title_fallback() -> None:
    runner = _make_runner()
    assert runner._sanitize_discord_thread_title("") == "Hermes Chat"
    assert runner._sanitize_discord_thread_title("   ") == "Hermes Chat"


def test_sanitize_exactly_100_chars() -> None:
    runner = _make_runner()
    title = "X" * 100
    result = runner._sanitize_discord_thread_title(title)
    assert result == title
    assert len(result) == 100


# ── _rename_discord_thread_for_session_title ───────────────────────────

@pytest.mark.asyncio
async def test_rename_calls_adapter_rename_thread() -> None:
    runner = _make_runner()
    adapter = MagicMock()
    adapter.rename_thread = AsyncMock(return_value=True)
    runner.adapters = {Platform.DISCORD: adapter}

    source = _make_discord_source(thread_id="42")
    await runner._rename_discord_thread_for_session_title(
        source, "sess-1", "  My Chat Title  "
    )

    adapter.rename_thread.assert_awaited_once_with(
        thread_id="42",
        name="My Chat Title",
    )


@pytest.mark.asyncio
async def test_rename_no_adapter_no_crash() -> None:
    runner = _make_runner()
    runner.adapters = {}
    source = _make_discord_source(thread_id="42")
    # Must not raise
    await runner._rename_discord_thread_for_session_title(
        source, "sess-1", "Title"
    )


@pytest.mark.asyncio
async def test_rename_adapter_without_rename_thread() -> None:
    runner = _make_runner()
    adapter = MagicMock(spec=[])  # no rename_thread attr
    runner.adapters = {Platform.DISCORD: adapter}
    source = _make_discord_source(thread_id="42")
    # Must not raise
    await runner._rename_discord_thread_for_session_title(
        source, "sess-1", "Title"
    )


# ── _schedule_discord_thread_title_rename ──────────────────────────────

@pytest.mark.asyncio
async def test_schedule_skips_wrong_platform() -> None:
    """Telegram source must not trigger Discord rename scheduling."""
    runner = _make_runner()
    tele_source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="12345",
        chat_id="987654",
        user_name="tester",
        chat_type="dm",
        thread_id="42",
    )
    # Must not raise, even without a running event loop
    runner._schedule_discord_thread_title_rename(
        tele_source,
        "sess-1",
        "Should Be Ignored",
    )


def test_schedule_skips_no_title() -> None:
    runner = _make_runner()
    source = _make_discord_source(thread_id="42")
    # Must not raise — should short-circuit on empty title
    runner._schedule_discord_thread_title_rename(source, "sess-1", "")


def test_schedule_skips_no_thread_id() -> None:
    runner = _make_runner()
    source = _make_discord_source(thread_id=None)
    runner._schedule_discord_thread_title_rename(source, "sess-1", "Title")


@pytest.mark.asyncio
async def test_rename_thread_normalizes_whitespace() -> None:
    """Adapter's rename_thread collapses multiple spaces like the gateway sanitizer."""
    adapter = _make_discord_adapter()
    client = adapter._client
    assert client is not None
    mock_thread = MagicMock()
    mock_thread.edit = AsyncMock()
    client.get_channel.return_value = mock_thread

    await adapter.rename_thread("12345", "  Build   Discord   UX  ")

    mock_thread.edit.assert_awaited_once_with(name="Build Discord UX")


@pytest.mark.asyncio
async def test_schedule_dispatches_to_adapter() -> None:
    """The scheduling happy path: _schedule_discord_thread_title_rename
    creates a coroutine and passes it to safe_schedule_threadsafe."""
    import asyncio
    from unittest.mock import patch

    runner = _make_runner()
    adapter = MagicMock()
    adapter.rename_thread = AsyncMock()
    runner.adapters = {Platform.DISCORD: adapter}
    source = _make_discord_source(thread_id="42")

    # bypass early returns: we need a running loop
    loop = asyncio.get_running_loop()

    with patch("gateway.run.safe_schedule_threadsafe") as mock_schedule:
        mock_schedule.return_value = asyncio.Future()
        mock_schedule.return_value.set_result(None)

        runner._schedule_discord_thread_title_rename(source, "sess-1", "Healthy Title")

        # safe_schedule_threadsafe should have been called once
        mock_schedule.assert_called_once()
        # The first positional arg is the coroutine
        coro = mock_schedule.call_args[0][0]
        # Drive the coroutine ourselves inside the test loop
        await coro
        # Now the adapter's rename_thread should have been called
        adapter.rename_thread.assert_awaited_once_with(
            thread_id="42",
            name="Healthy Title",
        )
