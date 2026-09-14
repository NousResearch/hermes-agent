"""Regression for #110707: the synchronous skill-command scan backing the Telegram command
menu must not execute on the gateway event loop. With ~145 skill commands the scan stalls
the loop, the shutdown watchdog sees missed liveness probes and kills the gateway with
exit 75 (restart mid-session). Both registration sites are covered: post-connect
``_register_command_menu`` and lazy forum ``_ensure_forum_commands``."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_test_adapter():
    """Build a TelegramAdapter without running __init__."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
    # ``name`` is a property derived from platform.value.title()
    adapter._bot = MagicMock()
    adapter._bot.set_my_commands = AsyncMock()
    adapter._forum_command_registered = set()
    adapter._forum_lock = asyncio.Lock()
    return adapter


def _recording_scan(seen):
    """Stand-in for ``telegram_menu_commands`` that records the thread it runs on."""

    def _scan(*args, **kwargs):
        seen.append(threading.get_ident())
        return [], 0

    return _scan


@pytest.mark.asyncio
async def test_register_command_menu_scan_runs_off_event_loop():
    """Post-connect menu registration must not run the skill scan on the loop thread."""
    adapter = _make_test_adapter()
    seen = []
    loop_ident = threading.get_ident()
    with patch(
        "hermes_cli.commands_platforms.telegram_menu_commands",
        new=_recording_scan(seen),
    ):
        with patch("telegram.BotCommand"), patch("telegram.BotCommandScopeDefault"), patch(
            "telegram.BotCommandScopeAllPrivateChats"
        ), patch("telegram.BotCommandScopeAllGroupChats"):
            await adapter._register_command_menu()
    assert seen, "expected the command-menu scan to run"
    assert seen[0] != loop_ident, (
        "command-menu skill scan executed on the event loop thread; "
        "a slow scan blocks liveness probes and trips the shutdown watchdog (#110707)"
    )
    adapter._bot.set_my_commands.assert_awaited()


@pytest.mark.asyncio
async def test_ensure_forum_commands_scan_runs_off_event_loop():
    """Lazy forum registration has the same scan; it must not run on the loop thread either."""
    adapter = _make_test_adapter()
    msg = SimpleNamespace(chat=SimpleNamespace(id=-123, is_forum=True))
    seen = []
    loop_ident = threading.get_ident()
    with patch(
        "hermes_cli.commands_platforms.telegram_menu_commands",
        new=_recording_scan(seen),
    ):
        with patch("telegram.BotCommand"), patch("telegram.BotCommandScopeChat"):
            await adapter._ensure_forum_commands(msg)
    assert -123 in adapter._forum_command_registered
    assert seen, "expected the forum command scan to run"
    assert seen[0] != loop_ident, (
        "forum command skill scan executed on the event loop thread; "
        "a slow scan blocks liveness probes and trips the shutdown watchdog (#110707)"
    )
