"""The gmail-triage callback must resolve scripts under HERMES_HOME, not Path.home()/.hermes.

Without this, running under a non-default profile (where HERMES_HOME points to
~/.hermes/profiles/<name>) resolves scripts from the default profile's directory,
silently using the wrong scripts — or none at all.
"""

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_fake_telegram():
    """Minimal stubs so the adapter can be imported without python-telegram-bot."""
    mod = types.ModuleType("telegram")
    mod.Bot = MagicMock
    mod.Update = MagicMock
    mod.InlineKeyboardButton = MagicMock
    mod.InlineKeyboardMarkup = MagicMock
    ext = types.ModuleType("telegram.ext")
    ext.Application = MagicMock
    ext.ApplicationBuilder = MagicMock
    ext.CommandHandler = MagicMock
    ext.MessageHandler = MagicMock
    ext.CallbackQueryHandler = MagicMock
    ext.filters = MagicMock()
    return {
        "telegram": mod,
        "telegram.ext": ext,
        "telegram.ext.filters": ext.filters,
    }


def _make_adapter():
    with patch.dict("sys.modules", _make_fake_telegram()):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = object.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter.config = PlatformConfig(enabled=True, token="fake-token")
        adapter._bot = AsyncMock()
        adapter._name = "telegram-test"
        return adapter


@pytest.mark.asyncio
async def test_gmail_triage_resolves_under_hermes_home(monkeypatch, tmp_path):
    """Script path must follow HERMES_HOME, not hardcoded ~/.hermes.

    When running under a non-default profile, HERMES_HOME points to
    ~/.hermes/profiles/<name>.  The callback must resolve scripts there.
    """
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    scripts_dir = profile_home / "scripts" / "gmail-triage"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / "archive.sh"
    script.write_text("#!/bin/sh\necho ok\n")
    script.chmod(0o755)

    adapter = _make_adapter()

    query = AsyncMock()
    query.from_user = MagicMock()
    query.from_user.id = 42

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_gmail_triage_callback(
            query,
            data="gt:archive:msg123",
            query_chat_id="1",
            query_chat_type="private",
            query_thread_id=None,
            query_user_name="tester",
        )

    calls = [str(c) for c in query.answer.call_args_list]
    assert not any("missing" in c.lower() for c in calls), (
        f"Script was reported missing — path did not resolve under HERMES_HOME: {calls}"
    )


@pytest.mark.asyncio
async def test_gmail_triage_missing_script_in_profile(monkeypatch, tmp_path):
    """When the script dir doesn't exist under HERMES_HOME, report missing."""
    profile_home = tmp_path / "empty_profile"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    adapter = _make_adapter()
    query = AsyncMock()
    query.from_user = MagicMock()
    query.from_user.id = 42

    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_gmail_triage_callback(
            query,
            data="gt:archive:msg123",
            query_chat_id="1",
            query_chat_type="private",
            query_thread_id=None,
            query_user_name="tester",
        )

    calls = [str(c) for c in query.answer.call_args_list]
    assert any("missing" in c.lower() for c in calls), (
        f"Expected 'missing' error when script doesn't exist: {calls}"
    )
