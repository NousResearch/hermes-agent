"""Regression tests for profile-aware Telegram gmail-triage callbacks."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class _FakeProc:
    returncode = 0

    async def communicate(self):
        return b"", b""


class TestTelegramGmailTriageProfile:
    @pytest.mark.asyncio
    async def test_uses_profile_local_gmail_triage_script(self, monkeypatch, tmp_path):
        profile_home = tmp_path / "profiles" / "triage-profile"
        script_dir = profile_home / "scripts" / "gmail-triage"
        script_dir.mkdir(parents=True)
        script_path = script_dir / "send-draft.sh"
        script_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        adapter = _make_adapter()
        adapter._is_callback_user_authorized = MagicMock(return_value=True)

        query = SimpleNamespace(
            from_user=SimpleNamespace(id="12345", first_name="Tester"),
            message=SimpleNamespace(text="Original gmail triage message"),
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )

        seen = {}

        async def fake_create_subprocess_exec(*cmd, **kwargs):
            seen["cmd"] = cmd
            seen["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(
            "gateway.platforms.telegram.asyncio.create_subprocess_exec",
            fake_create_subprocess_exec,
        )

        await adapter._handle_gmail_triage_callback(
            query,
            "gt:send:thread-123",
            query_chat_id="12345",
            query_chat_type="private",
            query_thread_id=None,
            query_user_name="tester",
        )

        assert seen["cmd"][0] == str(script_path)
        assert seen["cmd"][1:] == ("thread-123",)
        query.answer.assert_awaited_once_with(
            text=adapter._GT_VERB_DISPATCH["send"][2],
        )
        query.edit_message_text.assert_awaited_once()
