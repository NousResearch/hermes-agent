"""Tests for Telegram gmail-triage callbacks."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_telegram_mock():
    """Install a minimal telegram mock so TelegramAdapter can import."""
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

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


@pytest.fixture()
def adapter():
    tg_adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    tg_adapter._is_callback_user_authorized = lambda *_a, **_kw: True
    return tg_adapter


class _FakeProc:
    returncode = 0

    async def communicate(self):
        return b"", b""


@pytest.mark.asyncio
async def test_gmail_triage_callback_uses_hermes_home_for_script_lookup(
    adapter, tmp_path, monkeypatch
):
    hermes_home = tmp_path / "profile-home"
    script_dir = hermes_home / "scripts" / "gmail-triage"
    script_dir.mkdir(parents=True)
    script_path = script_dir / "send-draft.sh"
    script_path.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    legacy_home = tmp_path / "legacy-home"
    legacy_home.mkdir()

    query = SimpleNamespace(
        from_user=SimpleNamespace(id=123, first_name="Ada"),
        message=SimpleNamespace(text="Original email"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    seen = {}

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return _FakeProc()

    with patch.object(Path, "home", return_value=legacy_home), patch(
        "gateway.platforms.telegram.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ):
        await adapter._handle_gmail_triage_callback(
            query,
            "gt:send:msg-123",
            query_chat_id="12345",
            query_chat_type="private",
            query_thread_id=None,
            query_user_name="Ada",
        )

    assert seen["cmd"] == (str(script_path), "msg-123")
    assert seen["kwargs"]["stdout"] is not None
    assert seen["kwargs"]["stderr"] is not None
    query.answer.assert_awaited_once()
    assert "missing" not in query.answer.await_args.kwargs["text"].lower()
    query.edit_message_text.assert_awaited_once()
