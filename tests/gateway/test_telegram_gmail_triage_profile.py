"""Gmail-triage inline-button callbacks must be profile-aware.

Each Hermes profile (`hermes -p work`) has its own ``HERMES_HOME``; the
gmail-triage helper scripts live under ``<HERMES_HOME>/scripts/gmail-triage/``
(same convention as ``tools/cronjob_tools.py``). The ``gt:verb:arg`` callback
handler previously hardcoded ``Path.home()/.hermes/scripts`` and reported
"script missing" whenever the script lived in a non-default profile directory.

Regression: the resolved script path must follow ``get_hermes_home()``.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_telegram_mock():
    """Wire up the minimal mocks required to import TelegramAdapter."""
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

from gateway.platforms.telegram import TelegramAdapter
from gateway.config import PlatformConfig


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_gt_query(data: str):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.text = "New email from boss"
    query.from_user = MagicMock()
    query.from_user.id = "12345"
    query.from_user.first_name = "Norbert"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query
    return update, query


class _FakeProc:
    returncode = 0

    async def communicate(self):
        return (b"", b"")


@pytest.mark.asyncio
async def test_gmail_triage_resolves_script_under_profile_hermes_home(tmp_path, monkeypatch):
    """gt: callbacks resolve scripts under HERMES_HOME, not ~/.hermes."""
    profile_home = tmp_path / "profiles" / "work"
    scripts_dir = profile_home / "scripts" / "gmail-triage"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / "archive.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    adapter = _make_adapter()
    update, query = _make_gt_query("gt:archive:msg123")

    captured = {}

    async def _fake_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        with patch("asyncio.create_subprocess_exec", new=_fake_exec):
            await adapter._handle_callback_query(update, MagicMock())

    assert "cmd" in captured, (
        "script under the profile HERMES_HOME was not executed — "
        "callback likely resolved the wrong directory and reported it missing"
    )
    executed_path = Path(captured["cmd"][0])
    assert executed_path == script
    assert str(profile_home) in str(executed_path)
    # The callback acknowledged success, not a missing-script error.
    answer_text = query.answer.call_args[1]["text"]
    assert "missing" not in answer_text.lower()


@pytest.mark.asyncio
async def test_gmail_triage_missing_script_does_not_spawn(tmp_path, monkeypatch):
    """When the script is absent under HERMES_HOME, report missing and skip exec."""
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    adapter = _make_adapter()
    update, query = _make_gt_query("gt:archive:msg123")

    spawned = SimpleNamespace(called=False)

    async def _fake_exec(*cmd, **kwargs):
        spawned.called = True
        return _FakeProc()

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        with patch("asyncio.create_subprocess_exec", new=_fake_exec):
            await adapter._handle_callback_query(update, MagicMock())

    assert spawned.called is False
    answer_text = query.answer.call_args[1]["text"]
    assert "missing" in answer_text.lower()
