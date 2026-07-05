"""Telegram durable SEO PR approval card button tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


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
from plugins.platforms.telegram.adapter import TelegramAdapter
import plugins.platforms.telegram.adapter as telegram_adapter_module


class Button:
    def __init__(self, text, callback_data=None, url=None):
        self.text = text
        self.callback_data = callback_data
        self.url = url


class Markup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    adapter._is_callback_user_authorized = lambda *args, **kwargs: True
    return adapter


def _load_spa(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.seo_pr_approvals as spa

    return importlib.reload(spa)


@pytest.fixture()
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def inline_keyboard_classes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(telegram_adapter_module, "InlineKeyboardButton", Button)
    monkeypatch.setattr(telegram_adapter_module, "InlineKeyboardMarkup", Markup)


def _sample_payload(**overrides):
    payload = {
        "site": "Kavera SEO Demo",
        "route": "/services/test-page",
        "target_keyword": "test page agency",
        "repo": "KaveraAI/kavera-seo-pages",
        "pr_url": "https://github.com/KaveraAI/kavera-seo-pages/pull/42",
        "pr_number": 42,
        "branch": "seo/test-page",
        "preview_url": "https://preview.example.com/services/test-page",
        "checks_summary": "3 passed",
        "checks_status": "passed",
        "source_platform": "telegram",
        "source_chat_id": "12345",
        "merge_payload": {"dry_run": True, "merge_method": "squash"},
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_send_seo_pr_approval_card_includes_durable_buttons(hermes_home, monkeypatch):
    spa = _load_spa(hermes_home, monkeypatch)
    store = spa.SEOApprovalStore()
    approval_id = store.create(_sample_payload())
    item = store.get(approval_id)
    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=99))

    result = await adapter.send_seo_pr_approval_card(chat_id="12345", approval=item)

    assert result.success is True
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 12345
    assert "Kavera SEO Demo" in kwargs["text"]
    assert item.pr_url in kwargs["text"]
    assert item.preview_url in kwargs["text"]
    rows = kwargs["reply_markup"].inline_keyboard
    callbacks = [button.callback_data for row in rows for button in row]
    assert f"seo:approve:{approval_id}" in callbacks
    assert f"seo:revise:{approval_id}" in callbacks
    assert f"seo:hold:{approval_id}" in callbacks


@pytest.mark.asyncio
async def test_telegram_callback_hold_uses_durable_store_without_memory_state(hermes_home, monkeypatch):
    spa = _load_spa(hermes_home, monkeypatch)
    store = spa.SEOApprovalStore()
    approval_id = store.create(_sample_payload())
    adapter = _make_adapter()
    # Simulate a restarted adapter: no in-memory approval state is populated.
    adapter._approval_state.clear()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=100))

    query = SimpleNamespace(
        data=f"seo:hold:{approval_id}",
        from_user=SimpleNamespace(id="42", first_name="Arthur"),
        message=SimpleNamespace(
            chat_id=12345,
            chat=SimpleNamespace(type="private"),
            message_thread_id=None,
            message_id=55,
            text="approval card",
        ),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, None)

    refreshed = spa.SEOApprovalStore().get(approval_id)
    assert refreshed.status == "held"
    query.answer.assert_awaited()
    query.edit_message_text.assert_awaited()


@pytest.mark.asyncio
async def test_telegram_callback_duplicate_approve_reports_consumed_once(hermes_home, monkeypatch):
    spa = _load_spa(hermes_home, monkeypatch)
    store = spa.SEOApprovalStore()
    approval_id = store.create(_sample_payload())
    adapter = _make_adapter()

    def make_query():
        return SimpleNamespace(
            data=f"seo:approve:{approval_id}",
            from_user=SimpleNamespace(id="42", first_name="Arthur"),
            message=SimpleNamespace(
                chat_id=12345,
                chat=SimpleNamespace(type="private"),
                message_thread_id=None,
                message_id=55,
                text="approval card",
            ),
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )

    first = make_query()
    second = make_query()

    await adapter._handle_callback_query(SimpleNamespace(callback_query=first), None)
    await adapter._handle_callback_query(SimpleNamespace(callback_query=second), None)

    assert spa.SEOApprovalStore().get(approval_id).status == "consumed"
    second.answer.assert_awaited()
    assert "already" in second.answer.await_args.kwargs["text"].lower()
