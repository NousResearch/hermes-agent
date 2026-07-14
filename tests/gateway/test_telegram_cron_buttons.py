"""Tests for Telegram inline buttons on cron deliveries."""

import json
import sys
from pathlib import Path
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
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

TelegramAdapter = load_plugin_adapter("telegram").TelegramAdapter


def _make_adapter(extra=None):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_send_attaches_cron_buttons_to_telegram_message(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    adapter = _make_adapter()
    mock_msg = MagicMock()
    mock_msg.message_id = 123
    adapter._bot.send_message = AsyncMock(return_value=mock_msg)

    result = await adapter.send(
        "12345",
        "Cron text",
        metadata={
            "cron_buttons": {
                "job_id": "abc123",
                "job_name": "test cron",
                "buttons": [
                    {"text": "1 — done", "value": "done"},
                    {"text": "2 — skipped", "value": "skipped"},
                ],
            }
        },
    )

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    assert bot.send_message.call_args is not None
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 12345
    assert kwargs["text"] == "Cron text"
    assert kwargs["reply_markup"] is not None
    tokens_path = tmp_path / "cron" / "button_tokens.json"
    tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
    assert len(tokens) == 2
    stored = list(tokens.values())
    assert stored[0]["job_id"] == "abc123"
    assert stored[0]["job_name"] == "test cron"
    assert stored[0]["button_text"] == "1 — done"
    assert stored[0]["button_value"] == "done"


@pytest.mark.asyncio
async def test_cron_button_callback_records_response(tmp_path, monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **k: True)

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    token = adapter._register_cron_button_token(
        job_id="abc123",
        job_name="test cron",
        button_index=1,
        label="2 — skipped",
        value="skipped",
    )

    query = AsyncMock()
    query.data = f"cj:{token}"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_id = 987
    query.message.message_thread_id = 42
    query.message.text = "Cron text"
    query.message.chat = MagicMock()
    query.message.chat.type = "supergroup"
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    await adapter._handle_callback_query(update, MagicMock())

    query.answer.assert_awaited_once()
    query.edit_message_text.assert_awaited_once()
    response_path = tmp_path / "cron" / "button_responses.jsonl"
    assert response_path.exists()
    line = response_path.read_text(encoding="utf-8").strip()
    assert '"job_id": "abc123"' in line
    assert '"button_value": "skipped"' in line
    assert '"thread_id": "42"' in line


@pytest.mark.asyncio
async def test_cron_button_callback_uses_delivery_time_mapping_after_job_edit(
    tmp_path,
    monkeypatch,
):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **k: True)

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    token = adapter._register_cron_button_token(
        job_id="abc123",
        job_name="test cron",
        button_index=0,
        label="Original choice",
        value="original-value",
    )

    query = AsyncMock()
    query.data = f"cj:{token}"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_id = 987
    query.message.message_thread_id = None
    query.message.text = "Cron text"
    query.message.chat = MagicMock()
    query.message.chat.type = "private"
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    # Simulate the job being edited after this delivery.  The callback must not
    # reload current job.buttons by job_id/index, because that would record the
    # new choice rather than the immutable choice shown on the old message.
    import cron.jobs as jobs_mod
    monkeypatch.setattr(
        jobs_mod,
        "get_job",
        lambda job_id: {
            "id": job_id,
            "name": "test cron",
            "buttons": [{"text": "Edited choice", "value": "edited-value"}],
        },
    )

    update = MagicMock()
    update.callback_query = query

    await adapter._handle_callback_query(update, MagicMock())

    response_path = tmp_path / "cron" / "button_responses.jsonl"
    line = response_path.read_text(encoding="utf-8").strip()
    assert '"button_text": "Original choice"' in line
    assert '"button_value": "original-value"' in line
    assert "edited-value" not in line
