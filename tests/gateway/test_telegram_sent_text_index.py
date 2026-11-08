"""Regression tests for Telegram quote-reply text recovery.

Telegram does not echo a message body back in ``reply_to_message``: the quoted text is
resolved from Hermes' own sent-message index (``gateway/rich_sent_store``).

Before the fix only the RICH send path (``sendRichMessage``) wrote that index, while the
ordinary MarkdownV2 path (``sendMessage``) — which carries the vast majority of outbound
messages — wrote nothing. A user quoting such a message therefore produced an inbound event
with ``reply_to_message_id`` set and ``reply_to_text`` empty: the agent answered a question
whose subject it could not see, and nothing in the logs showed why.

Covered here:
- the legacy send path is indexed (the regression itself);
- a quote-reply resolves the quoted body from the index;
- the native partial quote still wins over the index (control);
- an unrecoverable quote is logged instead of silently dropped;
- a quoted entry is kept alive when the index is trimmed (FIFO).
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import rich_sent_store
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

CHAT_ID = "182990963"


@pytest.fixture()
def sent_index(tmp_path, monkeypatch):
    """Point the sent-message index at a temp file (never the live ``~/.hermes/state`` file)."""
    path = tmp_path / "rich_sent_index.json"
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(path))
    return path


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))


def _make_inbound(reply_to_id=41219, quote_text=None):
    """A Telegram update that replies to ``reply_to_id`` with no body echoed back."""
    chat = SimpleNamespace(id=int(CHAT_ID), type="private", title=None, full_name="Александр Гогин")
    reply_to_message = SimpleNamespace(message_id=reply_to_id, text=None, caption=None)
    quote = SimpleNamespace(text=quote_text) if quote_text is not None else None
    return SimpleNamespace(chat=chat, reply_to_message=reply_to_message, quote=quote)


@pytest.mark.asyncio
async def test_legacy_send_is_indexed(sent_index):
    """A plain MarkdownV2 send must land in the sent index — the regression this fixes."""
    adapter = _make_adapter()
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=777))

    result = await adapter.send(CHAT_ID, "Прогресс бар сделали?")

    assert result.success
    data = json.loads(sent_index.read_text(encoding="utf-8"))
    stored = data.get(f"{CHAT_ID}:777", {}).get("t")
    assert stored is not None, "legacy send was not indexed"
    assert "Прогресс бар сделали?" in stored


def test_quote_reply_recovers_text_from_index(sent_index):
    """The quoted body comes back from the index when Telegram echoes nothing."""
    rich_sent_store.record(CHAT_ID, "41219", "Текст процитированного сообщения")

    reply_to_id, reply_to_text = _make_adapter()._reply_context(_make_inbound(reply_to_id=41219))

    assert reply_to_id == "41219"
    assert reply_to_text == "Текст процитированного сообщения"


def test_native_quote_still_wins_over_index(sent_index):
    """Control: a selected substring keeps priority over the whole indexed message."""
    rich_sent_store.record(CHAT_ID, "41219", "Полный текст сообщения")

    _, reply_to_text = _make_adapter()._reply_context(_make_inbound(reply_to_id=41219, quote_text="выделенная фраза"))

    assert reply_to_text == "выделенная фраза"


def test_unrecoverable_quote_is_logged(sent_index, caplog):
    """A quote with no recoverable body must be visible in the log, not silent."""
    with caplog.at_level(logging.WARNING, logger="hermes_plugins.platforms__telegram.adapter"):
        reply_to_id, reply_to_text = _make_adapter()._reply_context(_make_inbound(reply_to_id=999))

    assert reply_to_id == "999"
    assert reply_to_text is None
    assert any("no recoverable text" in record.getMessage() for record in caplog.records)


def test_quoted_entry_survives_index_trim(sent_index, monkeypatch):
    """Quoting an entry refreshes its timestamp, so the FIFO trim evicts an older entry instead."""
    monkeypatch.setattr(rich_sent_store, "_max_entries", lambda: 3)
    # Three entries with explicit ages: oldest / middle / the one the user quotes.
    sent_index.write_text(json.dumps({
        f"{CHAT_ID}:old": {"t": "старое сообщение", "ts": 1},
        f"{CHAT_ID}:mid": {"t": "среднее сообщение", "ts": 2},
        f"{CHAT_ID}:41219": {"t": "цитируемое сообщение", "ts": 3},
    }), encoding="utf-8")

    _, reply_to_text = _make_adapter()._reply_context(_make_inbound(reply_to_id=41219))
    assert reply_to_text == "цитируемое сообщение"

    rich_sent_store.record(CHAT_ID, "new", "новое сообщение")  # store grows past the cap

    final = json.loads(sent_index.read_text(encoding="utf-8"))
    assert f"{CHAT_ID}:41219" in final, "quoted entry was evicted by the FIFO trim"
    assert f"{CHAT_ID}:old" not in final, "the oldest entry should have been evicted instead"
