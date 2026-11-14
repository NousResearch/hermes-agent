"""Every final Telegram delivery is indexed for quote-reply recovery (#147287 review findings).

Telegram echoes no body in ``reply_to_message``, so a quote-reply to a message we sent resolves only
through ``gateway.rich_sent_store``. Driven through ``TelegramAdapter.send`` / ``edit_message`` with a
fake Bot API: a final edit that succeeds only after a short flood wait, an oversized final split into
continuations, code that holds emphasis characters, and an index write the disk refuses.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import rich_sent_store
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Flood(Exception):
    def __init__(self):
        super().__init__("Flood control exceeded. Retry after 0.01 seconds")
        self.retry_after = 0.01


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(tmp_path / "sent.json"))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())
    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._bot = MagicMock()
    a._bot.edit_message_text = AsyncMock(return_value=True)
    return a


@pytest.mark.asyncio
async def test_final_edit_that_lands_after_a_flood_wait_is_indexed(adapter):
    adapter._bot.edit_message_text = AsyncMock(side_effect=[_Flood(), True])
    result = await adapter.edit_message("555", "41", "The deploy window is Tuesday", finalize=True)
    assert result.success
    assert rich_sent_store.lookup("555", "41") == "The deploy window is Tuesday"


@pytest.mark.asyncio
async def test_every_chunk_of_an_oversized_final_is_indexed(adapter):
    object.__setattr__(adapter, "MAX_MESSAGE_LENGTH", 160)
    ids = iter(range(900, 950))
    adapter._bot.send_message = AsyncMock(side_effect=lambda **kw: SimpleNamespace(message_id=next(ids)))
    content = " ".join(f"word{i}" for i in range(120))
    result = await adapter.edit_message("555", "41", content, finalize=True)
    assert result.success and result.continuation_message_ids
    for mid in ("41", *result.continuation_message_ids):
        text = rich_sent_store.lookup("555", mid)
        assert text and text.split()[0].startswith("word"), (mid, text)
    assert "word119" in rich_sent_store.lookup("555", result.continuation_message_ids[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["send", "final_edit"])
async def test_indexed_quote_keeps_code_verbatim(adapter, path):
    content = "Compute `a*b*c` then run:\n```\nx = a*b*c \\ 2\n```\nand **done**"
    if path == "send":
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=77))
        adapter._bot.do_api_request = AsyncMock(side_effect=RuntimeError("rich unavailable"))
        adapter._rich_messages_enabled = False
        result = await adapter.send("555", content)
        mid = result.message_id
    else:
        await adapter.edit_message("555", "41", content, finalize=True)
        mid = "41"
    text = rich_sent_store.lookup("555", mid)
    assert "`a*b*c`" in text and "x = a*b*c \\ 2" in text, text
    assert "**" not in text, text


@pytest.mark.asyncio
async def test_a_failed_index_write_is_logged(adapter, tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(blocker / "state" / "sent.json"))
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.telegram.adapter"):
        result = await adapter.edit_message("555", "41", "hello there", finalize=True)
    assert result.success
    assert any("failed to index sent message 41" in r.getMessage() for r in caplog.records)
