"""Negative or malformed ``offset`` values must not corrupt inline-picker pagination.

``inline_query.offset`` is client-controlled text echoed back from our previous ``next_offset``;
``int()`` accepts negatives, which would slice the catalog with a negative index (tail items or an
empty page) and emit a ``next_offset`` that resumes from the wrong position.
"""

import sys
from types import SimpleNamespace

import pytest

from plugins.platforms.telegram import inline_picker


@pytest.fixture
def catalog(monkeypatch):
    items = [{"name": f"cmd{i:03d}", "description": f"command {i}"} for i in range(120)]
    monkeypatch.setattr(inline_picker, "collect_inline_catalog", lambda: items)
    return items


def test_negative_offset_returns_first_page(catalog):
    results, next_offset = inline_picker.build_inline_results("cmd", "-5")
    first, first_next = inline_picker.build_inline_results("cmd", "0")
    assert results == first
    assert next_offset == first_next == "50"


@pytest.mark.parametrize("offset", ["-1", "-50", "-1000"])
def test_any_negative_offset_is_first_page(catalog, offset):
    results, next_offset = inline_picker.build_inline_results("cmd", offset)
    assert len(results) == 50
    assert results[0]["title"] == "/cmd000"
    assert next_offset == "50"


def test_negative_offset_does_not_serve_catalog_tail(monkeypatch):
    """With a catalog smaller than one page the negative slice used to return the tail."""
    items = [{"name": f"cmd{i:02d}", "description": ""} for i in range(40)]
    monkeypatch.setattr(inline_picker, "collect_inline_catalog", lambda: items)
    results, next_offset = inline_picker.build_inline_results("cmd", "-3")
    assert len(results) == 40
    assert next_offset == ""


@pytest.mark.parametrize("offset", ["", "abc", "3.5", "--5", None])
def test_malformed_offset_falls_back_to_first_page(catalog, offset):
    results, next_offset = inline_picker.build_inline_results("cmd", offset)
    assert len(results) == 50
    assert next_offset == "50"


def test_valid_offset_paginates(catalog):
    results, next_offset = inline_picker.build_inline_results("cmd", "50")
    assert results[0]["title"] == "/cmd050"
    assert next_offset == "100"
    last, last_next = inline_picker.build_inline_results("cmd", "100")
    assert len(last) == 20
    assert last_next == ""


def test_past_end_offset_returns_empty(catalog):
    results, next_offset = inline_picker.build_inline_results("cmd", "99999")
    assert results == []
    assert next_offset == ""


# --- e2e through the real handler: Update -> authz -> build_inline_results -> answer() ---


class _FakeArticle:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeMessageContent:
    def __init__(self, message_text):
        self.message_text = message_text


class _FakeInlineQuery:
    def __init__(self, query, offset, user_id="42"):
        self.query = query
        self.offset = offset
        self.from_user = SimpleNamespace(id=user_id, username="alice")
        self.answer_calls = []

    async def answer(self, articles, cache_time=None, is_personal=None, next_offset=None):
        self.answer_calls.append(
            {"articles": articles, "cache_time": cache_time,
             "is_personal": is_personal, "next_offset": next_offset})


def _adapter(monkeypatch, authorized=True):
    """A TelegramAdapter with just the pieces _handle_inline_query touches."""
    import types as _types
    from gateway.config import Platform
    from plugins.platforms.telegram.adapter import TelegramAdapter

    fake_telegram = _types.ModuleType("telegram")
    fake_telegram.InlineQueryResultArticle = _FakeArticle
    fake_telegram.InputTextMessageContent = _FakeMessageContent
    monkeypatch.setitem(sys.modules, "telegram", fake_telegram)

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM  # `name` is a property over self.platform
    monkeypatch.setattr(
        adapter, "_is_callback_user_authorized", lambda *a, **kw: authorized)
    return adapter


@pytest.mark.asyncio
async def test_e2e_negative_offset_serves_first_page(monkeypatch, catalog):
    """``inline_query.offset`` arrives client-controlled on the Update; '-3' must answer the
    first page, not the catalog tail or an empty page with a rewound next_offset."""
    adapter = _adapter(monkeypatch)
    iq = _FakeInlineQuery("cmd", "-3")
    await adapter._handle_inline_query(SimpleNamespace(inline_query=iq), None)
    (call,) = iq.answer_calls
    assert [a.title for a in call["articles"]][:3] == ["/cmd000", "/cmd001", "/cmd002"]
    assert len(call["articles"]) == 50
    assert call["next_offset"] == "50"


@pytest.mark.asyncio
async def test_e2e_valid_offset_paginates(monkeypatch, catalog):
    adapter = _adapter(monkeypatch)
    iq = _FakeInlineQuery("cmd", "50")
    await adapter._handle_inline_query(SimpleNamespace(inline_query=iq), None)
    (call,) = iq.answer_calls
    assert call["articles"][0].title == "/cmd050"
    assert call["next_offset"] == "100"


@pytest.mark.asyncio
async def test_e2e_unauthorized_gets_empty_answer(monkeypatch, catalog):
    """The catalog is never leaked: unauthorized senders get an empty page with no
    pagination regardless of offset."""
    adapter = _adapter(monkeypatch, authorized=False)
    iq = _FakeInlineQuery("cmd", "-3")
    await adapter._handle_inline_query(SimpleNamespace(inline_query=iq), None)
    (call,) = iq.answer_calls
    assert call["articles"] == []
    assert call["next_offset"] is None
