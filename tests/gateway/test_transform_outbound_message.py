"""``transform_outbound_message`` — display-only rewrite of a final reply (#156752).

``transform_llm_output`` runs *before* the assistant row is persisted, so anything it returns
becomes conversation and is replayed to the model on later turns. A presentation change (folding
detail behind a tap, a footer, de-emphasis) must not: this hook fires on the delivery path instead,
after persistence, and only the text handed to the platform adapter changes.

Pinned here: the wire text is the rewrite, the failure mode is fail-open, and the payload carries
the routing context a plugin needs to act per surface.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base_delivery_ledger import transform_outbound_text
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._bot = AsyncMock()
    return adapter


def _event(platform="telegram", chat_type="dm"):
    return SimpleNamespace(source=SimpleNamespace(platform=platform, chat_type=chat_type, chat_id="42"))


def _hook(monkeypatch, fn):
    """Point the plugin dispatcher at *fn*, as a registered plugin would."""
    from hermes_cli import plugins

    calls: list[dict] = []

    def _invoke(name, **kwargs):
        calls.append({"name": name, **kwargs})
        return fn(**kwargs)

    monkeypatch.setattr(plugins, "invoke_hook", _invoke)
    return calls


def test_no_hook_leaves_the_reply_untouched(monkeypatch):
    calls = _hook(monkeypatch, lambda **kw: [])
    adapter = _make_adapter()

    assert transform_outbound_text(adapter, _event(), "sess", "ciao") == "ciao"
    assert calls[0]["name"] == "transform_outbound_message"


def test_rewrite_replaces_what_goes_out(monkeypatch):
    _hook(monkeypatch, lambda **kw: ["**> ripiegato||"])
    adapter = _make_adapter()

    assert transform_outbound_text(adapter, _event(), "sess", "testo originale") == "**> ripiegato||"


def test_first_non_empty_string_wins(monkeypatch):
    _hook(monkeypatch, lambda **kw: [None, "", 42, "primo", "secondo"])
    adapter = _make_adapter()

    assert transform_outbound_text(adapter, _event(), "sess", "x") == "primo"


def test_non_string_returns_are_ignored(monkeypatch):
    _hook(monkeypatch, lambda **kw: [None, {"handled": True}, 7])
    adapter = _make_adapter()

    assert transform_outbound_text(adapter, _event(), "sess", "x") == "x"


def test_raising_hook_is_fail_open(monkeypatch):
    def boom(**kw):
        raise RuntimeError("plugin esploso")

    _hook(monkeypatch, boom)
    adapter = _make_adapter()

    assert transform_outbound_text(adapter, _event(), "sess", "risposta") == "risposta"


def test_empty_reply_short_circuits_before_the_hook(monkeypatch):
    calls = _hook(monkeypatch, lambda **kw: ["non dovrebbe accadere"])
    adapter = _make_adapter()

    assert transform_outbound_text(adapter, _event(), "sess", "") == ""
    assert calls == []


def test_payload_carries_the_routing_context(monkeypatch):
    calls = _hook(monkeypatch, lambda **kw: [])
    adapter = _make_adapter()

    transform_outbound_text(adapter, _event(platform="telegram", chat_type="forum"), "sess-key", "ciao")

    payload = calls[0]
    assert payload["text"] == "ciao"
    assert payload["platform"] == "telegram"
    assert payload["chat_type"] == "forum"
    assert payload["session_id"] == "sess-key"


def test_missing_source_degrades_to_empty_context(monkeypatch):
    calls = _hook(monkeypatch, lambda **kw: [])
    adapter = _make_adapter()

    transform_outbound_text(adapter, SimpleNamespace(), "s", "ciao")

    assert calls[0]["platform"] == ""
    assert calls[0]["chat_type"] == ""


@pytest.mark.asyncio
async def test_send_final_text_delivers_the_rewrite(monkeypatch):
    _hook(monkeypatch, lambda **kw: ["**> ripiegato||"])
    adapter = _make_adapter()
    sent = {}

    async def _capture(event, session_key, text_content, metadata, **kwargs):
        sent["text"] = text_content
        return MagicMock(success=True, message_id="1"), adapter

    adapter.send_final_ledgered = _capture

    await adapter._send_final_text(_event(), "sess", "testo lungo originale", {},
                                   is_ephemeral_response=False, ephemeral_ttl=0,
                                   record_delivery=lambda *_: None)

    assert sent["text"] == "**> ripiegato||"
