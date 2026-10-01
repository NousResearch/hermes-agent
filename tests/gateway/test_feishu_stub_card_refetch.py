"""Tests for the Feishu adapter's interactive-card stub refetch (issue #129924).

Background
----------
Feishu's WebSocket gateway delivers bot-to-bot interactive cards as a
server-side compatibility skeleton — a single ``img`` placeholder plus
the literal text ``请升级至最新版本客户端，以查看内容`` — instead of
the original Card 1.0/2.0 JSON the sender submitted. The original
payload is only retrievable via REST when the caller passes
``card_msg_content_type=user_card_content``.

This file tests the four pieces the fix introduces on
``FeishuAdapter``:

* ``_normalize_interactive_message`` flags the stub payload via
  ``metadata.is_card_stub``.
* ``_extract_message_content`` triggers the refetch when the flag is
  set and a refetch hasn't been attempted yet for the same
  ``message_id``.
* ``_refetch_interactive_card_payload`` builds the right request and
  parses the response into a card-body JSON string the walker can
  re-extract.
* ``_remember_card_refetch`` is the module-level dedupe so the refetch
  path cannot loop on a chatty WebSocket delivery.

These are behavior-contract tests, not snapshot tests: they assert how
the function reacts to a given payload (e.g. ``is_card_stub`` is
``True`` iff the payload matches the placeholder shape), and they
describe the SDK call shape rather than freezing a builder-internal value.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_adapter() -> FeishuAdapter:
    """A FeishuAdapter that has only the fields needed by the methods under test.

    Bypasses ``__init__`` so we don't have to wire the whole gateway
    startup chain. The methods under test read ``self._client``, call
    ``self._run_blocking``, and (via ``_normalize``) read the bot
    identity fields that ``__init__`` normally installs through
    ``_apply_settings``. Everything else is either pure functional or
    already a classmethod.
    """
    adapter = object.__new__(FeishuAdapter)
    adapter._client = MagicMock()
    # Set via setattr: the fields are installed dynamically from settings,
    # so they are not class-level annotations.
    for _name in ("_bot_open_id", "_bot_user_id", "_bot_name"):
        setattr(adapter, _name, None)
    return adapter


_STUB_PAYLOAD = {
    "title": None,
    "elements": [[
        {"tag": "img", "image_key": "img_v3_02ad_test"},
        {"tag": "text", "text": "请升级至最新版本客户端，以查看内容"},
        {"tag": "text", "text": ""},
    ]],
}

_FULL_PAYLOAD = {
    "body": {
        "elements": [
            {"content": "我是 Mino 🌙", "element_id": "content", "tag": "markdown"},
            {"element_id": "_1", "tag": "hr"},
            {"content": "Agent: main", "element_id": "note", "tag": "markdown"},
        ],
    },
    "config": {"enable_forward_interaction": False},
    "schema": "2.0",
}


# ---------------------------------------------------------------------------
# 1. is_card_stub detection on the walker
# ---------------------------------------------------------------------------

def test_normalize_flags_degraded_placeholder_as_stub():
    """Placeholder-only body must surface ``is_card_stub=True``.

    Asserts a relationship (placeholder shape → flag set), not a
    snapshot of the metadata dict — the exact keys/values are
    deliberately not pinned.
    """
    adapter = _bare_adapter()
    normalized = adapter._normalize("interactive", json.dumps(_STUB_PAYLOAD, ensure_ascii=False), [])

    assert normalized.relation_kind == "interactive"
    assert normalized.metadata.get("is_card_stub") is True


def test_normalize_does_not_flag_real_card_payload():
    """A real Card 2.0 body must not be marked as a stub."""
    adapter = _bare_adapter()
    normalized = adapter._normalize("interactive", json.dumps(_FULL_PAYLOAD, ensure_ascii=False), [])

    assert normalized.relation_kind == "interactive"
    assert normalized.metadata.get("is_card_stub") is False


# ---------------------------------------------------------------------------
# 2. _extract_message_content triggers the refetch on the stub path
# ---------------------------------------------------------------------------

def _fake_message(message_id: str = "om_test_refetch_001"):
    """Build a stub-card inbound message using the captured placeholder shape."""
    return SimpleNamespace(
        message_type="interactive",
        content=json.dumps(_STUB_PAYLOAD, ensure_ascii=False),
        message_id=message_id,
        mentions=[],
    )


def _build_response_payload_as_text() -> SimpleNamespace:
    """Build the SDK response shape ``_refetch_interactive_card_payload`` parses.

    Mirrors the lark-oapi GetMessage response:
    ``data.items[0].body.content`` is a JSON string holding the original
    card JSON.
    """
    return SimpleNamespace(
        success=lambda: True,
        code="0",
        msg="success",
        data=SimpleNamespace(
            items=[
                SimpleNamespace(
                    body=SimpleNamespace(content=json.dumps(_FULL_PAYLOAD, ensure_ascii=False)),
                ),
            ],
        ),
    )


@pytest.mark.asyncio
async def test_extract_message_content_refetches_stub_card(monkeypatch, tmp_path):
    """On a stub card, ``_extract_message_content`` must refetch via REST and re-normalize.

    Asserts the behavior contract:
      * the refetched content is parsed as JSON and re-walked, so the
        resulting text contains the *real* body content, not the
        placeholder text or ``[Interactive message]``;
      * the SDK call shape is ``im.v1.message.get(request)`` — the
        exact request builder internals are deliberately not pinned.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _bare_adapter()
    adapter._bot_identity = lambda: None  # required by _normalize

    response = _build_response_payload_as_text()
    sdk_call = MagicMock(return_value=response)
    adapter._client.im.v1.message.get = sdk_call

    # _run_blocking is async by contract; the real one offloads to a
    # thread pool, so for a unit test we just return the canned
    # response synchronously inside the async wrapper.
    async def _run_blocking(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    adapter._run_blocking = _run_blocking

    # Mock _download_feishu_message_resources so the test does not
    # touch image fetching.
    adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))

    # Reset the module-level refetch cache for isolation.
    import plugins.platforms.feishu.adapter as feishu_module
    monkeypatch.setattr(feishu_module, "_card_refetch_seen", set())
    monkeypatch.setattr(feishu_module, "_card_refetch_seen_order", __import__("collections").OrderedDict())

    message = _fake_message(message_id="om_test_refetch_001")
    text, inbound_type, media_urls, media_types, is_at, mentions = (
        await adapter._extract_message_content(message)
    )

    # Real content, not placeholder text.
    assert "我是 Mino" in text
    assert "请升级至最新版本" not in text

    # The SDK was hit exactly once with the GetMessage endpoint.
    assert sdk_call.call_count == 1
    # ``_run_blocking(sdk_call, request)`` — the request is the only
    # positional arg the SDK call itself receives.
    (positional_args, _kwargs) = sdk_call.call_args
    request = positional_args[0]
    # The request builder path: ``message_id`` is the message_id we
    # passed; ``card_msg_content_type`` is the value that triggers the
    # real-payload path on Feishu.
    assert getattr(request, "message_id", None) == "om_test_refetch_001"
    assert getattr(request, "card_msg_content_type", None) == "user_card_content"


@pytest.mark.asyncio
async def test_extract_message_content_does_not_refetch_real_card(monkeypatch, tmp_path):
    """A real (non-stub) inbound card must NOT trigger a REST refetch.

    This is the negative path: the refetch exists only for the
    placeholder shape. Asserting the SDK is not called guards against
    a regression where every interactive message costs an extra REST
    round trip.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _bare_adapter()
    adapter._bot_identity = lambda: None

    sdk_call = MagicMock()
    adapter._client.im.v1.message.get = sdk_call

    async def _run_blocking(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    adapter._run_blocking = _run_blocking
    adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))

    import plugins.platforms.feishu.adapter as feishu_module
    monkeypatch.setattr(feishu_module, "_card_refetch_seen", set())
    monkeypatch.setattr(feishu_module, "_card_refetch_seen_order", __import__("collections").OrderedDict())

    message = SimpleNamespace(
        message_type="interactive",
        content=json.dumps(_FULL_PAYLOAD, ensure_ascii=False),
        message_id="om_test_no_refetch_002",
        mentions=[],
    )
    text, *_ = await adapter._extract_message_content(message)

    # Real card was walked directly — no refetch needed.
    assert "我是 Mino" in text
    assert sdk_call.call_count == 0


# ---------------------------------------------------------------------------
# 3. Module-level dedupe: _remember_card_refetch
# ---------------------------------------------------------------------------

def test_remember_card_refetch_is_idempotent(monkeypatch):
    """Calling ``_remember_card_refetch`` twice with the same id must return False the second time.

    This is the gate that prevents the refetch path from looping when
    Feishu re-delivers the same WebSocket event. Behaviour contract:
    ``True`` on first sight, ``False`` on every subsequent sight of the
    same id (within the dedup window).
    """
    import plugins.platforms.feishu.adapter as feishu_module
    monkeypatch.setattr(feishu_module, "_card_refetch_seen", set())
    monkeypatch.setattr(feishu_module, "_card_refetch_seen_order", __import__("collections").OrderedDict())

    assert feishu_module._remember_card_refetch("om_dedupe_test") is True
    # The second call within the dedup window must report "already seen".
    assert feishu_module._remember_card_refetch("om_dedupe_test") is False
    # A different id is fresh.
    assert feishu_module._remember_card_refetch("om_dedupe_other") is True


# ---------------------------------------------------------------------------
# 4. _build_get_message_request_with_full_card fallback on old SDKs
# ---------------------------------------------------------------------------

def test_build_request_falls_back_when_sdk_lacks_card_msg_content_type(monkeypatch):
    """A lark-oapi version without ``card_msg_content_type`` on the builder must fall back, not crash.

    Pre-1.7 SDKs don't expose ``card_msg_content_type`` on
    ``GetMessageRequest.builder()``; the helper must detect the
    missing method, log a debug message, and return the default
    request (which will then return only the placeholder body — but
    won't crash).
    """
    import plugins.platforms.feishu.adapter as feishu_module

    # Build a fake GetMessageRequest whose builder() lacks the
    # card_msg_content_type method, simulating a stale SDK.
    class FakeBuilder:
        def __init__(self):
            self.message_id_value = None

        def message_id(self, value):
            self.message_id_value = value
            return self

        def build(self):
            return SimpleNamespace(message_id=self.message_id_value)

    class FakeGetMessageRequest:
        @staticmethod
        def builder():
            return FakeBuilder()

    monkeypatch.setattr(feishu_module, "GetMessageRequest", FakeGetMessageRequest)

    request = FeishuAdapter._build_get_message_request_with_full_card("om_fallback_test")

    # Default request has the message_id but no card_msg_content_type;
    # the helper must not raise and must produce a usable request.
    assert getattr(request, "message_id", None) == "om_fallback_test"
    # No assertion about card_msg_content_type: on a stale SDK it is
    # absent, on a modern SDK it is "user_card_content". Both are
    # acceptable outcomes of this fallback path.