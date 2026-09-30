"""Service-tier emission for the native Gemini adapter (PR #37059).

Kept in its own file (not appended to test_gemini_native_adapter.py) so routine
upstream growth of the adapter's test tail cannot conflict with these tests.
"""

from __future__ import annotations

import json


class DummyResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


def test_build_gemini_request_emits_top_level_service_tier():
    """Gemini takes service_tier as a TOP-LEVEL generateContent body field.

    Flex:     https://ai.google.dev/gemini-api/docs/flex-inference
    Priority: https://ai.google.dev/gemini-api/docs/generate-content/priority-inference

    Both docs show it as a sibling of ``contents``, not inside
    ``generationConfig`` — putting it in generationConfig would be silently
    ignored and bill at the standard rate.
    """
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-3.6-flash",
        service_tier="flex",
    )

    assert request["service_tier"] == "flex"
    assert "service_tier" not in request.get("generationConfig", {})


def test_build_gemini_request_omits_service_tier_when_unset():
    """No tier configured must mean no field — not an explicit standard."""
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(messages=[{"role": "user", "content": "hi"}])

    assert "service_tier" not in request


def test_build_gemini_request_accepts_priority():
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-3.6-flash",
        service_tier="priority",
    )

    assert request["service_tier"] == "priority"


def test_build_gemini_request_drops_tier_for_pre_2_5_models():
    """Adapter-level defense: a stale pinned tier must never 400 the session.

    A tier pinned into request_overrides at build survives a runtime ``/model``
    switch verbatim, and Gemini's native REST rejects the whole request on an
    unexpected body field — so the adapter drops the field for models Google
    does not list as tier-eligible, rather than hard-failing every turn.
    """
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-2.0-flash",
        service_tier="flex",
    )

    assert "service_tier" not in request


def test_native_client_sends_service_tier_on_the_wire(monkeypatch):
    """End-to-end: the tier must appear in the posted generateContent body."""
    from agent.gemini_native_adapter import GeminiNativeClient

    posted = {}

    class _HTTP:
        def post(self, url, json=None, headers=None, timeout=None):
            posted["url"] = url
            posted["body"] = json
            return DummyResponse(
                payload={
                    "candidates": [
                        {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
                    ]
                }
            )

    client = GeminiNativeClient(api_key="test-key", http_client=_HTTP())
    client._create_chat_completion(
        model="gemini-3.6-flash",
        messages=[{"role": "user", "content": "hi"}],
        service_tier="flex",
    )

    assert posted["body"]["service_tier"] == "flex"

