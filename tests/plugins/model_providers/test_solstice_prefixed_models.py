"""A prefixed Solstice model id gets the same thinking rules as the id the client sends.

The native client strips ``gemini/`` and ``google/`` before building the URL, so ``gemini/gemini-4-neon``
reaches Neon; the reasoning mapping must key on that same id, or the requested effort is dropped.
"""

from __future__ import annotations

import json

import httpx
import pytest

import providers

_BASE = "https://generativelanguage.googleapis.com/v1alpha"
_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}


def _send(model: str, reasoning_config: dict) -> tuple[str, dict]:
    """Run the real Solstice profile and client for one turn; return the request URL path and body."""
    profile = providers.get_provider_profile("solstice")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    client = profile.create_client(api_key="token", base_url=_BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    extra_body = profile.build_extra_body(model=model, reasoning_config=reasoning_config, base_url=_BASE)
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], extra_body=extra_body)
    (request,) = seen
    return request.url.path, json.loads(request.content)


@pytest.mark.parametrize("model", ["gemini/gemini-4-neon", "google/gemini-4-neon", "GEMINI/gemini-4-neon"])
@pytest.mark.parametrize(
    ("reasoning_config", "level"),
    [({"enabled": True, "effort": "xhigh"}, "xhigh"), ({"enabled": True, "effort": "low"}, "low")],
)
def test_prefixed_model_keeps_its_effort_on_the_wire(model, reasoning_config, level):
    path, body = _send(model, reasoning_config)

    assert path.endswith("/models/gemini-4-neon:generateContentPerUserQuota")
    assert body["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": level}


def test_prefixed_model_maps_like_the_bare_id():
    config = {"enabled": False}

    assert _send("gemini/gemini-4-neon", config)[1]["generationConfig"]["thinkingConfig"] == (
        _send("gemini-4-neon", config)[1]["generationConfig"]["thinkingConfig"]
    )


def test_models_prefix_is_mapped_on_the_id_the_client_sends():
    """The client does not strip ``models/``; the mapping must see that same id, so the two never disagree."""
    path, body = _send("models/gemini-4-neon", {"enabled": True, "effort": "xhigh"})

    assert path.endswith("/models/models/gemini-4-neon:generateContentPerUserQuota")
    assert "thinkingConfig" not in body.get("generationConfig", {})
