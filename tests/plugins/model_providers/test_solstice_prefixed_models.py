"""A prefixed Solstice model id gets the same thinking rules as the id the client sends.

The native client strips ``gemini/`` and ``google/`` before it names the model on the request, so
``gemini/gemini-4-neon`` reaches Neon; the reasoning mapping must key on that same id, or the requested effort is dropped.
"""

from __future__ import annotations

import json

import httpx
import pytest

import providers

_BASE = "https://generativelanguage.googleapis.com/v1alpha"
_REPLY = {"id": "i1", "status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]}


def _send(model: str, reasoning_config: dict) -> dict:
    """Run the real Solstice profile and client for one turn; return the posted body."""
    profile = providers.get_provider_profile("solstice")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    client = profile.create_client(api_key="token", base_url=_BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    extra_body = profile.build_extra_body(model=model, reasoning_config=reasoning_config, base_url=_BASE)
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], extra_body=extra_body)
    (request,) = seen
    assert request.url.path.endswith("/interactions:createPerUserQuota")
    return json.loads(request.content)


@pytest.mark.parametrize("model", ["gemini/gemini-4-neon", "google/gemini-4-neon", "GEMINI/gemini-4-neon"])
@pytest.mark.parametrize(
    ("reasoning_config", "level"),
    [({"enabled": True, "effort": "xhigh"}, "xhigh"), ({"enabled": True, "effort": "low"}, "low")],
)
def test_prefixed_model_keeps_its_effort_on_the_wire(model, reasoning_config, level):
    body = _send(model, reasoning_config)

    assert body["model"] == "gemini-4-neon"
    assert body["generation_config"]["thinking_level"] == level
    assert body["generation_config"]["thinking_summaries"] == "auto"


def test_prefixed_model_maps_like_the_bare_id():
    config = {"enabled": False}

    assert _send("gemini/gemini-4-neon", config)["generation_config"] == _send("gemini-4-neon", config)["generation_config"]


def test_models_prefix_is_mapped_on_the_id_the_client_sends():
    """The client does not strip ``models/``; the mapping must see that same id, so the two never disagree."""
    body = _send("models/gemini-4-neon", {"enabled": True, "effort": "xhigh"})

    assert body["model"] == "models/gemini-4-neon"
    assert "thinking_level" not in body["generation_config"]


@pytest.mark.parametrize("model", ["gemini/gemini-4-carbon", "google/gemini-4-carbon", "GEMINI/gemini-4-carbon"])
def test_prefixed_listed_model_gets_its_own_unset_default(model):
    """No effort set: a prefixed Carbon id gets Carbon's default level with thoughts on, like the bare id,
    through the profile and on the posted request."""
    profile = providers.get_provider_profile("solstice")

    assert profile.build_extra_body(model=model, reasoning_config=None, base_url=_BASE) == {
        "thinking_config": {"includeThoughts": True, "thinkingLevel": "medium"},
    }
    body = _send(model, None)
    assert body["model"] == "gemini-4-carbon"
    assert (body["generation_config"]["thinking_level"], body["generation_config"]["thinking_summaries"]) == (
        "medium", "auto")


@pytest.mark.parametrize("model", ["gemini/gemini-4-carbon", "google/gemini-4-carbon", "GEMINI/gemini-4-carbon"])
def test_setup_effort_picker_offers_carbons_levels_for_a_prefixed_id(model):
    from hermes_cli.main_provider_setup import _main_model_reasoning_efforts

    assert _main_model_reasoning_efforts(model, "solstice") == ["low", "medium", "high"]


def test_models_prefix_is_not_a_listed_model_for_the_setup_picker_either():
    """Same rule as the client, which sends ``models/...`` as typed: not the listed model's table."""
    from hermes_cli.main_provider_setup import _main_model_reasoning_efforts
    from hermes_constants import VALID_REASONING_EFFORTS

    assert _main_model_reasoning_efforts("models/gemini-4-carbon", "solstice") == list(VALID_REASONING_EFFORTS)
