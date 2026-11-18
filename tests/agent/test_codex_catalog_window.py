"""Account-published context limits survive outages without crossing route or credential boundaries."""

import base64
import json
import time
from types import SimpleNamespace

import pytest

from agent import model_metadata as metadata


def _token(subject):
    encode = lambda obj: base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    return f"{encode({'alg': 'RS256'})}.{encode({'sub': subject})}.test-signature"


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(metadata, "_codex_oauth_context_cache", {})
    monkeypatch.setattr(metadata, "_codex_oauth_max_context_cache", {})
    monkeypatch.setattr(metadata, "save_context_length", lambda *args: None)
    calls = []

    def serve(rows):
        def get(url, **kwargs):
            calls.append(url)
            return SimpleNamespace(status_code=200, json=lambda: {"models": rows})
        monkeypatch.setattr(metadata.model_metadata_http, "get", get)

    def down():
        def get(url, **kwargs):
            calls.append(url)
            raise OSError("synthetic catalog outage")
        monkeypatch.setattr(metadata.model_metadata_http, "get", get)

    return serve, down, calls


def test_extended_variant_inherits_the_registered_base_reasoning_ladder(monkeypatch):
    from agent import reasoning_effort as reasoning

    # Exact base registrations must also apply to the wire-stripped context alias.
    monkeypatch.setattr(reasoning, "ASTRA_MODEL_IDS", frozenset({"gpt-6.2-astra"}))
    assert reasoning.codex_supported_efforts("openai/gpt-6.2-astra-900k") == reasoning.CODEX_ASTRA_EFFORTS

    from agent.transports.codex import ResponsesApiTransport
    params = dict(
        messages=[{"role": "user", "content": "Hello"}], tools=[], provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", is_codex_backend=True,
        reasoning_config={"effort": "none"}, request_overrides={"temperature": 0.5, "logprobs": True},
    )
    base = ResponsesApiTransport().build_kwargs(model="gpt-6.2-astra", **params)
    assert ResponsesApiTransport().build_kwargs(model="gpt-6.2-astra-900k", **params) == base


@pytest.mark.parametrize("model", ["gpt-6.2-astra", "gpt-6.2-sol"])
def test_observed_maximum_survives_outage_only_for_the_same_account_and_route(monkeypatch, catalog, model):
    serve, down, _ = catalog
    route = dict(base_url="https://chatgpt.com/backend-api/codex", provider="openai-codex", api_key=_token("first"))
    pin = model + "-900k"
    serve([dict(slug=model, context_window=272_000, max_context_window=872_000)])
    assert metadata.get_model_context_length(pin, **route) == 872_000
    # Advance past the existing cache TTL, then reproduce a catalog outage.
    later = time.time() + metadata._CODEX_OAUTH_CONTEXT_CACHE_TTL + 1
    monkeypatch.setattr(metadata.time, "time", lambda: later)
    down()
    assert metadata.get_model_context_length(pin, **route) == 872_000
    assert metadata.get_model_context_length(pin, **(route | {"api_key": _token("second")})) == 272_000
    assert metadata.get_model_context_length(pin, **(route | {"base_url": "https://relay.example/codex"})) == 272_000


@pytest.mark.parametrize("model", ["gpt-6.2-astra", "gpt-6.2-sol"])
@pytest.mark.parametrize("maximum,expected", [(872_000, 872_000), (1_050_000, 900_000), (200_000, 200_000), (None, 272_000)])
def test_custom_codex_route_uses_only_its_own_published_maximum(catalog, model, maximum, expected):
    serve, _, calls = catalog
    base_url = "https://relay.example/codex"
    # A proxy's generic context_window may describe the direct API. Do not use it without a max.
    serve([dict(slug=model, context_window=1_050_000, max_context_window=maximum)])
    result = metadata.get_model_context_length(
        model + "-900k", base_url=base_url, api_key="synthetic-relay-key", provider="custom:relay",
        custom_providers=[dict(name="relay", base_url=base_url, api_mode="codex_responses")],
    )
    assert result == expected
    assert calls and all(url.startswith(base_url + "/models?") for url in calls)
