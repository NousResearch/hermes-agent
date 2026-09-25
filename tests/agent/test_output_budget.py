"""Catalog ceilings must reach the first request and survive every retry."""

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI

from agent import model_metadata
from agent.output_budget import apply_output_budget, boosted_output_cap, model_output_limit


@pytest.fixture
def catalog(monkeypatch):
    entries = {
        "vendor/large": {"context_length": 200000, "max_completion_tokens": 128000},
        "vendor/small": {"context_length": 200000, "max_completion_tokens": 16000},
    }
    monkeypatch.setattr(model_metadata, "_model_metadata_cache", entries)
    monkeypatch.setattr(model_metadata, "_model_metadata_cache_time", time.time())
    return entries


def agent(**changes):
    values = dict(api_mode="chat_completions", provider="openrouter",
                  base_url="https://openrouter.ai/api/v1", api_key="test-key",
                  model="vendor/large@preset/demo", max_tokens=None,
                  _max_tokens_param=lambda n: {"max_tokens": n})
    return SimpleNamespace(**(values | changes))


@pytest.mark.parametrize("model", ["vendor/large", "vendor/large@preset/demo", "vendor/large:nitro@preset/demo"])
def test_catalog_lookup_preserves_wire_model(catalog, model):
    a = agent(model=model)
    kwargs = apply_output_budget(a, {"model": model, "messages": []})
    assert kwargs["max_tokens"] == 128000
    assert kwargs["model"] == model
    assert a.max_tokens is None


@pytest.mark.parametrize("value", [None, 0, -1, True, "unknown", 1.5])
def test_invalid_catalog_limit_stays_unknown(catalog, value):
    catalog["vendor/large"]["max_completion_tokens"] = value
    a = agent()
    assert model_output_limit(a) is None
    assert apply_output_budget(a, {"messages": []}) == {"messages": []}
    assert boosted_output_cap(a, None, 1) is None


def test_missing_metadata_never_shrinks_a_known_provider_budget(catalog):
    a = agent(model="vendor/unknown")
    assert [boosted_output_cap(a, 65536, n) for n in range(1, 5)] == [65536] * 4


def test_retry_never_exceeds_catalog_ceiling(catalog):
    a = agent()
    assert boosted_output_cap(a, None, 1) == 128000
    assert [boosted_output_cap(a, 128000, n) for n in range(1, 5)] == [128000] * 4
    assert boosted_output_cap(a, 65536, 1) == 128000
    assert boosted_output_cap(a, 256000, 1) == 128000


@pytest.mark.parametrize("explicit,expected", [(2048, 2048), (256000, 128000)])
def test_explicit_budget_is_preserved_within_ceiling(catalog, explicit, expected):
    a = agent(max_tokens=explicit)
    assert apply_output_budget(a, {"max_tokens": explicit})["max_tokens"] == expected
    assert boosted_output_cap(a, explicit, 1) == expected
    assert a.max_tokens == explicit


@pytest.mark.parametrize("enabled,expected", [(True, 65535), (False, 4096)])
def test_gemini_transport_thinking_headroom_is_preserved(catalog, enabled, expected):
    from agent.transports.chat_completions import ChatCompletionsTransport

    catalog["gemini-2.5-pro"] = {"max_completion_tokens": 65536}
    a = agent(model="gemini-2.5-pro", max_tokens=4096,
              reasoning_config={"enabled": enabled, "effort": "high"})
    kwargs = ChatCompletionsTransport().build_kwargs(
        model=a.model, messages=[], max_tokens=a.max_tokens,
        reasoning_config=a.reasoning_config, max_tokens_param_fn=a._max_tokens_param,
    )
    assert apply_output_budget(a, kwargs)["max_tokens"] == expected


def test_switching_models_does_not_reuse_previous_ceiling(catalog):
    a = agent()
    assert apply_output_budget(a, {})["max_tokens"] == 128000
    a.model = "vendor/small@preset/demo"
    assert apply_output_budget(a, {})["max_tokens"] == 16000
    assert boosted_output_cap(a, 128000, 1) == 16000
    a.model = "vendor/unknown"
    assert apply_output_budget(a, {}) == {}


def test_custom_endpoint_does_not_inherit_public_model_limit(catalog, monkeypatch):
    fetch = Mock(return_value={"vendor/large": {"max_completion_tokens": 3000}})
    monkeypatch.setattr(model_metadata, "fetch_endpoint_model_metadata", fetch)
    a = agent(base_url="https://private.example/v1", model="vendor/large")
    assert model_output_limit(a) == 3000
    fetch.assert_called_once_with(a.base_url, api_key=a.api_key)


def test_catalog_is_scoped_to_the_actual_provider(monkeypatch):
    from agent import models_dev

    lookup = Mock(return_value=SimpleNamespace(max_output_tokens=24000))
    monkeypatch.setattr(models_dev, "get_model_capabilities", lookup)
    # The concrete base URL wins over a stale provider label after a fallback.
    a = agent(base_url="https://api.openai.com/v1", model="catalog-model")
    assert apply_output_budget(a, {})["max_tokens"] == 24000
    assert lookup.call_args.args == ("openai", "catalog-model")


def test_lookalike_host_uses_its_own_catalog(catalog, monkeypatch):
    fetch = Mock(return_value={})
    monkeypatch.setattr(model_metadata, "fetch_endpoint_model_metadata", fetch)
    assert model_output_limit(agent(base_url="https://openrouter.ai.example/v1")) is None
    assert fetch.call_count == 1


def test_metadata_failure_keeps_request_and_retry_unspecified(monkeypatch):
    monkeypatch.setattr(model_metadata, "fetch_model_metadata", Mock(side_effect=OSError("offline")))
    a = agent()
    assert apply_output_budget(a, {}) == {}
    assert boosted_output_cap(a, None, 1) is None


def test_provider_and_credentials_invalidate_route_cache(catalog, monkeypatch):
    fetch = Mock(return_value={"vendor/large": {"max_completion_tokens": 3000}})
    monkeypatch.setattr(model_metadata, "fetch_endpoint_model_metadata", fetch)
    a = agent(model="vendor/large")
    assert model_output_limit(a) == 128000
    a.provider, a.base_url = "custom", "https://private.example/v1"
    assert model_output_limit(a) == 3000
    a.api_key = "second-account"
    fetch.return_value = {"vendor/large": {"max_completion_tokens": 6000}}
    assert model_output_limit(a) == 6000
    assert fetch.call_count == 2


def test_unknown_catalog_is_cached_briefly_and_recovers(monkeypatch):
    fetch = Mock(side_effect=[{}, {"vendor/large": {"max_completion_tokens": 128000}}])
    monkeypatch.setattr(model_metadata, "fetch_model_metadata", fetch)
    clock = Mock(return_value=10)
    monkeypatch.setattr("agent.output_budget.time.monotonic", clock)
    a = agent()
    assert model_output_limit(a) is None
    assert model_output_limit(a) is None
    assert fetch.call_count == 1
    clock.return_value = 71
    assert model_output_limit(a) == 128000


def test_provider_owned_preset_without_concrete_model_stays_unspecified(catalog):
    assert apply_output_budget(agent(model="@preset/demo"), {}) == {}


def test_catalog_fetch_does_not_invent_four_thousand_tokens(monkeypatch):
    response = Mock()
    response.json.return_value = {"data": [{"id": "vendor/unknown"}]}
    monkeypatch.setattr(model_metadata, "_model_metadata_cache", {})
    # Patch only the network boundary; exercise the real catalog parser and disk write.
    monkeypatch.setattr(model_metadata.model_metadata_http, "get", lambda *a, **k: response)
    assert model_metadata.fetch_model_metadata(force_refresh=True)["vendor/unknown"]["max_completion_tokens"] is None


def test_request_overrides_and_remaining_context_are_respected(catalog):
    a = agent(context_compressor=SimpleNamespace(context_length=20000))
    kwargs = apply_output_budget(a, {
        "messages": [{"role": "user", "content": "x" * 40000}],
        "max_tokens": 128000, "extra_body": {"max_completion_tokens": 15000},
    })
    assert 1 <= kwargs["max_tokens"] < 10000
    assert kwargs["extra_body"]["max_completion_tokens"] == kwargs["max_tokens"]


def test_responses_api_uses_its_own_parameter(catalog):
    a = agent(api_mode="codex_responses")
    assert apply_output_budget(a, {"input": []}) == {"input": [], "max_output_tokens": 128000}


@pytest.mark.parametrize("provider,base_url", [
    ("openai-codex", "https://gateway.example/codex"),
    ("custom", "https://chatgpt.com/backend-api/codex"),
])
def test_codex_backend_never_probes_or_sends_an_unsupported_cap(monkeypatch, provider, base_url):
    lookup = Mock()
    monkeypatch.setattr("agent.output_budget.get_model_output_limit", lookup)
    a = agent(api_mode="codex_responses", provider=provider, base_url=base_url)
    assert model_output_limit(a) is None
    assert apply_output_budget(a, {"input": []}) == {"input": []}
    assert boosted_output_cap(a, None, 1) is None
    lookup.assert_not_called()


def test_catalog_budget_reaches_serialized_sdk_requests_and_retries(catalog):
    from agent.transports.chat_completions import ChatCompletionsTransport

    bodies = []

    def receive(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                                       "model": "vendor/large", "choices": []})

    a = agent()
    transport = ChatCompletionsTransport()
    with OpenAI(api_key="test", base_url=a.base_url,
                http_client=httpx.Client(transport=httpx.MockTransport(receive))) as client:
        cap = None
        for n in range(3):
            kwargs = transport.build_kwargs(model=a.model, messages=[{"role": "user", "content": "hello"}],
                ephemeral_max_output_tokens=cap, max_tokens_param_fn=a._max_tokens_param)
            kwargs = apply_output_budget(a, kwargs)
            client.chat.completions.create(**kwargs)
            cap = boosted_output_cap(a, kwargs["max_tokens"], n + 1)
    assert [body["max_tokens"] for body in bodies] == [128000] * 3
    assert all(body["model"] == a.model for body in bodies)
