"""Learned image prices belong to an endpoint, not every server on its host."""

import json
from types import SimpleNamespace

import pytest

from agent import image_token_cost as itc
from agent.model_metadata import estimate_messages_tokens_rough
from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor


@pytest.fixture
def cost_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(itc, "_LEARNED", {})
    monkeypatch.setattr(itc, "_LOADED", False)
    with itc.image_cost_context(None):
        yield itc._cache_path()


def _calibrate(endpoint, price):
    history = [{"role": "user", "content": "start"}, {"role": "assistant", "content": "ok"}]
    anchor = capture_usage_anchor(10_000, 5, history)
    image = {"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}
    history.append(image)
    with itc.image_cost_context(0):
        text_only = anchored_context_tokens(history, anchor)
    agent = SimpleNamespace(_usage_anchor=anchor, model="vision-local", base_url=endpoint)
    return itc.calibrate_from_usage(agent, history, text_only + price), image


@pytest.mark.parametrize("first, second", [
    ("http://localhost:8080/v1", "http://localhost:8081/v1"),
    ("https://proxy.example/vision-a/v1", "https://proxy.example/vision-b/v1"),
    ("http://proxy.example/v1", "https://proxy.example/v1"),
])
def test_endpoint_prices_survive_reload_and_bind_to_estimators(cost_cache, monkeypatch, first, second):
    assert _calibrate(first, 4_000)[0] == 4_000
    assert itc.learned_image_token_cost("vision-local", second) == itc.DEFAULT_IMAGE_TOKEN_COST
    learned, image = _calibrate(second, 800)
    assert learned == 800

    # Reload the real cache file as a new process would, then exercise the turn binding.
    monkeypatch.setattr(itc, "_LEARNED", {})
    monkeypatch.setattr(itc, "_LOADED", False)
    estimates = []
    for endpoint, price in [(first, 4_000), (second, 800)]:
        assert itc.learned_image_token_cost("vision-local", endpoint + "/") == price
        itc.bind_image_token_cost(SimpleNamespace(model="vision-local", base_url=endpoint))
        estimates.append(estimate_messages_tokens_rough([image]))
    assert estimates[0] - estimates[1] == 4_000 - 800


def test_ambiguous_legacy_host_price_is_relearned_without_persisting_url_secrets(cost_cache):
    cost_cache.parent.mkdir(parents=True)
    cost_cache.write_text(json.dumps({"vision-local@localhost": 4_000}), encoding="utf-8")
    endpoint = "http://user:example-password@localhost:8080/v1?token=example-query-token"
    assert itc.learned_image_token_cost("vision-local", endpoint) == itc.DEFAULT_IMAGE_TOKEN_COST
    assert _calibrate(endpoint, 800)[0] == 800
    assert itc.learned_image_token_cost("vision-local", endpoint) == 800
    persisted = cost_cache.read_text(encoding="utf-8")
    assert "example-password" not in persisted
    assert "example-query-token" not in persisted
