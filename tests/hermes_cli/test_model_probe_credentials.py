"""Metadata probes of the main endpoint send the ``model:`` block's own credential.

A bare ``provider: custom`` model with ``key_env`` (or ``api_key``) authenticates every turn with
it. The probes of the same endpoint send it too: the context-length probe behind model info, the
vision route's local-server fingerprint and aux-usage pricing. A keyed endpoint (for example
llama-server with ``--api-key``) answers 401 without it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

BASE = "http://127.0.0.1:47970/v1"
MODEL_BLOCK = {"default": "qwen-lab", "provider": "custom", "base_url": BASE, "key_env": "LAB_LLAMA_KEY",
               "context_length": 65536}


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("LAB_LLAMA_KEY", "lab-key")
    from hermes_cli.config import save_config
    save_config({"model": dict(MODEL_BLOCK)})


def test_model_block_key_is_bound_to_its_own_base_url(monkeypatch):
    from hermes_cli.runtime_provider_custom import model_block_key_for

    monkeypatch.setenv("LAB_LLAMA_KEY", "lab-key")
    assert model_block_key_for(MODEL_BLOCK, BASE + "/") == "lab-key"
    assert model_block_key_for({**MODEL_BLOCK, "api_key": "inline"}, BASE) == "inline"
    assert model_block_key_for(MODEL_BLOCK, "https://elsewhere.invalid/v1") == ""
    assert model_block_key_for({k: v for k, v in MODEL_BLOCK.items() if k != "key_env"}, BASE) == ""
    assert model_block_key_for("qwen-lab", BASE) == ""


def test_model_info_probes_context_length_with_the_configured_key(keyed, monkeypatch):
    from hermes_cli.web_routers import models as route

    seen = {}

    def probe(**kwargs):
        seen.update(kwargs)
        return 262144

    monkeypatch.setattr("agent.model_metadata.get_model_context_length", probe)
    info = route.get_model_info()
    assert seen["api_key"] == "lab-key" and seen["config_context_length"] is None
    assert info["auto_context_length"] == 262144 and info["config_context_length"] == 65536


def test_vision_route_fingerprints_the_local_server_with_the_configured_key(keyed, monkeypatch):
    from agent import image_routing
    from hermes_cli.config import load_config

    monkeypatch.setattr(image_routing, "_runtime_main", lambda key: "")
    assert image_routing._resolve_inference_api_key(load_config(), "custom") == "lab-key"


def test_aux_usage_pricing_lookup_sends_the_configured_key(keyed, monkeypatch):
    from agent import aux_accounting
    from agent import usage_pricing

    seen = {}

    def estimate(model, usage, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(amount_usd=None)

    monkeypatch.setattr(usage_pricing, "estimate_usage_cost", estimate)
    db = SimpleNamespace(record_auxiliary_usage=lambda *a, **k: None)
    token = aux_accounting.set_accounting_context(db, "s1")
    try:
        response = SimpleNamespace(model="qwen-lab", usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=3, total_tokens=13))
        aux_accounting.record_aux_usage(response, "title_generation", provider="custom", base_url=BASE + "/")
    finally:
        aux_accounting.reset_accounting_context(token)
    assert seen.get("api_key") == "lab-key"


def _failed_mint():
    raise RuntimeError("secret-bearing command failure")


_UNMINTED = [_failed_mint, lambda: object(), lambda: ""]
_SAVED = [{"key_env": "LAB_LLAMA_KEY"}, {"api_key": "inline-saved"}]


@pytest.fixture
def live_main():
    """Bind a live main runtime on the model block's endpoint whose credential is *source*."""
    from agent import auxiliary_client

    def bind(source):
        auxiliary_client.set_runtime_main("custom", "qwen-lab", base_url=BASE, api_key=source)

    yield bind
    auxiliary_client.clear_runtime_main()


@pytest.mark.parametrize("source", _UNMINTED)
@pytest.mark.parametrize("saved", _SAVED)
def test_a_failed_credential_source_keeps_the_vision_probe_keyless(monkeypatch, live_main, source, saved):
    """A declared credential source owns authentication even when its mint fails: the probe sends no
    bearer, never the model block's saved key for the same endpoint."""
    from agent import image_routing

    monkeypatch.setenv("LAB_LLAMA_KEY", "lab-key")
    cfg = {"model": {"default": "qwen-lab", "provider": "custom", "base_url": BASE, **saved}}
    live_main(source)
    assert image_routing._resolve_inference_base_url(cfg, "custom") == BASE
    assert image_routing._resolve_inference_api_key(cfg, "custom") == ""


@pytest.mark.parametrize("source, expected", [(lambda: "minted", "minted"), ("static", "static"), ("", "lab-key")])
def test_the_vision_probe_key_order_with_a_matching_model_block(keyed, live_main, source, expected):
    from agent import image_routing
    from hermes_cli.config import load_config

    live_main(source)
    assert image_routing._resolve_inference_api_key(load_config(), "custom") == expected


def _record_aux_pricing_key(monkeypatch):
    from agent import aux_accounting
    from agent import usage_pricing

    seen = {}

    def estimate(model, usage, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(amount_usd=None)

    monkeypatch.setattr(usage_pricing, "estimate_usage_cost", estimate)
    db = SimpleNamespace(record_auxiliary_usage=lambda *a, **k: None)
    token = aux_accounting.set_accounting_context(db, "s1")
    try:
        response = SimpleNamespace(model="qwen-lab", usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=3, total_tokens=13))
        aux_accounting.record_aux_usage(response, "title_generation", provider="custom", base_url=BASE)
    finally:
        aux_accounting.reset_accounting_context(token)
    return seen


@pytest.mark.parametrize("source", [*_UNMINTED, lambda: "minted"])
def test_aux_pricing_lookup_leaves_a_declared_credential_source_its_endpoint(keyed, monkeypatch, live_main, source):
    """While a credential source owns the live main endpoint, the model block's saved key is not sent."""
    live_main(source)
    assert _record_aux_pricing_key(monkeypatch) == {"provider": "custom", "base_url": BASE, "api_key": None}


def test_aux_pricing_lookup_sends_the_configured_key_beside_a_static_live_key(keyed, monkeypatch, live_main):
    live_main("lab-key")
    assert _record_aux_pricing_key(monkeypatch)["api_key"] == "lab-key"


def test_a_placeholder_inline_key_does_not_mask_a_usable_key_env(monkeypatch):
    """The probes pick the model block's first USABLE credential, as the runtime does: a template
    ``api_key: your-api-key`` beside a working ``key_env`` probes with the key_env value."""
    from hermes_cli.runtime_provider_custom import model_block_key_for

    monkeypatch.setenv("LAB_LLAMA_KEY", "lab-key")
    block = {**MODEL_BLOCK, "api_key": "your-api-key"}
    assert model_block_key_for(block, BASE) == "lab-key"
    assert model_block_key_for({**MODEL_BLOCK, "api_key": "your-api-key", "key_env": ""}, BASE) == ""


def test_every_main_endpoint_probe_skips_a_placeholder_inline_key(monkeypatch, live_main):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from agent import image_routing
    from hermes_cli.auth_declared_providers import custom_catalog_probe_api_key

    monkeypatch.setenv("LAB_LLAMA_KEY", "lab-key")
    save_config({"model": {**MODEL_BLOCK, "api_key": "your-api-key"}})
    cfg = load_config()
    assert resolve_runtime_provider(requested="custom")["api_key"] == "lab-key"
    live_main("")
    assert image_routing._resolve_inference_api_key(cfg, "custom") == "lab-key"
    assert custom_catalog_probe_api_key(cfg["model"], BASE) == "lab-key"
    assert _record_aux_pricing_key(monkeypatch)["api_key"] == "lab-key"
