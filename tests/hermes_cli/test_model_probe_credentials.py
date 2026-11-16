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
