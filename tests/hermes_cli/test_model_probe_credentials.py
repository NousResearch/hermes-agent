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
    from hermes_cli.runtime_provider_custom import main_endpoint_key_for

    monkeypatch.setenv("LAB_LLAMA_KEY", "lab-key")
    assert main_endpoint_key_for(MODEL_BLOCK, BASE + "/") == "lab-key"
    assert main_endpoint_key_for({**MODEL_BLOCK, "api_key": "inline"}, BASE) == "inline"
    assert main_endpoint_key_for(MODEL_BLOCK, "https://elsewhere.invalid/v1") == ""
    assert main_endpoint_key_for({k: v for k, v in MODEL_BLOCK.items() if k != "key_env"}, BASE) == ""
    assert main_endpoint_key_for("qwen-lab", BASE) == ""


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
    from hermes_cli.runtime_provider_custom import main_endpoint_key_for

    monkeypatch.setenv("LAB_LLAMA_KEY", "lab-key")
    block = {**MODEL_BLOCK, "api_key": "your-api-key"}
    assert main_endpoint_key_for(block, BASE) == "lab-key"
    assert main_endpoint_key_for({**MODEL_BLOCK, "api_key": "your-api-key", "key_env": ""}, BASE) == ""


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


def _main_endpoint_probe_keys(monkeypatch, provider):
    """The key each main-endpoint probe sends, beside the one inference sends, for the saved config."""
    from agent import image_routing
    from hermes_cli.auth_declared_providers import custom_catalog_probe_api_key
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.web_routers import models as route

    cfg = load_config()
    seen = {}

    def probe(**kwargs):
        seen.update(kwargs)
        return 1

    monkeypatch.setattr("agent.model_metadata.get_model_context_length", probe)
    route.get_model_info()
    monkeypatch.setattr(image_routing, "_runtime_main", lambda key: "")
    keys = {
        "runtime": resolve_runtime_provider(requested=provider)["api_key"],
        "vision": image_routing._resolve_inference_api_key(cfg, provider),
        "aux": _record_aux_pricing_key(monkeypatch)["api_key"],
        "model_info": seen["api_key"],
    }
    if provider == "custom":  # the catalog probe serves the bare custom route only
        keys["catalog"] = custom_catalog_probe_api_key(cfg["model"], BASE)
    return keys


@pytest.mark.parametrize("provider", ["myprov", "custom"])
def test_a_providers_entry_key_beats_a_stale_model_key_env_on_every_probe(monkeypatch, provider):
    """The runtime authenticates a ``providers:`` entry's route with the entry's own credential and
    never consults ``model.key_env`` for it; every probe of that endpoint sends the same key."""
    from hermes_cli.config import save_config

    monkeypatch.setenv("STALE_KEY", "stale-key")
    save_config({"model": {"default": "qwen-lab", "provider": provider, "base_url": BASE, "key_env": "STALE_KEY"},
                 "providers": {provider: {"base_url": BASE, "api_key": "entry-key-1234"}}})
    keys = _main_endpoint_probe_keys(monkeypatch, provider)
    assert set(keys.values()) == {"entry-key-1234"}, keys


def _seed_pool(slug, key):
    import json
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {slug: [
        {"id": "k1", "label": "primary", "auth_type": "api_key", "priority": 0, "source": "manual",
         "access_token": key, "base_url": BASE}]}}), encoding="utf-8")


@pytest.mark.parametrize("provider", ["lab", "custom"])
def test_a_credential_pool_owning_the_endpoint_beats_the_model_block_on_every_probe(monkeypatch, provider):
    """A credential pool (``hermes auth add``) owning the endpoint authenticates the runtime ahead of
    ``model.key_env``, on the named route and on the bare ``custom`` route alike; the probes send it too."""
    from hermes_cli.config import save_config

    monkeypatch.setenv("LAB_LLAMA_KEY", "lab-key")
    save_config({"model": {**MODEL_BLOCK, "provider": provider}, "providers": {"lab": {"base_url": BASE}}})
    _seed_pool("lab", "pool-key-5678")
    keys = _main_endpoint_probe_keys(monkeypatch, provider)
    assert set(keys.values()) == {"pool-key-5678"}, keys


def test_an_unavailable_config_backend_leaves_the_vision_probe_keyless(keyed, monkeypatch):
    """Resolving the endpoint's credential reads config; the backend's outage is a SystemExit on
    purpose, so it must not escape an optional probe and abort the image turn."""
    from agent import image_routing
    from hermes_cli import runtime_provider
    from hermes_cli.config import load_config
    from hermes_cli.config_backend import ConfigBackendUnavailable

    cfg = load_config()

    def unavailable():
        raise ConfigBackendUnavailable("remote config unavailable")

    monkeypatch.setattr(runtime_provider, "load_config", unavailable)
    monkeypatch.setattr(image_routing, "_runtime_main", lambda key: "")
    assert image_routing._resolve_inference_api_key(cfg, "custom") == ""


def test_probes_read_the_pool_without_rotating_it(monkeypatch):
    """A probe reads the row the pool would select; only a real request counts and rotates it."""
    import json

    from hermes_cli.config import save_config
    from hermes_constants import get_hermes_home

    save_config({"model": {**MODEL_BLOCK, "key_env": ""}, "providers": {"lab": {"base_url": BASE}},
                 "credential_pool_strategies": {"lab": "round_robin"}})
    rows = [{"id": f"k{i}", "label": f"row{i}", "auth_type": "api_key", "priority": i, "source": "manual",
             "access_token": f"pool-key-{i}000", "base_url": BASE} for i in (1, 2)]
    auth = get_hermes_home() / "auth.json"
    auth.write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {"lab": rows}}), encoding="utf-8")
    from hermes_cli.runtime_provider_custom import main_endpoint_key_for

    before = json.loads(auth.read_text(encoding="utf-8"))["credential_pool"]
    assert {main_endpoint_key_for(MODEL_BLOCK, BASE) for _ in range(3)} == {"pool-key-1000"}
    assert json.loads(auth.read_text(encoding="utf-8"))["credential_pool"] == before


def _seed_rows(rows):
    import json
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {"lab": [
        {"label": f"row-{row['id']}", "auth_type": "api_key", "source": "manual", "base_url": BASE, **row}
        for row in rows]}}), encoding="utf-8")


def test_probes_pick_the_pool_row_the_strategy_picks(monkeypatch):
    """Under ``least_used`` the runtime takes the least-used row, not the one the pool is on; the
    probe sends that row's key too, and still counts nothing."""
    import json

    from hermes_cli.config import save_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.runtime_provider_custom import main_endpoint_key_for
    from hermes_constants import get_hermes_home

    save_config({"model": {**MODEL_BLOCK, "provider": "lab", "key_env": ""}, "providers": {"lab": {"base_url": BASE}},
                 "credential_pool_strategies": {"lab": "least_used"}})
    _seed_rows([{"id": "a", "priority": 0, "access_token": "pool-key-busy", "request_count": 7},
                {"id": "b", "priority": 1, "access_token": "pool-key-idle", "request_count": 0}])
    auth = get_hermes_home() / "auth.json"
    counts = lambda: {r["id"]: r.get("request_count") for r in json.loads(auth.read_text(encoding="utf-8"))["credential_pool"]["lab"]}
    before = counts()
    assert main_endpoint_key_for(MODEL_BLOCK, BASE) == "pool-key-idle"
    assert counts() == before
    assert resolve_runtime_provider(requested="lab")["api_key"] == "pool-key-idle"


def test_a_loopback_pool_row_with_a_legacy_short_key_probes_keyless(monkeypatch):
    """The runtime sends ``no-key-required`` for a loopback pool row whose legacy key is unusable
    ('123'); the probe sends no bearer rather than the raw short value."""
    from hermes_cli.config import save_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.runtime_provider_custom import main_endpoint_key_for

    save_config({"model": {**MODEL_BLOCK, "provider": "lab", "key_env": ""}, "providers": {"lab": {"base_url": BASE}}})
    _seed_rows([{"id": "a", "priority": 0, "access_token": "123"}])
    assert resolve_runtime_provider(requested="lab")["api_key"] == "no-key-required"
    assert main_endpoint_key_for(MODEL_BLOCK, BASE) == ""


def test_a_bare_custom_endpoint_probes_with_its_host_gated_env_key(monkeypatch):
    """A bare ``custom`` model block with no key of its own authenticates with the env key gated on
    its host (``api.deepseek.com`` -> DEEPSEEK_API_KEY); the probes send the same key."""
    from hermes_cli.config import load_config, save_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.runtime_provider_custom import main_endpoint_key_for

    remote = "https://api.deepseek.com/v1"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env-key")
    save_config({"model": {"default": "deepseek-chat", "provider": "custom", "base_url": remote}})
    assert resolve_runtime_provider(requested="custom")["api_key"] == "deepseek-env-key"
    assert main_endpoint_key_for(load_config()["model"], remote) == "deepseek-env-key"


@pytest.mark.parametrize("provider", ["custom", "lmstudio"])
def test_the_vision_probe_sends_no_placeholder_inline_key(monkeypatch, provider):
    """A placeholder-only inline key is never a credential: the vision fingerprint probe of
    ``model.base_url`` goes keyless instead of falling back to the raw ``api_key`` value."""
    from agent import image_routing
    from hermes_cli.config import load_config, save_config

    save_config({"model": {"default": "qwen-lab", "provider": provider, "base_url": BASE, "api_key": "your-api-key"}})
    monkeypatch.setattr(image_routing, "_runtime_main", lambda key: "")
    cfg = load_config()
    assert image_routing._resolve_inference_base_url(cfg, provider) == BASE
    assert image_routing._resolve_inference_api_key(cfg, provider) == ""
