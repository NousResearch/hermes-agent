"""Regression tests for /model support of config.yaml custom_providers.

The terminal `hermes model` flow already exposes `custom_providers`, but the
shared slash-command pipeline (`/model` in CLI/gateway/Telegram) historically
only looked at `providers:`.
"""

import hermes_cli.providers as providers_mod
import pytest
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.providers import resolve_provider_full


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


@pytest.fixture(autouse=True)
def _disable_live_custom_provider_model_probe(monkeypatch):
    """Keep custom-provider picker fixtures independent of local model servers."""
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids", lambda *_a, **_kw: []
    )
    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids", lambda *_a, **_kw: []
    )


def test_list_authenticated_providers_includes_custom_providers(monkeypatch):
    """No-args /model menus should include saved custom_providers entries."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
                "model": "rotator-openrouter-coding",
            }
        ],
        max_models=50,
    )

    assert any(
        p["slug"] == "custom:local-(127.0.0.1:4141)"
        and p["name"] == "Local (127.0.0.1:4141)"
        and p["models"] == ["rotator-openrouter-coding"]
        and p["api_url"] == "http://127.0.0.1:4141/v1"
        for p in providers
    )








def test_is_routing_aggregator_excludes_flat_namespace_resellers():
    """opencode-go / opencode-zen stay ``is_aggregator=True`` (model-switch
    relies on it to search their flat bare-name catalog), but they are NOT
    routing aggregators — their models are first-party, so the picker dedup
    must not strip them. (#47077)"""
    # Still aggregators for model-switch flat-catalog resolution.
    assert providers_mod.is_aggregator("opencode-go") is True
    assert providers_mod.is_aggregator("opencode-zen") is True
    # But NOT routing aggregators for picker-dedup purposes.
    assert providers_mod.is_routing_aggregator("opencode-go") is False
    assert providers_mod.is_routing_aggregator("opencode-zen") is False
    # True routers and custom proxies remain routing aggregators.
    assert providers_mod.is_routing_aggregator("openrouter") is True
    assert providers_mod.is_routing_aggregator("custom:litellm") is True
    assert providers_mod.is_routing_aggregator("not-a-provider") is False


def test_picker_selection_resolves_named_custom_provider_model_id(monkeypatch):
    """Picker prefixes must not leak into a named custom provider API model id."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "test-key",
            "base_url": "https://token.sensenova.cn/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: _MOCK_VALIDATION,
    )
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_model_capabilities",
        lambda *a, **k: None,
    )

    result = switch_model(
        raw_input="sensenova/deepseek-v4-flash",
        current_provider="openai-codex",
        current_model="gpt-5.4",
        explicit_provider="custom:sensenova",
        user_providers={},
        custom_providers=[
            {
                "name": "sensenova",
                "base_url": "https://token.sensenova.cn/v1",
                "models": [
                    {"id": "deepseek-v4-flash", "name": "deepseek-v4-flash"}
                ],
            }
        ],
    )

    assert result.success is True
    assert result.target_provider == "custom:sensenova"
    assert result.new_model == "deepseek-v4-flash"








# ─────────────────────────────────────────────────────────────────────────────
# #9210: group custom_providers by (base_url, api_key) in /model picker
# ─────────────────────────────────────────────────────────────────────────────


def test_list_authenticated_providers_bare_custom_slug_recovers(monkeypatch):
    """Regression for #17478: when a prior failed switch left the bare
    literal "custom" in model.provider, the picker must NOT propagate
    that broken slug. It must fall back to the canonical
    ``custom:<name>`` form so the picker stays usable."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://localhost:11434/v1",
        user_providers={},
        custom_providers=[
            {"name": "Ollama — GLM 5.1", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "glm-5.1"},
        ],
        max_models=50,
    )

    matches = [p for p in providers if p.get("is_user_defined")]
    assert len(matches) == 1
    group = matches[0]
    # Canonical slug, NOT the bare "custom" that caused #17478
    assert group["slug"] == "custom:ollama"
    assert group["is_current"] is True




def test_custom_providers_uses_live_models_for_multi_model_endpoint(monkeypatch):
    """Custom providers with api_key + base_url should prefer live /models.

    Custom providers (section 4 of list_authenticated_providers) point at
    gateways like Bifrost that expose hundreds of models.  Reading only the
    static ``models:`` dict from config.yaml leaves the /model picker with
    a stale subset.  Live discovery fills the picker with all available
    models from the endpoint.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["gateway-model-a", "gateway-model-b", "gateway-model-c"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    custom_providers = [
        {
            "name": "my-gateway",
            "api_key": "sk-gateway-key",
            "base_url": "https://gateway.example.com/v1",
            "model": "gateway-model-a",
            "models": {
                "gateway-model-a": {"context_length": 128000},
                "gateway-model-b": {"context_length": 128000},
            },
        }
    ]

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        custom_providers=custom_providers,
        max_models=50,
    )

    gateway_prov = next(
        (
            p
            for p in providers
            if p.get("api_url") == "https://gateway.example.com/v1"
        ),
        None,
    )

    assert gateway_prov is not None, "Custom provider group not found in results"
    assert calls == [
        ("sk-gateway-key", "https://gateway.example.com/v1", {"headers": None})
    ], "fetch_api_models must be called with the custom provider's credentials"
    assert gateway_prov["models"] == [
        "gateway-model-a",
        "gateway-model-b",
        "gateway-model-c",
    ], "Live models must replace the static subset"
    assert gateway_prov["total_models"] == 3


def test_same_endpoint_different_extra_headers_not_collapsed(monkeypatch):
    """Entries sharing (api_url, credential, api_mode) but declaring different
    extra_headers must NOT collapse into one picker row — each is a distinct
    header-authenticated endpoint (e.g. per-tenant routing behind one proxy)
    and must probe /models with its own headers."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs.get("headers")))
        # Return a per-tenant model list keyed by the routing header so we can
        # assert each row got its OWN probe rather than a shared one.
        tenant = (kwargs.get("headers") or {}).get("X-Tenant", "none")
        return [f"model-{tenant}"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        custom_providers=[
            {
                "name": "Proxy Tenant A",
                "api_key": "shared-key",
                "base_url": "http://localhost:8081/v1",
                "extra_headers": {"X-Tenant": "a"},
            },
            {
                "name": "Proxy Tenant B",
                "api_key": "shared-key",
                "base_url": "http://localhost:8081/v1",
                "extra_headers": {"X-Tenant": "b"},
            },
        ],
        max_models=50,
    )

    rows = [
        p for p in providers if p.get("api_url") == "http://localhost:8081/v1"
    ]
    # Two distinct rows, not one collapsed row.
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}: {rows}"

    # Each tenant was probed with its OWN header set (order-independent).
    assert ("shared-key", "http://localhost:8081/v1", {"X-Tenant": "a"}) in calls
    assert ("shared-key", "http://localhost:8081/v1", {"X-Tenant": "b"}) in calls

    # Each row surfaces the model list its own headers unlocked.
    models_by_row = {tuple(r["models"]) for r in rows}
    assert models_by_row == {("model-a",), ("model-b",)}






def test_resolve_custom_provider_passes_key_env():
    """resolve_custom_provider should propagate key_env into api_key_env_vars.

    Regression: previously api_key_env_vars was always (), silently dropping
    the configured env var and causing 401s on every request.
    """
    from hermes_cli.providers import resolve_custom_provider

    resolved = resolve_custom_provider(
        "custom:token-plan",
        custom_providers=[
            {
                "name": "token-plan",
                "base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
                "key_env": "XIAOMI_MIMO_API_KEY",
                "model": "mimo-v2-pro",
            }
        ],
    )

    assert resolved is not None
    assert resolved.api_key_env_vars == ("XIAOMI_MIMO_API_KEY",)
    assert resolved.base_url == "https://token-plan-sgp.xiaomimimo.com/v1"


def test_discovered_models_auto_saved_to_cache(monkeypatch):
    """Discovered models are persisted to config so ``discover_models: false``
    has a populated cache on the next read (#65652).

    When a successful probe returns live models, ``_save_discovered_models_to_config``
    must be called with the provider's base_url and the discovered model list.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    save_calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        return ["discovered-a", "discovered-b", "discovered-c"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)
    monkeypatch.setattr(
        "hermes_cli.model_switch._save_discovered_models_to_config",
        lambda api_url, model_ids: save_calls.append((api_url, model_ids)),
    )

    custom_providers = [
        {
            "name": "my-gateway",
            "api_key": "***",
            "base_url": "https://gateway.example.com/v1",
            "discover_models": True,
            "model": "only-model",
            "models": {"only-model": {"context_length": 128000}},
        }
    ]

    providers = list_authenticated_providers(
        current_provider="my-gateway",
        current_base_url="https://gateway.example.com/v1",
        custom_providers=custom_providers,
        max_models=50,
        probe_custom_providers=True,
    )

    assert len(save_calls) == 1, (
        "_save_discovered_models_to_config must be called after a successful probe"
    )
    assert save_calls[0][0] == "https://gateway.example.com/v1"
    assert save_calls[0][1] == ["discovered-a", "discovered-b", "discovered-c"]

    gateway_prov = next(
        (p for p in providers if p.get("api_url") == "https://gateway.example.com/v1"),
        None,
    )
    assert gateway_prov is not None
    assert gateway_prov["models"] == ["discovered-a", "discovered-b", "discovered-c"]




def test_save_discovered_models_preserves_dict_form(monkeypatch):
    """``_save_discovered_models_to_config`` must not replace a dict-form
    ``models`` mapping (per-model metadata like ``context_length``) with
    a flat list of strings (#67841)."""
    from hermes_cli.model_switch import _save_discovered_models_to_config

    save_calls = []

    def fake_save(config):
        save_calls.append(dict(config))

    monkeypatch.setattr("hermes_cli.config.save_config", fake_save)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "my-gateway",
                    "base_url": "https://gateway.example.com/v1",
                    "models": {
                        "configured-model": {"context_length": 8192},
                    },
                }
            ]
        },
    )

    # Dict-form models must NOT be overwritten by discovered models
    _save_discovered_models_to_config(
        "https://gateway.example.com/v1",
        ["configured-model", "discovered-model"],
    )
    assert save_calls == [], (
        "Dict-form models must not be replaced with a flat list"
    )


def test_shared_url_different_display_names_are_separate_rows(monkeypatch):
    """Multiple custom_providers entries sharing base_url + api_key + api_mode
    but with *different* display-name prefixes (e.g. a proxy fronting
    cerebras, groq and perplexity at one URL) must each get their own picker
    row, not collapse into one."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    # Stub live discovery so the test is deterministic regardless of network.
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key, base_url, **kwargs: [],
    )

    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        user_providers={},
        custom_providers=[
            {"name": "Cerebras", "base_url": "https://proxy.example.com/v1",
             "api_key": "proxy-key", "model": "llama-4-scout"},
            {"name": "Groq", "base_url": "https://proxy.example.com/v1",
             "api_key": "proxy-key", "model": "llama-4-scout"},
            {"name": "Perplexity", "base_url": "https://proxy.example.com/v1",
             "api_key": "proxy-key", "model": "sonar-pro"},
        ],
        max_models=50,
    )

    custom = [p for p in providers if p.get("is_user_defined")]
    names = sorted(p["name"] for p in custom)
    assert names == ["Cerebras", "Groq", "Perplexity"], (
        f"expected three separate rows, got {names}"
    )
    # Each row carries only its own model (no cross-contamination).
    by_name = {p["name"]: p["models"] for p in custom}
    assert by_name["Cerebras"] == ["llama-4-scout"]
    assert by_name["Groq"] == ["llama-4-scout"]
    assert by_name["Perplexity"] == ["sonar-pro"]


def test_excluded_providers_hides_builtin_row(monkeypatch):
    """``excluded_providers`` must hide a built-in provider row that would
    otherwise surface when its credentials are present."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    baseline = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )
    assert any(p["slug"] == "openrouter" for p in baseline), (
        "sanity: openrouter row must appear when OPENROUTER_API_KEY is set"
    )

    filtered = list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
        user_providers={},
        custom_providers=[],
        max_models=50,
        excluded_providers=["openrouter"],
    )
    assert not any(p["slug"] == "openrouter" for p in filtered), (
        "excluded_providers=['openrouter'] must hide the openrouter row"
    )


