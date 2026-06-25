"""Regression tests for /model support of config.yaml custom_providers.

The terminal `hermes model` flow already exposes `custom_providers`, but the
shared slash-command pipeline (`/model` in CLI/gateway/Telegram) historically
only looked at `providers:`.
"""

import hermes_cli.providers as providers_mod
import pytest
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.providers import resolve_provider_full
from providers.base import ProviderProfile


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


def _install_unit_test_provider_profile(monkeypatch) -> ProviderProfile:
    profile = ProviderProfile(
        name="unit-test-profile-provider",
        aliases=("unit-test-profile", "utpp"),
        display_name="Unit Test Profile Provider",
        description="ProviderProfile-only provider used by resolver tests.",
        signup_url="https://unit-test-provider.example/signup",
        env_vars=("UTPP_API_KEY", "UTPP_BASE_URL"),
        base_url="https://unit-test-provider.example/v1",
        api_mode="anthropic_messages",
        fallback_models=("unit-profile-model",),
    )

    def fake_get_provider_profile(name):
        if name in {profile.name, *profile.aliases}:
            return profile
        return None

    import providers as provider_registry
    from hermes_cli import auth as auth_mod

    monkeypatch.setattr(provider_registry, "get_provider_profile", fake_get_provider_profile)
    monkeypatch.setitem(
        auth_mod.PROVIDER_REGISTRY,
        profile.name,
        auth_mod.ProviderConfig(
            id=profile.name,
            name=profile.display_name,
            auth_type="api_key",
            inference_base_url=profile.base_url,
            api_key_env_vars=("UTPP_API_KEY",),
            base_url_env_var="UTPP_BASE_URL",
            extra={"api_mode": profile.api_mode},
        ),
    )
    for alias in profile.aliases:
        monkeypatch.setitem(auth_mod.PROVIDER_REGISTRY, alias, auth_mod.PROVIDER_REGISTRY[profile.name])
    return profile


def test_provider_profile_alias_resolves_through_cli_provider_identity(monkeypatch):
    """ProviderProfile plugins should work in shared --provider resolution."""
    profile = _install_unit_test_provider_profile(monkeypatch)

    resolved = resolve_provider_full("unit-test-profile")

    assert resolved is not None
    assert resolved.id == profile.name
    assert resolved.name == profile.display_name
    assert resolved.transport == "anthropic_messages"
    assert resolved.api_key_env_vars == ("UTPP_API_KEY",)
    assert resolved.base_url_env_var == "UTPP_BASE_URL"
    assert resolved.base_url == profile.base_url
    assert resolved.doc == profile.description
    assert resolved.source == "provider-profile"


def test_normalize_provider_does_not_trigger_profile_discovery(monkeypatch):
    """Hot-path normalization should stay cheap; get_provider handles profiles."""
    import agent.models_dev as models_dev
    import providers as provider_registry

    monkeypatch.setattr(models_dev, "get_provider_info", lambda key: None)

    def fail_get_provider_profile(name):
        raise AssertionError("normalize_provider must not discover provider profiles")

    monkeypatch.setattr(provider_registry, "get_provider_profile", fail_get_provider_profile)

    assert providers_mod.normalize_provider("missing-profile-provider") == "missing-profile-provider"


def test_get_provider_does_not_trigger_profile_discovery(monkeypatch):
    """Built-in lookup stays cheap; profile plugins belong to full resolution."""
    import agent.models_dev as models_dev
    import providers as provider_registry

    monkeypatch.setattr(models_dev, "get_provider_info", lambda key: None)

    profile_calls = []

    def record_get_provider_profile(name):
        profile_calls.append(name)
        return None

    monkeypatch.setattr(provider_registry, "get_provider_profile", record_get_provider_profile)

    assert providers_mod.get_provider("missing-profile-provider") is None
    assert profile_calls == []


def test_switch_model_accepts_explicit_provider_profile_alias(monkeypatch):
    """Explicit /model --provider should accept ProviderProfile aliases."""
    profile = _install_unit_test_provider_profile(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider: {"api_key": "profile-key", "base_url": profile.base_url, "source": "test"},
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="unit-profile-model",
        current_provider="openrouter",
        current_model="anthropic/claude-sonnet-4.6",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="unused",
        explicit_provider="unit-test-profile",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert result.target_provider == profile.name
    assert result.provider_label == profile.display_name
    assert result.new_model == "unit-profile-model"
    assert result.base_url == profile.base_url
    assert result.api_mode == "anthropic_messages"


def test_provider_profile_aliases_do_not_override_existing_provider_ids():
    """A profile alias must not recanonicalize existing Hermes/models.dev ids."""
    assert providers_mod.normalize_provider("opencode") == "opencode"
    assert providers_mod.is_aggregator("opencode-zen") is True
    assert providers_mod.is_aggregator("opencode-go") is True


def test_builtin_custom_profile_does_not_change_get_provider_semantics():
    """Bare custom endpoint handling remains owned by model_switch fallback."""
    assert providers_mod.get_provider("custom") is None


def test_bare_custom_provider_keeps_current_base_url_for_autodetect(monkeypatch):
    """The built-in custom ProviderProfile must not bypass bare-custom fallback."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._auto_detect_local_model",
        lambda base_url: "local-autodetected-model" if base_url == "http://127.0.0.1:11434/v1" else "",
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="",
        current_provider="custom",
        current_model="old-local-model",
        current_base_url="http://127.0.0.1:11434/v1",
        current_api_key="unused",
        explicit_provider="custom",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert result.target_provider == "custom"
    assert result.new_model == "local-autodetected-model"
    assert result.base_url == "http://127.0.0.1:11434/v1"


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






def test_resolve_provider_full_finds_named_custom_provider():
    """Explicit /model --provider should resolve saved custom_providers entries."""
    resolved = resolve_provider_full(
        "custom:local-(127.0.0.1:4141)",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:4141)",
                "base_url": "http://127.0.0.1:4141/v1",
            }
        ],
    )

    assert resolved is not None
    assert resolved.id == "custom:local-(127.0.0.1:4141)"
    assert resolved.name == "Local (127.0.0.1:4141)"
    assert resolved.base_url == "http://127.0.0.1:4141/v1"
    assert resolved.source == "user-config"


@pytest.mark.parametrize(
    "requested",
    [
        "Local Ollama",
        "local-ollama",
        "local-127.0.0.1:11434",
        "custom:local-ollama",
        "custom:local-127.0.0.1:11434",
    ],
)
def test_keyed_custom_provider_legacy_aliases_resolve_to_stable_key(requested):
    """Every historical identity resolves, but keyed providers return one ID."""
    resolved = resolve_provider_full(
        requested,
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "provider_key": "local-127.0.0.1:11434",
                "base_url": "http://127.0.0.1:11434/v1",
            }
        ],
    )

    assert resolved is not None
    assert resolved.id == "custom:local-127.0.0.1:11434"
    assert resolved.name == "Local Ollama"


def test_keyed_custom_provider_bare_custom_fallback_uses_stable_key():
    resolved = resolve_provider_full(
        "custom",
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "provider_key": "local-127.0.0.1:11434",
                "base_url": "http://127.0.0.1:11434/v1",
            }
        ],
    )

    assert resolved is not None
    assert resolved.id == "custom:local-127.0.0.1:11434"


@pytest.mark.parametrize(
    "requested",
    ["foo", "custom:foo", "custom:custom:foo"],
)
def test_prefixed_provider_key_does_not_accumulate_custom_prefixes(requested):
    """Accept the historical doubled form without writing a third identity."""
    resolved = resolve_provider_full(
        requested,
        user_providers={},
        custom_providers=[
            {
                "name": "Foo Relay",
                "provider_key": "custom:foo",
                "base_url": "https://foo.example/v1",
            }
        ],
    )

    assert resolved is not None
    assert resolved.id == "custom:foo"


def test_list_authenticated_providers_includes_active_bare_custom_endpoint(monkeypatch):
    """Bare model.provider=custom + model.base_url should still populate /model.

    Users can configure a one-off OpenAI-compatible endpoint directly under
    ``model:`` without a named ``providers:`` or ``custom_providers:`` row.
    The gateway picker receives only the current model/base_url slice, so it
    must surface that active endpoint rather than looking like config was
    ignored.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="https://www.ccsub.net/v1",
        current_model="gpt-4o",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )

    bare_custom = next((p for p in providers if p["slug"] == "custom"), None)
    assert bare_custom is not None
    assert bare_custom["name"] == "Custom endpoint"
    assert bare_custom["is_current"] is True
    assert bare_custom["is_user_defined"] is True
    assert bare_custom["models"] == ["gpt-4o"]
    assert bare_custom["api_url"] == "https://www.ccsub.net/v1"


def test_list_authenticated_providers_can_probe_active_bare_custom_endpoint(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key, api_url, **kwargs: ["gpt-4o", "gpt-4o-mini"],
    )

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="https://www.ccsub.net/v1",
        current_model="gpt-4o",
        user_providers={},
        custom_providers=[],
        probe_custom_providers=False,
        probe_current_custom_provider=True,
    )

    bare_custom = next(p for p in providers if p["slug"] == "custom")
    assert bare_custom["is_current"] is True
    assert bare_custom["models"] == ["gpt-4o", "gpt-4o-mini"]


def test_switch_model_accepts_explicit_bare_custom_current_endpoint(monkeypatch):
    """Picker selections for bare custom endpoints should route to current base_url."""
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="gpt-4o-mini",
        current_provider="custom",
        current_model="gpt-4o",
        current_base_url="https://www.ccsub.net/v1",
        current_api_key="sk-test",
        explicit_provider="custom",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert result.target_provider == "custom"
    assert result.provider_label == "Custom endpoint"
    assert result.new_model == "gpt-4o-mini"
    assert result.base_url == "https://www.ccsub.net/v1"
    assert result.api_key == "sk-test"


def test_is_aggregator_recognizes_named_custom_provider():
    assert providers_mod.is_aggregator("custom:hpc-ai") is True
    assert providers_mod.is_aggregator("custom:litellm") is True


def test_is_aggregator_leaves_unknown_provider_non_aggregator():
    assert providers_mod.is_aggregator("not-a-provider") is False


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


def test_compatible_keyed_provider_uses_stable_key_and_accepts_legacy_current_name(
    monkeypatch,
):
    """The merged providers view keeps the config key while old IDs stay current."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom:local-ollama",
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "provider_key": "local-127.0.0.1:11434",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen3.5:9b",
            }
        ],
        max_models=50,
        probe_custom_providers=False,
    )

    row = next(p for p in providers if p.get("is_user_defined"))
    assert row["slug"] == "custom:local-127.0.0.1:11434"
    assert row["is_current"] is True


def test_user_provider_row_recognizes_stable_custom_key_as_current(monkeypatch):
    """Section 3 keeps its legacy row slug but recognizes the stable ID."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom:local-127.0.0.1:11434",
        user_providers={
            "local-127.0.0.1:11434": {
                "name": "Local Ollama",
                "base_url": "http://127.0.0.1:11434/v1",
                "default_model": "qwen3.5:9b",
            }
        },
        custom_providers=[],
        max_models=50,
        probe_custom_providers=False,
    )

    row = next(p for p in providers if p.get("is_user_defined"))
    assert row["slug"] == "local-127.0.0.1:11434"
    assert row["is_current"] is True


def test_list_authenticated_providers_distinct_endpoints_stay_separate(monkeypatch):
    """Entries with different base_urls must produce separate picker rows
    even if some display names happen to be similar."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {"name": "Ollama — GLM 5.1", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "glm-5.1"},
            {"name": "Moonshot", "base_url": "https://api.moonshot.cn/v1",
             "api_key": "sk-m", "model": "moonshot-v1"},
            {"name": "Ollama — Qwen3-coder", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "qwen3-coder"},
        ],
        max_models=50,
        probe_custom_providers=False,
    )

    custom_groups = [p for p in providers if p.get("is_user_defined")]
    assert len(custom_groups) == 2
    # Ollama endpoint collapses to one row with both models
    ollama = next(p for p in custom_groups if p["name"] == "Ollama")
    assert set(ollama["models"]) == {"glm-5.1", "qwen3-coder"}
    moonshot = next(p for p in custom_groups if p["name"] == "Moonshot")
    assert moonshot["models"] == ["moonshot-v1"]


def test_list_authenticated_providers_same_url_different_keys_disambiguated(monkeypatch):
    """Two custom_providers entries with the same base_url but different
    api_keys (and identical cleaned names) must both stay visible in the
    picker — slug is suffixed to disambiguate."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {"name": "OpenAI — key A", "base_url": "https://api.openai.com/v1",
             "api_key": "sk-AAA", "model": "gpt-5.4"},
            {"name": "OpenAI — key B", "base_url": "https://api.openai.com/v1",
             "api_key": "sk-BBB", "model": "gpt-4.6"},
        ],
        max_models=50,
    )

    custom_groups = [p for p in providers if p.get("is_user_defined")]
    assert len(custom_groups) == 2
    slugs = sorted(p["slug"] for p in custom_groups)
    # First group keeps the base slug, second gets a numeric suffix
    assert slugs == ["custom:openai", "custom:openai-2"]
    # Each row has a distinct model
    models = {p["slug"]: p["models"] for p in custom_groups}
    assert models["custom:openai"] == ["gpt-5.4"]
    assert models["custom:openai-2"] == ["gpt-4.6"]


def test_list_authenticated_providers_same_url_different_key_env_and_api_mode_stay_separate(monkeypatch):
    """Same gateway host but different key_env/api_mode entries are distinct providers."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom:gpt",
        current_base_url="https://gateway.example.com",
        user_providers={},
        custom_providers=[
            {
                "name": "gpt",
                "base_url": "https://gateway.example.com",
                "key_env": "GPT_KEY",
                "api_mode": "codex_responses",
                "model": "gpt-5.5",
            },
            {
                "name": "claude",
                "base_url": "https://gateway.example.com",
                "key_env": "CLAUDE_KEY",
                "api_mode": "anthropic_messages",
                "model": "claude-opus-4-8",
            },
        ],
        max_models=50,
    )

    custom = [p for p in providers if p.get("is_user_defined")]
    by_slug = {p["slug"]: p for p in custom}

    assert set(by_slug) == {"custom:gpt", "custom:claude"}
    assert by_slug["custom:gpt"]["models"] == ["gpt-5.5"]
    assert by_slug["custom:claude"]["models"] == ["claude-opus-4-8"]
    assert by_slug["custom:gpt"]["is_current"] is True
    assert by_slug["custom:claude"]["is_current"] is False


def test_list_authenticated_providers_total_models_reflects_grouped_count(monkeypatch):
    """After grouping six entries into one row, total_models must reflect
    the full count, and every grouped model appears in the list."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    entries = [
        {"name": f"Ollama \u2014 Model {i}", "base_url": "http://localhost:11434/v1",
         "api_key": "ollama", "model": f"model-{i}"}
        for i in range(6)
    ]
    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=entries,
        max_models=4,
        probe_custom_providers=False,
    )

    groups = [p for p in providers if p.get("is_user_defined")]
    assert len(groups) == 1
    group = groups[0]
    assert group["total_models"] == 6
    # All six models are preserved in the grouped row.
    assert sorted(group["models"]) == sorted(f"model-{i}" for i in range(6))


def test_lmstudio_picker_probes_active_config_base_url(monkeypatch):
    """When `provider: lmstudio` is saved with a remote base_url and no
    LM_BASE_URL env var, the picker must probe the saved base_url — not
    127.0.0.1. Regression: prior behavior always probed localhost, so users
    with LM Studio on a lab box saw the wrong (or empty) model list.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.delenv("LM_BASE_URL", raising=False)
    monkeypatch.delenv("LM_API_KEY", raising=False)

    captured: dict = {}

    def _fake_fetch(api_key=None, base_url=None, timeout=5.0):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return ["qwen/qwen3-coder-30b"]

    monkeypatch.setattr("hermes_cli.models.fetch_lmstudio_models", _fake_fetch)

    list_authenticated_providers(
        current_provider="lmstudio",
        current_base_url="http://192.168.1.10:1234/v1",
        current_model="qwen/qwen3-coder-30b",
    )

    assert captured["base_url"] == "http://192.168.1.10:1234/v1"


def test_lmstudio_picker_lm_base_url_env_wins_over_active_config(monkeypatch):
    """LM_BASE_URL env var must still take precedence over the saved
    base_url so users can temporarily redirect the picker without editing
    config.yaml.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setenv("LM_BASE_URL", "http://override.local:9999/v1")
    monkeypatch.delenv("LM_API_KEY", raising=False)

    captured: dict = {}

    def _fake_fetch(api_key=None, base_url=None, timeout=5.0):
        captured["base_url"] = base_url
        return []

    monkeypatch.setattr("hermes_cli.models.fetch_lmstudio_models", _fake_fetch)

    list_authenticated_providers(
        current_provider="lmstudio",
        current_base_url="http://192.168.1.10:1234/v1",
    )

    assert captured["base_url"] == "http://override.local:9999/v1"


def test_lmstudio_picker_skips_probe_when_not_configured(monkeypatch):
    """If the user has never configured LM Studio (no LM_API_KEY / LM_BASE_URL
    and not on lmstudio), the picker must not pay the localhost probe cost
    just to discover LM Studio is unavailable.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.delenv("LM_BASE_URL", raising=False)
    monkeypatch.delenv("LM_API_KEY", raising=False)

    captured: dict = {}

    def _fake_fetch(api_key=None, base_url=None, timeout=5.0):
        captured["base_url"] = base_url
        return []

    monkeypatch.setattr("hermes_cli.models.fetch_lmstudio_models", _fake_fetch)

    list_authenticated_providers(
        current_provider="openrouter",
        current_base_url="https://openrouter.ai/api/v1",
    )

    assert "base_url" not in captured


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


