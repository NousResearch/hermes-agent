"""Regression tests for /model support of config.yaml custom_providers.

The terminal `hermes model` flow already exposes `custom_providers`, but the
shared slash-command pipeline (`/model` in CLI/gateway/Telegram) historically
only looked at `providers:`.

Fixtures that set ``discover_models=False`` deliberately opt out of live
endpoint probing; they test saved/configured-provider behavior rather than a
local model server. The live-probe cases set it separately when discovery is
the behavior under test.
"""

import time

import hermes_cli.providers as providers_mod
import pytest
import yaml
from hermes_cli.model_switch import (
    _fetch_picker_live_models,
    _save_discovered_models_to_config,
    list_authenticated_providers,
    switch_model,
)
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
    monkeypatch.setattr(
        "hermes_cli.models.fetch_ollama_local_models", lambda *_a, **_kw: None
    )


def test_picker_native_probe_failure_falls_back_to_openai_catalog(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.models.should_use_ollama_native_catalog", lambda *a, **k: True
    )
    monkeypatch.setattr(
        "hermes_cli.models._get_ollama_native_headers", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_ollama_local_models", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models", lambda *a, **k: ["fallback-model"]
    )

    assert _fetch_picker_live_models(
        "key", "http://127.0.0.1:11434/v1", "ollama", False
    ) == ["fallback-model"]


def test_picker_generic_discovery_preserves_api_mode(monkeypatch):
    calls = []

    def cached(*args, **kwargs):
        calls.append((args, kwargs))
        return ["model-a"]

    monkeypatch.setattr(
        "hermes_cli.models.should_use_ollama_native_catalog", lambda *a, **k: False
    )
    monkeypatch.setattr("hermes_cli.models.cached_fetch_api_models", cached)

    assert _fetch_picker_live_models(
        "key",
        "https://proxy.example/anthropic/v1",
        "custom",
        False,
        api_mode="anthropic_messages",
    ) == ["model-a"]
    assert calls[0][1]["api_mode"] == "anthropic_messages"


def test_list_authenticated_providers_includes_custom_providers(monkeypatch):
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



def test_list_authenticated_providers_numeric_yaml_provider_dict_key(monkeypatch):
    """Unquoted YAML `providers: {2070: ...}` must not 500 the Model tab."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])

    rows = list_authenticated_providers(
        current_provider=2070,
        current_base_url="http://192.168.1.10:8082/v1",
        current_model="Qwen3.5-9B-Q4_K_M.gguf",
        user_providers={
            2070: {
                "name": 2070,
                "base_url": "http://192.168.1.10:8082/v1",
                "model": "Qwen3.5-9B-Q4_K_M.gguf",
            }
        },
        custom_providers=[],
        max_models=0,
        probe_custom_providers=False,
    )

    match = next(p for p in rows if str(p.get("slug")) == "2070")
    assert match["name"] == "2070"
    assert match.get("is_current") is True


def test_list_authenticated_providers_numeric_custom_provider_name(monkeypatch):
    """Legacy custom_providers list with name: 2070 (int) must not .strip() crash."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])

    rows = list_authenticated_providers(
        current_provider=2070,
        current_base_url="http://192.168.1.10:8082/v1",
        current_model="Qwen3.5-9B-Q4_K_M.gguf",
        user_providers={},
        custom_providers=[
            {
                "name": 2070,
                "base_url": "http://192.168.1.10:8082/v1",
                "model": "Qwen3.5-9B-Q4_K_M.gguf",
            }
        ],
        max_models=0,
        probe_custom_providers=False,
    )

    assert any(
        str(p.get("name")) == "2070" or "2070" in str(p.get("slug"))
        for p in rows
    )


def test_providers_singular_model_does_not_suppress_ollama_native_discovery(monkeypatch):
    """A saved selection in ``providers:`` is not an explicit catalog."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.models.fetch_ollama_local_models",
        lambda *a, **k: ["qwen3:latest", "llama3.2:latest"],
    )

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={
            "ollama": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen3:latest",
            }
        },
        custom_providers=[],
        max_models=50,
    )

    ollama = next(p for p in providers if p["slug"] == "ollama")
    assert ollama["models"] == ["qwen3:latest", "llama3.2:latest"]


def test_list_authenticated_providers_can_skip_custom_provider_live_probe(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    fetch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected probe"))
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {
                "name": "Slow Local",
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key": "sk-local",
                "model": "local-model",
            }
        ],
        probe_custom_providers=False,
    )

    row = next(p for p in providers if p["slug"] == "custom:slow-local")
    assert row["models"] == ["local-model"]
    assert row["total_models"] == 1





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


def test_switch_model_does_not_send_ollama_headers_to_unrelated_custom_endpoint(monkeypatch):
    """A custom endpoint must not inherit headers from configured Ollama."""
    seen_headers = []
    validation_headers = []

    def fake_native_detection(provider, base_url, headers=None):
        seen_headers.append(headers)
        return True

    def fake_validation(*args, **kwargs):
        validation_headers.append(kwargs.get("headers"))
        return _MOCK_VALIDATION

    monkeypatch.setattr(
        "hermes_cli.models.should_use_ollama_native_catalog",
        fake_native_detection,
    )
    monkeypatch.setattr(
        "hermes_cli.models._get_ollama_request_headers",
        lambda: {"Authorization": "Bearer configured-ollama-secret"},
    )
    monkeypatch.setattr(
        "hermes_cli.models._get_provider_config_dict",
        lambda provider: (
            {"base_url": "https://trusted-ollama.example:11434"}
            if provider == "ollama"
            else {}
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "custom-key",
            "base_url": "https://attacker.example:11434/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", fake_validation)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="new-model",
        current_provider="custom",
        current_model="old-model",
        current_base_url="https://attacker.example:11434/v1",
        current_api_key="custom-key",
        explicit_provider="",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert seen_headers == [{}]
    assert validation_headers == [None]



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






def test_list_groups_same_name_custom_providers_into_one_row(monkeypatch):
    """Multiple custom_providers entries sharing a name should produce one row
    with all models collected, not N duplicate rows."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])

    providers = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "qwen3-coder:480b-cloud"},
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "glm-5.1:cloud"},
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "kimi-k2.5"},
            {"name": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "minimax-m2.7:cloud"},
            {"name": "Moonshot", "base_url": "https://api.moonshot.ai/v1", "model": "kimi-k2-thinking"},
        ],
        max_models=50,
    )

    ollama_rows = [p for p in providers if p["name"] == "Ollama Cloud"]
    assert len(ollama_rows) == 1, f"Expected 1 Ollama Cloud row, got {len(ollama_rows)}"
    assert ollama_rows[0]["models"] == [
        "qwen3-coder:480b-cloud", "glm-5.1:cloud", "kimi-k2.5", "minimax-m2.7:cloud"
    ]
    assert ollama_rows[0]["total_models"] == 4

    moonshot_rows = [p for p in providers if p["name"] == "Moonshot"]
    assert len(moonshot_rows) == 1
    assert moonshot_rows[0]["models"] == ["kimi-k2-thinking"]


def test_list_deduplicates_same_model_in_group(monkeypatch):
    """Duplicate model entries under the same provider name should not produce
    duplicate entries in the models list."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])

    providers = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={},
        custom_providers=[
            {"name": "MyProvider", "base_url": "http://localhost:11434/v1", "model": "llama3", "discover_models": False},
            {"name": "MyProvider", "base_url": "http://localhost:11434/v1", "model": "llama3", "discover_models": False},
            {"name": "MyProvider", "base_url": "http://localhost:11434/v1", "model": "mistral", "discover_models": False},
        ],
        max_models=50,
    )

    my_rows = [p for p in providers if p["name"] == "MyProvider"]
    assert len(my_rows) == 1
    assert my_rows[0]["models"] == ["llama3", "mistral"]
    assert my_rows[0]["total_models"] == 2


def test_custom_provider_no_key_singular_model_still_probes_live_models(monkeypatch):
    """A singular ``model:`` is the active selection, not an explicit catalog.

    No-key local OpenAI-compatible endpoints such as llama.cpp should still be
    probed so /model matches the terminal ``hermes model`` flow. Ollama-native
    discovery is covered separately with a fake ``/api/tags`` server.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    calls = []

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["llama3", "mistral", "qwen3-coder"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                # Keep this generic and off Ollama's default :11434: this case
                # covers /v1/models probing, not native /api/tags discovery.
                "name": "Local llama.cpp",
                "base_url": "http://localhost:8080/v1",
                "model": "llama3",
            }
        ],
        max_models=50,
    )

    assert calls == [
        ("", "http://localhost:8080/v1", {
            "timeout": 5.0,
            "api_mode": None,
            "headers": None,
        })
    ]
    row = next(p for p in providers if p["name"] == "Local llama.cpp")
    assert row["models"] == ["llama3", "mistral", "qwen3-coder"]
    assert row["total_models"] == 3


def test_custom_provider_model_metadata_dict_still_probes(monkeypatch):
    """Dict-shaped ``models:`` is metadata, not an explicit allowlist."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return ["unexpected-live-model"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:local-ollama",
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "base_url": "http://localhost:11434/v1",
                "model": "llama3",
                "models": {"llama3": {}},
            }
        ],
    )

    row = next(p for p in providers if p["name"] == "Local Ollama")
    assert calls == [
        (
            ("", "http://localhost:11434/v1"),
            {"timeout": 5.0, "api_mode": None, "headers": None},
        )
    ]
    assert row["models"] == ["unexpected-live-model"]


def test_custom_provider_group_explicit_duplicate_skips_probe(monkeypatch):
    """A later grouped entry can explicitly narrow to an existing model."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return ["unexpected-live-model"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:local-ollama",
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "base_url": "http://localhost:11434/v1",
                "model": "llama3",
            },
            {
                "name": "Local Ollama",
                "base_url": "http://localhost:11434/v1",
                "models": ["llama3"],
            },
        ],
    )

    row = next(p for p in providers if p["name"] == "Local Ollama")
    assert calls == []
    assert row["models"] == ["llama3"]


def test_custom_provider_current_only_probe_respects_explicit_catalog(monkeypatch):
    """Normal GUI opens probe only the active singular-only provider."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["live-a", "live-b"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:active",
        current_base_url="http://active.local/v1",
        user_providers={},
        custom_providers=[
            {
                "name": "Active",
                "base_url": "http://active.local/v1",
                "model": "seed",
            },
            {
                "name": "Offline",
                "base_url": "http://offline.local/v1",
                "model": "offline-seed",
            },
            {
                "name": "Static",
                "base_url": "http://static.local/v1",
                "model": "only",
                "models": ["only"],
            },
        ],
        probe_custom_providers=False,
        probe_current_custom_provider=True,
    )

    assert calls == [
        (
            "",
            "http://active.local/v1",
            {"timeout": 5.0, "api_mode": None, "headers": None},
        )
    ]
    rows = {row["name"]: row for row in providers if row.get("is_user_defined")}
    assert rows["Active"]["models"] == ["live-a", "live-b"]
    assert rows["Offline"]["models"] == ["offline-seed"]
    assert rows["Static"]["models"] == ["only"]


def test_custom_provider_current_explicit_catalog_skips_probe(monkeypatch):
    """Current-only GUI probing must still honor an explicit catalog."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return ["unexpected-live-model"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:static",
        current_base_url="http://static.local/v1",
        user_providers={},
        custom_providers=[
            {
                "name": "Static",
                "base_url": "http://static.local/v1",
                "model": "only",
                "models": ["only"],
            }
        ],
        probe_custom_providers=False,
        probe_current_custom_provider=True,
    )

    assert calls == []
    row = next(p for p in providers if p["name"] == "Static")
    assert row["is_current"] is True
    assert row["models"] == ["only"]


def test_custom_provider_empty_explicit_list_allows_probe(monkeypatch):
    """An empty ``models:`` declaration is not an explicit catalog."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["live-a", "live-b"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:local",
        user_providers={},
        custom_providers=[
            {
                "name": "Local",
                "base_url": "http://local.test/v1",
                "model": "seed",
                "models": [],
            }
        ],
    )

    assert calls == [
        (
            "",
            "http://local.test/v1",
            {"timeout": 5.0, "api_mode": None, "headers": None},
        )
    ]
    row = next(p for p in providers if p["name"] == "Local")
    assert row["models"] == ["live-a", "live-b"]


def test_list_enumerates_dict_format_models_alongside_default(monkeypatch):
    """custom_providers entry with dict-format ``models:`` plus singular
    ``model:`` should surface the default and every dict key.

    Regression: Hermes's own writer stores configured models as a dict
    keyed by model id, but the /model picker previously only honored the
    singular ``model:`` field, so multi-model custom providers appeared
    to have only the active model.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "api_mode": "chat_completions",
                "model": "deepseek-chat",
                "models": {
                    "deepseek-chat": {"context_length": 128000},
                    "deepseek-reasoner": {"context_length": 128000},
                },
            }
        ],
        max_models=50,
    )

    ds_rows = [p for p in providers if p["name"] == "DeepSeek"]
    assert len(ds_rows) == 1
    assert ds_rows[0]["models"] == ["deepseek-chat", "deepseek-reasoner"]
    assert ds_rows[0]["total_models"] == 2


def test_list_enumerates_dict_format_models_without_singular_model(monkeypatch):
    """Dict-format ``models:`` with no singular ``model:`` should still
    enumerate every dict key (previously the picker reported 0 models)."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "Thor",
                "base_url": "http://thor.lab:8337/v1",
                "models": {
                    "gemma-4-26B-A4B-it-MXFP4_MOE": {"context_length": 262144},
                    "Qwen3.5-35B-A3B-MXFP4_MOE": {"context_length": 262144},
                    "gemma-4-31B-it-Q4_K_M": {"context_length": 262144},
                },
            }
        ],
        max_models=50,
    )

    thor_rows = [p for p in providers if p["name"] == "Thor"]
    assert len(thor_rows) == 1
    assert set(thor_rows[0]["models"]) == {
        "gemma-4-26B-A4B-it-MXFP4_MOE",
        "Qwen3.5-35B-A3B-MXFP4_MOE",
        "gemma-4-31B-it-Q4_K_M",
    }
    assert thor_rows[0]["total_models"] == 3


def test_list_dedupes_dict_model_matching_singular_default(monkeypatch):
    """When the singular ``model:`` is also a key in the ``models:`` dict,
    it must appear exactly once in the picker."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        user_providers={},
        custom_providers=[
            {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "models": {
                    "deepseek-chat": {"context_length": 128000},
                    "deepseek-reasoner": {"context_length": 128000},
                },
            }
        ],
        max_models=50,
    )

    ds_rows = [p for p in providers if p["name"] == "DeepSeek"]
    assert ds_rows[0]["models"].count("deepseek-chat") == 1
    assert ds_rows[0]["models"] == ["deepseek-chat", "deepseek-reasoner"]





# ─────────────────────────────────────────────────────────────────────────────
# #9210: group custom_providers by (base_url, api_key) in /model picker
# ─────────────────────────────────────────────────────────────────────────────


def test_list_authenticated_providers_groups_same_endpoint(monkeypatch):
    """Multiple custom_providers entries sharing a base_url+api_key must be
    returned as a single picker row with all their models merged."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://localhost:11434/v1",
        user_providers={},
        custom_providers=[
            {"name": "Ollama — MiniMax M2.7", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "minimax-m2.7", "discover_models": False},
            {"name": "Ollama — GLM 5.1",      "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "glm-5.1", "discover_models": False},
            {"name": "Ollama — Qwen3-coder", "base_url": "http://localhost:11434/v1",
             "api_key": "ollama", "model": "qwen3-coder", "discover_models": False},
        ],
        max_models=50,
        probe_custom_providers=False,
    )

    custom_groups = [p for p in providers if p.get("is_user_defined")]
    assert len(custom_groups) == 1, (
        "Expected 1 group for shared endpoint, got "
        f"{[p['slug'] for p in custom_groups]}"
    )
    group = custom_groups[0]
    assert set(group["models"]) == {"minimax-m2.7", "glm-5.1", "qwen3-coder"}
    assert group["total_models"] == 3
    # Per-model suffix stripped from display name
    assert group["name"] == "Ollama"


def test_list_authenticated_providers_current_endpoint_uses_current_slug(monkeypatch):
    """When current_base_url matches the grouped endpoint, the slug must
    equal current_provider so picker selection routes through the live
    credential pipeline — provided current_provider is a real slug, not
    the corrupt bare "custom" (see #17478)."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom:ollama",
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
    assert group["slug"] == "custom:ollama"
    assert group["is_current"] is True


def test_picker_endpoint_authorization_overrides_inferred_bearer(monkeypatch):
    from hermes_cli.model_switch import _fetch_picker_live_models

    captured: dict[str, str] = {}

    def fake_native(_url, *, timeout, headers):
        captured.update(headers or {})
        return ["model-a"]

    monkeypatch.setattr("hermes_cli.models.should_use_ollama_native_catalog", lambda *a, **k: True)
    monkeypatch.setattr("hermes_cli.models.fetch_ollama_local_models", fake_native)
    result = _fetch_picker_live_models(
        "endpoint-key",
        "http://127.0.0.1:11434/v1",
        "ollama",
        False,
        headers={"authorization": "Api-Key explicit"},
    )

    assert result == ["model-a"]
    auth_headers = {
        key: value
        for key, value in captured.items()
        if key.lower() == "authorization"
    }
    assert auth_headers == {"authorization": "Api-Key explicit"}


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
        (
            "sk-gateway-key",
            "https://gateway.example.com/v1",
            {"timeout": 5.0, "api_mode": None, "headers": None},
        )
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
        lambda api_url, model_ids, **kwargs: save_calls.append((api_url, model_ids)),
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


def test_model_flow_named_custom_persists_discovered_models(monkeypatch):
    """The ``hermes model`` named-custom-provider flow persists the discovered
    catalog back to the entry's ``models:`` list.

    No-probe surfaces (dashboard, desktop, ACP) call
    ``build_models_payload(..., probe_custom_providers=False)`` and only show
    the configured ``models:`` list. The CLI flow probes and shows the full
    catalog but (before this fix) never saved it, so a provider added via
    ``hermes model`` collapsed to the single ``model:`` default everywhere but
    the CLI. It must persist discovered models the same way the picker path in
    ``_save_discovered_models_to_config`` does.
    """
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda api_key, base_url, **kw: [
            "discovered-a",
            "discovered-b",
            "discovered-c",
        ],
    )
    # Non-interactive model selection.
    monkeypatch.setattr(
        "hermes_cli.curses_ui.curses_radiolist", lambda *a, **k: 0
    )
    # No-op downstream writes so the test never touches a real config.
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {}, "providers": {}, "custom_providers": []},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

    save_calls = []
    monkeypatch.setattr(
        "hermes_cli.model_switch._save_discovered_models_to_config",
        lambda api_url, model_ids, **kwargs: save_calls.append(
            (api_url, model_ids, kwargs)
        ),
    )

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    _model_flow_named_custom(
        {},
        {
            "name": "Dragomes",
            "base_url": "http://example.com/v1",
            "api_mode": "anthropic_messages",
            "extra_headers": {"X-Tenant": "dragomes"},
            "api_key": "sk-test",
            "key_env": "",
            "model": "MiniMax-M3",
            "provider_key": "",
            "discover_models": True,
            "models": {},
        },
    )

    assert save_calls == [
        (
            "http://example.com/v1",
            ["discovered-a", "discovered-b", "discovered-c"],
            {
                "api_mode": "anthropic_messages",
                "headers": {"X-Tenant": "dragomes"},
            },
        )
    ], (
        "_model_flow_named_custom must persist each live catalog with its "
        "base URL, API mode, and endpoint headers"
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


def test_custom_provider_context_length_models_dict_still_probes(monkeypatch):
    """Dict-shaped ``models:`` from ``_save_custom_provider`` is metadata.

    ``hermes model`` writes ``models: {default: {context_length: N}}`` for
    local Ollama. That must not suppress live /v1/models discovery — otherwise
    Desktop/Telegram only show the saved default and Refresh does nothing.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
        return ["qwen3.6:35b-mlx", "gemma4:31b", "llama3"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:local-ollama",
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "base_url": "http://localhost:11434/v1",
                "model": "qwen3.6:35b-mlx",
                "models": {"qwen3.6:35b-mlx": {"context_length": 32768}},
            }
        ],
        # GUI picker path: probe current custom provider only.
        probe_custom_providers=False,
        probe_current_custom_provider=True,
        current_base_url="http://localhost:11434/v1",
    )

    assert len(calls) == 1
    assert calls[0][0] == ""
    assert calls[0][1] == "http://localhost:11434/v1"
    row = next(p for p in providers if p["name"] == "Local Ollama")
    assert row["models"] == ["qwen3.6:35b-mlx", "gemma4:31b", "llama3"]
    assert row["total_models"] == 3


def test_custom_provider_dict_models_pin_requires_discover_false(monkeypatch):
    """Dict-shaped catalogs pin only when ``discover_models: false``."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return ["unexpected-live-model"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="custom:local-ollama",
        user_providers={},
        custom_providers=[
            {
                "name": "Local Ollama",
                "base_url": "http://localhost:11434/v1",
                "model": "llama3",
                "models": {"llama3": {}},
                "discover_models": False,
            }
        ],
    )

    row = next(p for p in providers if p["name"] == "Local Ollama")
    assert calls == []
    assert row["models"] == ["llama3"]


# ─── No-probe picker opens still serve the cached catalog ───────────────
#
# #58183 stopped GUI picker opens from live-probing saved custom endpoints so
# a stopped local server could not stall the picker. It skipped the cached
# read along with the network one, so a non-current endpoint collapsed to the
# one model named in config even with a full catalog already on disk. These
# pin both halves: the cache is served, the network is not touched.


_LOCAL_ENDPOINT = "http://127.0.0.1:8000/v1"
_LOCAL_CATALOG = [f"omlx-model-{i}" for i in range(1, 9)]
_SHARED_PROXY_URL = "https://proxy.example.com/v1"


def _seed_custom_model_cache(monkeypatch, models, *, age_seconds=10):
    """Put *models* on disk for ``_LOCAL_ENDPOINT`` under the no-credential
    fingerprint the picker probes local endpoints with."""
    import hermes_cli.models as models_mod

    fp = models_mod._custom_endpoint_fingerprint("", None, None)
    cache = {
        f"custom:{_LOCAL_ENDPOINT}": {
            "fp": fp,
            "at": time.time() - age_seconds,
            "models": list(models),
        }
    }
    monkeypatch.setattr(models_mod, "_load_provider_models_cache", lambda: cache)


def _no_probe_local_row(monkeypatch, *, custom_providers=None, user_providers=None,
                        current_provider="nous", **kwargs):
    """Run the GUI picker path (no live probing) and return the local row
    plus every base_url a live fetch was attempted against."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    fetched = []

    def fetch(_api_key, base_url, **_kwargs):
        fetched.append(base_url)
        return ["should-not-be-reached"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider=current_provider,
        user_providers=user_providers or {},
        custom_providers=custom_providers or [],
        for_picker=True,
        refresh=False,
        probe_custom_providers=False,
        probe_current_custom_provider=True,
        **kwargs,
    )
    row = next(
        (p for p in providers if _LOCAL_ENDPOINT in str(p.get("api_url", ""))), None
    )
    return row, fetched


def test_no_probe_open_serves_cached_catalog_for_custom_provider(monkeypatch):
    """A ``custom_providers`` endpoint that is not the current provider still
    shows its full discovered catalog, from cache, with no network call."""
    _seed_custom_model_cache(monkeypatch, _LOCAL_CATALOG)

    row, fetched = _no_probe_local_row(
        monkeypatch,
        custom_providers=[
            {
                "name": "Local (127.0.0.1:8000)",
                "base_url": _LOCAL_ENDPOINT,
                "model": "omlx-model-1",
            }
        ],
    )

    assert row is not None
    assert row["is_current"] is False
    assert row["models"] == _LOCAL_CATALOG
    assert row["total_models"] == len(_LOCAL_CATALOG)
    assert fetched == []


def test_no_probe_open_serves_cached_catalog_for_user_provider(monkeypatch):
    """Same contract for a ``providers:`` entry (section 3)."""
    _seed_custom_model_cache(monkeypatch, _LOCAL_CATALOG)

    row, fetched = _no_probe_local_row(
        monkeypatch,
        user_providers={
            "local-8000": {
                "name": "Local (127.0.0.1:8000)",
                "base_url": _LOCAL_ENDPOINT,
                "default_model": "omlx-model-1",
            }
        },
    )

    assert row is not None
    assert row["models"] == _LOCAL_CATALOG
    assert fetched == []


def test_no_probe_open_serves_cached_catalog_for_bare_custom_endpoint(monkeypatch):
    """Same contract for the bare ``provider: custom`` shape (section 3b),
    where the fallback would otherwise be the single active model."""
    _seed_custom_model_cache(monkeypatch, _LOCAL_CATALOG)

    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    fetched = []
    monkeypatch.setattr(
        "hermes_cli.models.fetch_api_models",
        lambda _k, base_url, **_kw: (fetched.append(base_url), None)[1],
    )

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url=_LOCAL_ENDPOINT,
        current_model="omlx-model-1",
        user_providers={},
        custom_providers=[],
        for_picker=True,
        refresh=False,
        probe_custom_providers=False,
        probe_current_custom_provider=False,
    )

    row = next(p for p in providers if p["slug"] == "custom")
    assert row["models"] == _LOCAL_CATALOG
    assert fetched == []


def test_no_probe_open_without_cache_keeps_configured_models_and_stays_offline(
    monkeypatch,
):
    """The #58183 guarantee: a cold cache must not trigger a live probe. The
    row degrades to its configured list rather than stalling on a dead port."""
    _seed_custom_model_cache(monkeypatch, [], age_seconds=10)

    row, fetched = _no_probe_local_row(
        monkeypatch,
        custom_providers=[
            {
                "name": "Local (127.0.0.1:8000)",
                "base_url": _LOCAL_ENDPOINT,
                "model": "omlx-model-1",
            }
        ],
    )

    assert row is not None
    assert row["models"] == ["omlx-model-1"]
    assert fetched == []


def test_no_probe_open_respects_discover_models_false(monkeypatch):
    """A user who pinned their catalog must not have it replaced from cache."""
    _seed_custom_model_cache(monkeypatch, _LOCAL_CATALOG)

    row, fetched = _no_probe_local_row(
        monkeypatch,
        custom_providers=[
            {
                "name": "Local (127.0.0.1:8000)",
                "base_url": _LOCAL_ENDPOINT,
                "model": "pinned-model",
                "models": ["pinned-model"],
                "discover_models": False,
            }
        ],
    )

    assert row is not None
    assert row["models"] == ["pinned-model"]
    assert fetched == []


def test_cached_catalog_is_not_written_back_to_config(monkeypatch):
    """Only a real probe persists discovered models; a cache hit is already
    the product of the probe that saved it."""
    _seed_custom_model_cache(monkeypatch, _LOCAL_CATALOG)
    saves = []
    monkeypatch.setattr(
        "hermes_cli.model_switch._save_discovered_models_to_config",
        lambda api_url, model_ids, **kwargs: saves.append((api_url, model_ids)),
    )

    row, _ = _no_probe_local_row(
        monkeypatch,
        custom_providers=[
            {
                "name": "Local (127.0.0.1:8000)",
                "base_url": _LOCAL_ENDPOINT,
                "model": "omlx-model-1",
            }
        ],
    )

    assert row["models"] == _LOCAL_CATALOG
    assert saves == []


def test_keyless_endpoint_with_saved_catalog_still_reads_cache(monkeypatch):
    """A keyless local server must not be pinned by Hermes' own auto-save.

    ``_save_discovered_models_to_config()`` writes a plain list into
    ``models:``, which ``_models_config_is_allowlist()`` reads back as an
    explicit allowlist. Combined with the no-key discovery gate, a keyless
    endpoint (the common local-model-server shape) froze on the catalog of
    its first probe and could never widen again — the exact "lineup changes
    after config was written" case. The cache read must not be subject to the
    probe's network-cost gate.
    """
    _seed_custom_model_cache(monkeypatch, _LOCAL_CATALOG)

    row, fetched = _no_probe_local_row(
        monkeypatch,
        custom_providers=[
            {
                "name": "Local (127.0.0.1:8000)",
                "base_url": _LOCAL_ENDPOINT,
                "model": "omlx-model-1",
                # No api_key, and a models: list of the shape our own
                # auto-save writes after a successful probe.
                "models": ["omlx-model-1"],
            }
        ],
    )

    assert row is not None
    assert row["models"] == _LOCAL_CATALOG
    assert fetched == []


def test_keyless_endpoint_with_saved_catalog_is_still_not_probed(monkeypatch):
    """...but the network-cost gate it rides on must survive intact.

    The no-key + declared-catalog combination exists to keep Hermes from
    probing an endpoint it cannot authenticate to. Serving that endpoint from
    a warm cache is free; hitting the network is not. With a cold cache and
    live probing fully enabled, this row must still make zero fetches.
    """
    _seed_custom_model_cache(monkeypatch, [])  # cold: only a probe could answer
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    fetched = []

    def fetch(_api_key, base_url, **_kwargs):
        fetched.append(base_url)
        return ["should-not-be-reached"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    providers = list_authenticated_providers(
        current_provider="nous",
        user_providers={},
        custom_providers=[
            {
                "name": "Local (127.0.0.1:8000)",
                "base_url": _LOCAL_ENDPOINT,
                "model": "omlx-model-1",
                "models": ["omlx-model-1"],
            }
        ],
        for_picker=True,
        refresh=False,
        probe_custom_providers=True,  # live probing fully enabled
    )
    row = next(
        (p for p in providers if _LOCAL_ENDPOINT in str(p.get("api_url", ""))), None
    )

    assert row is not None
    assert row["models"] == ["omlx-model-1"]
    assert fetched == []


def test_api_mode_rows_do_not_share_a_cached_catalog(monkeypatch):
    """Two rows differing only by ``api_mode`` must not share a cache entry.

    ``api_mode`` selects the wire protocol — ``x-api-key`` +
    ``anthropic-version`` versus ``Authorization: Bearer`` — so it is part of
    both the picker's group identity and
    ``_custom_endpoint_fingerprint()``. The cache read has to pass it through
    or an ``anthropic_messages`` row renders whatever the OpenAI-mode row
    cached against the same base_url.
    """
    import hermes_cli.models as models_mod

    openai_catalog = ["gpt-oss-a", "gpt-oss-b"]
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    fetched = []

    def fetch(_api_key, base_url, **_kwargs):
        fetched.append(base_url)
        return ["should-not-be-reached"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fetch)

    # Only the OpenAI-mode probe (api_mode=None) is on disk.
    fp = models_mod._custom_endpoint_fingerprint("sk-shared", None, None)
    cache = {
        f"custom:{_SHARED_PROXY_URL}": {
            "fp": fp,
            "at": time.time() - 10,
            "models": list(openai_catalog),
        }
    }
    monkeypatch.setattr(models_mod, "_load_provider_models_cache", lambda: cache)

    def _row(entry):
        providers = list_authenticated_providers(
            current_provider="nous",
            user_providers={},
            custom_providers=[entry],
            for_picker=True,
            refresh=False,
            probe_custom_providers=False,
            probe_current_custom_provider=True,
        )
        return next(
            (p for p in providers if _SHARED_PROXY_URL in str(p.get("api_url", ""))),
            None,
        )

    anthropic_row = _row(
        {
            "name": "Proxy Anthropic",
            "base_url": _SHARED_PROXY_URL,
            "api_key": "sk-shared",
            "api_mode": "anthropic_messages",
            "model": "claude-via-proxy",
        }
    )
    openai_row = _row(
        {
            "name": "Proxy OpenAI",
            "base_url": _SHARED_PROXY_URL,
            "api_key": "sk-shared",
            "model": "gpt-via-proxy",
        }
    )

    assert anthropic_row is not None and openai_row is not None
    assert anthropic_row["models"] == ["claude-via-proxy"], (
        "an anthropic_messages row must not render the OpenAI-mode catalog "
        "cached against the same base_url"
    )
    # ...while the row the entry actually belongs to still resolves.
    assert openai_row["models"] == openai_catalog
    assert fetched == []


def test_auto_saved_catalog_round_trips_without_pinning(tmp_path, monkeypatch):
    """End-to-end: the shape we persist must not read back as a user pin.

    Guards the whole chain rather than one branch — probe saves a catalog,
    config is reloaded, and the endpoint must still be discoverable. If a
    future change makes the saved shape look like an intentional allowlist
    again, this fails even if the gate logic above is refactored away.
    """
    import hermes_cli.config as config_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "custom_providers:\n"
        f"  - name: Local MLX\n    base_url: {_LOCAL_ENDPOINT}\n"
        "    model: omlx-model-1\n"
    )
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(cfg_path), raising=False)

    _save_discovered_models_to_config(_LOCAL_ENDPOINT, list(_LOCAL_CATALOG))

    saved = yaml.safe_load(cfg_path.read_text())["custom_providers"][0]
    assert saved["models_discovered"] is True
    assert list(saved["models"]) == _LOCAL_CATALOG
    assert not any(m.startswith("__") for m in saved["models"]), (
        "sentinel keys must never appear inside the user-facing models mapping"
    )

    # The persisted shape is what the picker will read on the next open. It
    # must not, on a keyless entry, suppress discovery of a wider catalog.
    _seed_custom_model_cache(monkeypatch, [*_LOCAL_CATALOG, "omlx-model-9"])
    row, fetched = _no_probe_local_row(
        monkeypatch, custom_providers=[saved]
    )

    assert row is not None
    assert row["models"] == [*_LOCAL_CATALOG, "omlx-model-9"], (
        "an auto-saved catalog must not pin the endpoint against a newer "
        "cached lineup"
    )
    assert fetched == []


def test_legacy_sentinel_catalog_still_resolves_and_migrates(tmp_path, monkeypatch):
    """Old-shape configs (sentinels inside ``models``) keep working.

    Pre-fix Hermes wrote ``__discovered_model_catalog__: true`` (and
    ``__explicit_model_allowlist__``) inside the user-facing ``models``
    mapping. Reading such a config must (a) recognize the catalog as
    discovered — not a user pin, (b) never list the sentinels as model IDs,
    and (c) migrate to the clean entry-level ``models_discovered`` shape on
    the next discovery save.
    """
    import hermes_cli.config as config_mod
    from hermes_cli.model_switch import (
        _declared_model_ids,
        _entry_models_discovered,
        _models_config_is_allowlist,
    )

    legacy_entry = {
        "name": "Local MLX",
        "base_url": _LOCAL_ENDPOINT,
        "model": "omlx-model-1",
        "models": {
            "__discovered_model_catalog__": True,
            **{m: {} for m in _LOCAL_CATALOG},
        },
    }

    # (a) recognized as a discovered catalog, not an allowlist.
    assert _entry_models_discovered(legacy_entry) is True
    assert not _models_config_is_allowlist(
        legacy_entry["models"], _entry_models_discovered(legacy_entry)
    )

    # (b) sentinels never surface as model IDs.
    assert _declared_model_ids(legacy_entry["models"]) == _LOCAL_CATALOG
    normalized = config_mod._normalize_custom_provider_entry(dict(legacy_entry))
    assert normalized is not None
    assert normalized["models_discovered"] is True
    assert list(normalized["models"]) == _LOCAL_CATALOG
    assert not any(m.startswith("__") for m in normalized["models"])

    # ...and the picker row built from the legacy entry lists no phantoms.
    _seed_custom_model_cache(monkeypatch, [])
    row, fetched = _no_probe_local_row(
        monkeypatch, custom_providers=[legacy_entry]
    )
    assert row is not None
    assert not any(str(m).startswith("__") for m in row["models"])
    assert fetched == []

    # (c) the next discovery save rewrites to the clean shape.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"custom_providers": [legacy_entry]})
    )
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(cfg_path), raising=False)

    _save_discovered_models_to_config(_LOCAL_ENDPOINT, list(_LOCAL_CATALOG))

    saved = yaml.safe_load(cfg_path.read_text())["custom_providers"][0]
    assert saved["models_discovered"] is True
    assert list(saved["models"]) == _LOCAL_CATALOG
    assert "__discovered_model_catalog__" not in saved["models"]
    assert "__explicit_model_allowlist__" not in saved["models"]
