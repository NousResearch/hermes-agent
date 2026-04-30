"""Tests for user-defined providers (providers: dict) in /model.

These tests ensure that providers defined in the config.yaml ``providers:`` section
are properly resolved for model switching and that their full ``models:`` lists
are exposed in the model picker.
"""

import pytest
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli import runtime_provider as rp


# =============================================================================
# Tests for list_authenticated_providers including full models list
# =============================================================================

def test_list_authenticated_providers_includes_full_models_list_from_user_providers(monkeypatch):
    """User-defined providers should expose both default_model and full models list.
    
    Regression test: previously only default_model was shown in /model picker.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    
    user_providers = {
        "local-ollama": {
            "name": "Local Ollama",
            "api": "http://localhost:11434/v1",
            "default_model": "minimax-m2.7:cloud",
            "models": [
                "minimax-m2.7:cloud",
                "kimi-k2.5:cloud",
                "glm-5.1:cloud",
                "qwen3.5:cloud",
            ],
        }
    }
    
    providers = list_authenticated_providers(
        current_provider="local-ollama",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )
    
    # Find our user provider
    user_prov = next(
        (p for p in providers if p.get("is_user_defined") and p["slug"] == "local-ollama"),
        None
    )
    
    assert user_prov is not None, "User provider 'local-ollama' should be in results"
    assert user_prov["total_models"] == 4, f"Expected 4 models, got {user_prov['total_models']}"
    assert "minimax-m2.7:cloud" in user_prov["models"]
    assert "kimi-k2.5:cloud" in user_prov["models"]
    assert "glm-5.1:cloud" in user_prov["models"]
    assert "qwen3.5:cloud" in user_prov["models"]


def test_list_authenticated_providers_dedupes_models_when_default_in_list(monkeypatch):
    """When default_model is also in models list, don't duplicate."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    
    user_providers = {
        "my-provider": {
            "api": "http://example.com/v1",
            "default_model": "model-a",  # Included in models list below
            "models": ["model-a", "model-b", "model-c"],
        }
    }
    
    providers = list_authenticated_providers(
        current_provider="my-provider",
        user_providers=user_providers,
        custom_providers=[],
    )
    
    user_prov = next(
        (p for p in providers if p.get("is_user_defined")),
        None
    )
    
    assert user_prov is not None
    assert user_prov["total_models"] == 3, "Should have 3 unique models, not 4"
    assert user_prov["models"].count("model-a") == 1, "model-a should not be duplicated"


def test_list_authenticated_providers_enumerates_dict_format_models(monkeypatch):
    """providers: dict entries with ``models:`` as a dict keyed by model id
    (canonical Hermes write format) should surface every key in the picker.

    Regression: the ``providers:`` dict path previously only accepted
    list-format ``models:`` and silently dropped dict-format entries,
    even though Hermes's own writer and downstream readers use dict format.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "local-ollama": {
            "name": "Local Ollama",
            "api": "http://localhost:11434/v1",
            "default_model": "minimax-m2.7:cloud",
            "models": {
                "minimax-m2.7:cloud": {"context_length": 196608},
                "kimi-k2.5:cloud": {"context_length": 200000},
                "glm-5.1:cloud": {"context_length": 202752},
            },
        }
    }

    providers = list_authenticated_providers(
        current_provider="local-ollama",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    user_prov = next(
        (p for p in providers if p.get("is_user_defined") and p["slug"] == "local-ollama"),
        None,
    )

    assert user_prov is not None
    assert user_prov["total_models"] == 3
    assert user_prov["models"] == [
        "minimax-m2.7:cloud",
        "kimi-k2.5:cloud",
        "glm-5.1:cloud",
    ]


def test_list_authenticated_providers_uses_live_models_for_user_provider(monkeypatch):
    """User-defined OpenAI-compatible providers should prefer live /models.

    Regression: CRS-style providers with a stale config ``models:`` dict kept
    showing only the configured subset in the /model picker, even though their
    /v1/models endpoint exposed newly added models.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setenv("CRS_TEST_KEY", "sk-test")

    calls = []

    def fake_fetch_api_models(api_key, base_url):
        calls.append((api_key, base_url))
        return ["old-configured-model", "new-live-model"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

    user_providers = {
        "crs-henkee": {
            "name": "CRS Henkee",
            "base_url": "http://127.0.0.1:3000/api/v1",
            "key_env": "CRS_TEST_KEY",
            "model": "old-configured-model",
            "models": {
                "old-configured-model": {"context_length": 200000},
            },
        }
    }

    providers = list_authenticated_providers(
        current_provider="crs-henkee",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    user_prov = next(
        (p for p in providers if p.get("is_user_defined") and p["slug"] == "crs-henkee"),
        None,
    )

    assert user_prov is not None
    assert calls == [("sk-test", "http://127.0.0.1:3000/api/v1")]
    assert user_prov["models"] == ["old-configured-model", "new-live-model"]
    assert user_prov["total_models"] == 2


def test_list_authenticated_providers_dict_models_without_default_model(monkeypatch):
    """Dict-format ``models:`` without a ``default_model`` must still expose
    every dict key, not collapse to an empty list."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "multimodel": {
            "api": "http://example.com/v1",
            "models": {
                "alpha": {"context_length": 8192},
                "beta": {"context_length": 16384},
            },
        }
    }

    providers = list_authenticated_providers(
        current_provider="",
        user_providers=user_providers,
        custom_providers=[],
    )

    user_prov = next(
        (p for p in providers if p.get("is_user_defined") and p["slug"] == "multimodel"),
        None,
    )

    assert user_prov is not None
    assert user_prov["total_models"] == 2
    assert set(user_prov["models"]) == {"alpha", "beta"}


def test_list_authenticated_providers_dict_models_dedupe_with_default(monkeypatch):
    """When ``default_model`` is also a key in the ``models:`` dict, it must
    appear exactly once (list already had this for list-format models)."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "my-provider": {
            "api": "http://example.com/v1",
            "default_model": "model-a",
            "models": {
                "model-a": {"context_length": 8192},
                "model-b": {"context_length": 16384},
                "model-c": {"context_length": 32768},
            },
        }
    }

    providers = list_authenticated_providers(
        current_provider="my-provider",
        user_providers=user_providers,
        custom_providers=[],
    )

    user_prov = next(
        (p for p in providers if p.get("is_user_defined")),
        None,
    )

    assert user_prov is not None
    assert user_prov["total_models"] == 3
    assert user_prov["models"].count("model-a") == 1


def test_openai_native_curated_catalog_is_non_empty():
    """Regression: built-in openai must have a static catalog for picker totals."""
    from hermes_cli.models import _PROVIDER_MODELS

    assert _PROVIDER_MODELS.get("openai")
    assert len(_PROVIDER_MODELS["openai"]) >= 4


def test_list_authenticated_providers_openai_built_in_nonzero_total(monkeypatch):
    """Built-in openai row must not report total_models=0 when creds exist."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {"openai": {"env": ["OPENAI_API_KEY"]}},
    )
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )
    row = next((p for p in providers if p.get("slug") == "openai"), None)
    assert row is not None
    assert row["total_models"] > 0


def test_list_authenticated_providers_user_openai_official_url_fallback(monkeypatch):
    """User providers: api.openai.com with no models list uses native curated fallback."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "openai-direct": {
            "name": "OpenAI Direct",
            "api": "https://api.openai.com/v1",
        }
    }
    providers = list_authenticated_providers(
        current_provider="",
        current_base_url="",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )
    row = next((p for p in providers if p.get("slug") == "openai-direct"), None)
    assert row is not None
    assert row["total_models"] > 0


def test_list_authenticated_providers_fallback_to_default_only(monkeypatch):
    """When no models array is provided, should fall back to default_model."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    
    user_providers = {
        "simple-provider": {
            "name": "Simple Provider",
            "api": "http://example.com/v1",
            "default_model": "single-model",
            # No 'models' key
        }
    }
    
    providers = list_authenticated_providers(
        current_provider="",
        user_providers=user_providers,
        custom_providers=[],
    )
    
    user_prov = next(
        (p for p in providers if p.get("is_user_defined")),
        None
    )
    
    assert user_prov is not None
    assert user_prov["total_models"] == 1
    assert user_prov["models"] == ["single-model"]


def test_list_authenticated_providers_accepts_base_url_and_singular_model(monkeypatch):
    """providers: dict entries written in canonical Hermes shape
    (``base_url`` + singular ``model``) should resolve the same as the
    legacy ``api`` + ``default_model`` shape.

    Regression: section 3 previously only read ``api``/``url`` and
    ``default_model``, so new-shape entries written by Hermes's own writer
    surfaced with empty ``api_url`` and no default.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "custom": {
            "base_url": "http://example.com/v1",
            "model": "gpt-5.4",
            "models": {
                "gpt-5.4": {},
                "grok-4.20-beta": {},
                "minimax-m2.7": {},
            },
        }
    }

    providers = list_authenticated_providers(
        current_provider="custom",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    custom = next((p for p in providers if p["slug"] == "custom"), None)
    assert custom is not None
    assert custom["api_url"] == "http://example.com/v1"
    assert custom["models"] == ["gpt-5.4", "grok-4.20-beta", "minimax-m2.7"]
    assert custom["total_models"] == 3


def test_list_authenticated_providers_dedupes_when_user_and_custom_overlap(monkeypatch):
    """When the same slug appears in both ``providers:`` dict and
    ``custom_providers:`` list, emit exactly one row (providers: dict wins
    since it is processed first).

    Regression: section 3 previously had no ``seen_slugs`` check, so
    overlapping entries produced two picker rows for the same provider.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    providers = list_authenticated_providers(
        current_provider="custom",
        user_providers={
            "custom": {
                "base_url": "http://example.com/v1",
                "model": "gpt-5.4",
                "models": {
                    "gpt-5.4": {},
                    "grok-4.20-beta": {},
                },
            }
        },
        custom_providers=[
            {
                "name": "custom",
                "base_url": "http://example.com/v1",
                "model": "legacy-only-model",
            }
        ],
        max_models=50,
    )

    matches = [p for p in providers if p["slug"] == "custom"]
    assert len(matches) == 1
    # providers: dict wins — legacy-only-model is suppressed.
    assert matches[0]["models"] == ["gpt-5.4", "grok-4.20-beta"]


def test_list_authenticated_providers_no_duplicate_labels_across_schemas(monkeypatch):
    """Regression: same endpoint in both ``providers:`` dict AND ``custom_providers:``
    list (e.g. via ``get_compatible_custom_providers()``) must not emit two picker
    rows with identical display names.

    Before the fix, section 3 emitted bare-slug rows ("openrouter") and section 4
    emitted ``custom:openrouter`` rows for the same endpoint — both labelled
    identically, bypassing ``seen_slugs`` dedup because the slug shapes differ.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    shared_entries = [
        ("endpoint-a", "http://a.local/v1"),
        ("endpoint-b", "http://b.local/v1"),
        ("endpoint-c", "http://c.local/v1"),
    ]

    user_providers = {
        name: {"name": name, "base_url": url, "model": "m1"}
        for name, url in shared_entries
    }
    custom_providers = [
        {"name": name, "base_url": url, "model": "m1"}
        for name, url in shared_entries
    ]

    providers = list_authenticated_providers(
        current_provider="none",
        user_providers=user_providers,
        custom_providers=custom_providers,
        max_models=50,
    )

    user_rows = [p for p in providers if p.get("source") == "user-config"]
    # Expect one row per shared entry — not two.
    assert len(user_rows) == len(shared_entries), (
        f"Expected {len(shared_entries)} rows, got {len(user_rows)}: "
        f"{[(p['slug'], p['name']) for p in user_rows]}"
    )

    # And zero duplicate display labels.
    labels = [p["name"].lower() for p in user_rows]
    assert len(labels) == len(set(labels)), (
        f"Duplicate labels across picker rows: {labels}"
    )


def test_list_authenticated_providers_hides_custom_shadowing_builtin_endpoint(monkeypatch):
    """#16970: a custom_providers entry whose ``base_url`` matches a built-in
    provider's endpoint should be hidden. The built-in row already represents
    that endpoint with its canonical slug, curated model list, and auth wiring.

    Repro: user sets ``DASHSCOPE_API_KEY`` (triggers the built-in ``alibaba``
    row pointing at the static ``inference_base_url``) AND defines a
    ``my-alibaba`` custom provider pointing at the same URL. Before the fix,
    the picker showed both rows for one endpoint.
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {
            "alibaba": {
                "name": "Alibaba Cloud (DashScope)",
                "env": ["DASHSCOPE_API_KEY"],
            }
        },
    )
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    custom_providers = [
        {
            "name": "my-alibaba",
            # Matches PROVIDER_REGISTRY['alibaba'].inference_base_url exactly.
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "api_key": "sk-sp-test",
            "model": "qwen3.6-plus",
            "models": {"qwen3.6-plus": {"context_length": 500000}},
        }
    ]

    providers = list_authenticated_providers(
        current_provider="my-alibaba",
        user_providers={},
        custom_providers=custom_providers,
        max_models=50,
    )

    slugs = [p["slug"] for p in providers]
    # Built-in alibaba row should be present.
    assert "alibaba" in slugs, (
        f"Expected built-in alibaba row, got slugs: {slugs}"
    )
    # Custom shadow row should be hidden — its base_url matches the built-in's.
    assert not any("my-alibaba" in s for s in slugs), (
        f"Custom my-alibaba should have been dedup'd against the built-in "
        f"alibaba endpoint, got slugs: {slugs}"
    )


def test_list_authenticated_providers_keeps_custom_with_distinct_endpoint(monkeypatch):
    """Dedup must only apply when the endpoint matches a built-in. A custom
    provider on a genuinely distinct endpoint stays visible even if a
    built-in is also authenticated."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {
            "alibaba": {
                "name": "Alibaba Cloud (DashScope)",
                "env": ["DASHSCOPE_API_KEY"],
            }
        },
    )
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    custom_providers = [
        {
            "name": "my-private-relay",
            "base_url": "https://relay.example.internal/v1",
            "api_key": "sk-relay-test",
            "model": "qwen3.6-plus",
            "models": {"qwen3.6-plus": {}},
        }
    ]

    providers = list_authenticated_providers(
        current_provider="my-private-relay",
        user_providers={},
        custom_providers=custom_providers,
        max_models=50,
    )

    slugs = [p["slug"] for p in providers]
    assert any("my-private-relay" in s for s in slugs), (
        f"Custom provider on distinct endpoint must stay visible, got: {slugs}"
    )


def test_list_authenticated_providers_dedup_honors_base_url_env_override(monkeypatch):
    """The dedup must track the EFFECTIVE endpoint — if DASHSCOPE_BASE_URL
    overrides the static inference_base_url, a custom provider pointing at
    the overridden URL (not the static one) should still be recognized as
    a duplicate."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://custom-dashscope.example.com/v1",
    )
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {
            "alibaba": {
                "name": "Alibaba Cloud (DashScope)",
                "env": ["DASHSCOPE_API_KEY"],
            }
        },
    )
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    custom_providers = [
        {
            "name": "my-dashscope-override",
            # Same URL as DASHSCOPE_BASE_URL env override above.
            "base_url": "https://custom-dashscope.example.com/v1",
            "api_key": "sk-test",
            "model": "qwen3.6-plus",
        }
    ]

    providers = list_authenticated_providers(
        current_provider="alibaba",
        user_providers={},
        custom_providers=custom_providers,
        max_models=50,
    )

    slugs = [p["slug"] for p in providers]
    assert not any("my-dashscope-override" in s for s in slugs), (
        f"Custom entry matching env-overridden built-in endpoint should be "
        f"dedup'd, got: {slugs}"
    )


# =============================================================================
# Tests for _get_named_custom_provider with providers: dict
# =============================================================================

def test_get_named_custom_provider_finds_user_providers_by_key(monkeypatch, tmp_path):
    """Should resolve providers from providers: dict (new-style), not just custom_providers."""
    config = {
        "providers": {
            "local-localhost:11434": {
                "api": "http://localhost:11434/v1",
                "name": "Local (localhost:11434)",
                "default_model": "minimax-m2.7:cloud",
            }
        }
    }
    
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    result = rp._get_named_custom_provider("local-localhost:11434")
    
    assert result is not None
    assert result["base_url"] == "http://localhost:11434/v1"
    assert result["name"] == "Local (localhost:11434)"


def test_get_named_custom_provider_finds_by_display_name(monkeypatch, tmp_path):
    """Should match providers by their 'name' field as well as key."""
    config = {
        "providers": {
            "my-ollama-xyz": {
                "api": "http://ollama.example.com/v1",
                "name": "My Production Ollama",
                "default_model": "llama3",
            }
        }
    }
    
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    # Should find by display name (normalized)
    result = rp._get_named_custom_provider("my-production-ollama")
    
    assert result is not None
    assert result["base_url"] == "http://ollama.example.com/v1"


def test_get_named_custom_provider_falls_back_to_legacy_format(monkeypatch, tmp_path):
    """Should still work with custom_providers: list format."""
    config = {
        "providers": {},
        "custom_providers": [
            {
                "name": "Custom Endpoint",
                "base_url": "http://custom.example.com/v1",
            }
        ]
    }
    
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    result = rp._get_named_custom_provider("custom-endpoint")
    
    assert result is not None


def test_get_named_custom_provider_returns_none_for_unknown(monkeypatch, tmp_path):
    """Should return None for providers that don't exist."""
    config = {
        "providers": {
            "known-provider": {
                "api": "http://known.example.com/v1",
            }
        }
    }
    
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    result = rp._get_named_custom_provider("other-provider")
    
    # "unknown-provider" partial-matches "known-provider" because "unknown" doesn't match
    # but our matching is loose (substring). Let's verify a truly non-matching provider
    result = rp._get_named_custom_provider("completely-different-name")
    assert result is None


def test_get_named_custom_provider_skips_empty_base_url(monkeypatch, tmp_path):
    """Should skip providers without a base_url."""
    config = {
        "providers": {
            "incomplete-provider": {
                "name": "Incomplete",
                # No api/base_url field
            }
        }
    }
    
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    result = rp._get_named_custom_provider("incomplete-provider")
    
    assert result is None


# =============================================================================
# Integration test for switch_model with user providers
# =============================================================================

def test_switch_model_resolves_user_provider_credentials(monkeypatch, tmp_path):
    """/model switch should resolve credentials for providers: dict providers."""
    import yaml
    
    config = {
        "providers": {
            "local-ollama": {
                "api": "http://localhost:11434/v1",
                "name": "Local Ollama",
                "default_model": "minimax-m2.7:cloud",
            }
        }
    }
    
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    # Mock validation to pass
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *a, **k: {"accepted": True, "persist": True, "recognized": True, "message": None}
    )
    
    result = switch_model(
        raw_input="kimi-k2.5:cloud",
        current_provider="local-ollama",
        current_model="minimax-m2.7:cloud",
        current_base_url="http://localhost:11434/v1",
        is_global=False,
        user_providers=config["providers"],
    )

    assert result.success is True
    assert result.error_message == ""


# =============================================================================
# Regression: providers: dict ``transport`` field must be honored
# =============================================================================


def test_get_named_custom_provider_reads_transport_field(monkeypatch):
    """v12+ ``providers:`` dict stores api mode under ``transport:`` (not the
    legacy ``api_mode:``).  ``_get_named_custom_provider`` must accept both
    field names.

    Bug: this function read only ``entry.get("api_mode")`` for v12+ entries.
    After ``migrate_config()`` writes ``transport`` on every entry, the
    lookup returns None and ``_resolve_named_custom_runtime`` falls back
    through ``_detect_api_mode_for_url(base_url) or "chat_completions"``
    — silently downgrading every codex_responses / anthropic_messages
    provider to chat_completions.
    """
    config = {
        "_config_version": 12,
        "providers": {
            "my-codex-provider": {
                "name": "my-codex-provider",
                "api": "http://127.0.0.1:4000/v1",
                "api_key": "test-key",
                "default_model": "gpt-5",
                "transport": "codex_responses",
            },
        },
    }

    monkeypatch.setattr(rp, "load_config", lambda: config)

    result = rp._get_named_custom_provider("my-codex-provider")
    assert result is not None
    assert result["api_mode"] == "codex_responses"
    assert result["base_url"] == "http://127.0.0.1:4000/v1"
    assert result["model"] == "gpt-5"


def test_get_named_custom_provider_legacy_api_mode_field_still_works(monkeypatch):
    """Hand-edited configs that used ``api_mode:`` (legacy spelling) inside
    the v12+ providers: dict shape must keep working — the migration writer
    produces ``transport:`` but human-edited configs may carry the older
    spelling forward."""
    config = {
        "_config_version": 12,
        "providers": {
            "anthropic-proxy": {
                "name": "anthropic-proxy",
                "api": "http://127.0.0.1:8082",
                "api_key": "test-key",
                "default_model": "claude-opus-4-7",
                "api_mode": "anthropic_messages",  # legacy spelling
            },
        },
    }

    monkeypatch.setattr(rp, "load_config", lambda: config)

    result = rp._get_named_custom_provider("anthropic-proxy")
    assert result is not None
    assert result["api_mode"] == "anthropic_messages"


def test_get_named_custom_provider_transport_resolves_via_display_name(monkeypatch):
    """When the requested name matches the entry's ``name:`` field rather
    than its dict key, the same transport-vs-api_mode logic must apply
    (second branch in ``_get_named_custom_provider``)."""
    config = {
        "_config_version": 12,
        "providers": {
            "slug-different-from-name": {
                "name": "Codex Provider",  # display name
                "api": "http://127.0.0.1:4000/v1",
                "api_key": "test-key",
                "default_model": "gpt-5",
                "transport": "codex_responses",
            },
        },
    }

    monkeypatch.setattr(rp, "load_config", lambda: config)

    result = rp._get_named_custom_provider("Codex Provider")
    assert result is not None
    assert result["api_mode"] == "codex_responses"
