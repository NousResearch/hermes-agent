"""Tests for OpenRouter provider deduplication in the /model picker (#92430).

When OpenRouter is configured in config.yaml under ``providers:``, gateway and CLI
flows call ``get_compatible_custom_providers(cfg)`` which materializes
a custom_providers entry for OpenRouter. When passed to ``list_authenticated_providers``
or ``list_picker_providers``, Section 1 emits the canonical "openrouter" entry, and
Section 4 must NOT emit a duplicate "custom:openrouter" entry.
"""

import pytest
from hermes_cli import model_switch
from hermes_cli.config import get_compatible_custom_providers


@pytest.fixture(autouse=True)
def _disable_live_custom_provider_model_probe(monkeypatch):
    """Keep custom-provider picker fixtures independent of local model servers."""
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)


def test_openrouter_in_providers_config_not_duplicated_in_picker(monkeypatch):
    """Configuring openrouter in providers: should yield exactly one picker row (#92430)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {
        "openrouter": {"name": "OpenRouter", "env": ["OPENROUTER_API_KEY"]},
    })
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {
        "openrouter": "openrouter",
    })
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models", lambda *a, **kw: [("openrouter/auto", "Auto")])
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda slug: ["openrouter/auto"])

    cfg = {
        "providers": {
            "openrouter": {
                "name": "OpenRouter",
                "api": "https://openrouter.ai/api/v1",
                "key_env": "OPENROUTER_API_KEY",
                "transport": "chat_completions",
                "default_model": "openrouter/auto",
                "discover_models": False,
                "models": [],
            }
        }
    }

    user_provs = cfg.get("providers")
    custom_provs = get_compatible_custom_providers(cfg)

    rows = model_switch.list_picker_providers(
        user_providers=user_provs,
        custom_providers=custom_provs,
        max_models=50,
    )

    openrouter_rows = [
        r for r in rows
        if r.get("slug") == "openrouter" or "openrouter" in str(r.get("slug", "")).lower() or r.get("name") == "OpenRouter"
    ]

    assert len(openrouter_rows) == 1, (
        f"Expected exactly 1 OpenRouter row in picker, got {len(openrouter_rows)}: "
        f"{[r.get('slug') for r in openrouter_rows]}"
    )
    assert openrouter_rows[0]["slug"] == "openrouter"


def test_openrouter_in_custom_providers_list_deduplicated(monkeypatch):
    """Directly passing openrouter in custom_providers must not create duplicate (#92430)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {
        "openrouter": {"name": "OpenRouter", "env": ["OPENROUTER_API_KEY"]},
    })
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {
        "openrouter": "openrouter",
    })
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda slug: ["openrouter/auto"])

    custom_provs = [
        {
            "name": "OpenRouter",
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY",
            "model": "openrouter/auto",
            "discover_models": False,
        }
    ]

    rows = model_switch.list_authenticated_providers(
        custom_providers=custom_provs,
        max_models=50,
    )

    slugs = [r["slug"] for r in rows]
    openrouter_slugs = [s for s in slugs if "openrouter" in s.lower()]

    assert openrouter_slugs == ["openrouter"], (
        f"Expected only ['openrouter'], got: {openrouter_slugs}"
    )


def test_openrouter_custom_display_name_deduplicated_by_key_and_url(monkeypatch):
    """Non-canonical display name with OpenRouter provider_key/base_url is deduplicated."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {
        "openrouter": {"name": "OpenRouter", "env": ["OPENROUTER_API_KEY"]},
    })
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {
        "openrouter": "openrouter",
    })
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda slug: ["openrouter/auto"])

    custom_provs = [
        {
            "name": "My Custom Aggregator",
            "provider_key": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY",
            "model": "openrouter/auto",
            "discover_models": False,
        },
        {
            "name": "Another Router",
            "base_url": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY",
            "model": "openrouter/auto",
            "discover_models": False,
        },
    ]

    rows = model_switch.list_authenticated_providers(
        custom_providers=custom_provs,
        max_models=50,
    )

    slugs = [r["slug"] for r in rows]
    openrouter_slugs = [s for s in slugs if "openrouter" in s.lower()]

    assert openrouter_slugs == ["openrouter"], (
        f"Expected only ['openrouter'], got: {slugs}"
    )


def test_distinct_non_canonical_custom_provider_preserved(monkeypatch):
    """Legitimately distinct custom endpoints are not suppressed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {
        "openrouter": {"name": "OpenRouter", "env": ["OPENROUTER_API_KEY"]},
    })
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {
        "openrouter": "openrouter",
    })
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda slug: ["openrouter/auto"])

    custom_provs = [
        {
            "name": "Local Ollama",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": "llama3.3:70b",
            "discover_models": False,
        }
    ]

    rows = model_switch.list_authenticated_providers(
        custom_providers=custom_provs,
        max_models=50,
    )

    slugs = [r["slug"] for r in rows]
    assert "openrouter" in slugs
    assert "custom:local-ollama" in slugs
    ollama_row = next(r for r in rows if r["slug"] == "custom:local-ollama")
    assert ollama_row["name"] == "Local Ollama"
    assert ollama_row["models"] == ["llama3.3:70b"]


def test_custom_proxy_sharing_builtin_name_preserved(monkeypatch):
    """A custom proxy named after a built-in provider on a custom URL is preserved."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {
        "openrouter": {"name": "OpenRouter", "env": ["OPENROUTER_API_KEY"]},
    })
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {
        "openrouter": "openrouter",
    })
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda slug: ["openrouter/auto"])

    custom_provs = [
        {
            "name": "OpenRouter",
            "base_url": "https://my-internal-proxy.company.internal/v1",
            "api_key": "sk-internal",
            "model": "company-model-1",
            "discover_models": False,
        }
    ]

    rows = model_switch.list_authenticated_providers(
        custom_providers=custom_provs,
        max_models=50,
    )

    slugs = [r["slug"] for r in rows]
    # Both canonical built-in and the distinct custom proxy should appear
    assert "openrouter" in slugs
    assert "custom:openrouter" in slugs
    proxy_row = next(r for r in rows if r["slug"] == "custom:openrouter")
    assert proxy_row["is_user_defined"] is True
    assert proxy_row["api_url"] == "https://my-internal-proxy.company.internal/v1"
    assert proxy_row["models"] == ["company-model-1"]

