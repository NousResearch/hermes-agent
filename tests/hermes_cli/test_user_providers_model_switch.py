"""Tests for user-defined providers (providers: dict) in /model.

These tests ensure that providers defined in the config.yaml ``providers:`` section
are properly resolved for model switching and that their full ``models:`` lists
are exposed in the model picker.
"""

import pytest
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli import runtime_provider as rp


@pytest.fixture(autouse=True)
def _no_live_builtin_provider_probes(monkeypatch):
    """Keep picker tests offline: builtin-provider catalog fetches hit the network."""
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids", lambda *_a, **_kw: []
    )
    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids", lambda *_a, **_kw: []
    )


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

    def fake_fetch_api_models(api_key, base_url, **kwargs):
        calls.append((api_key, base_url, kwargs))
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
    assert calls == [
        (
            "sk-test",
            "http://127.0.0.1:3000/api/v1",
            {"timeout": 5.0, "api_mode": None, "headers": None},
        )
    ]
    assert user_prov["models"] == ["old-configured-model", "new-live-model"]
    assert user_prov["total_models"] == 2








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
    # Singular ``model:``-only entries are un-narrowed → section 3 now probes
    # them; stub the probe so the test stays hermetic (endpoints are fake).
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: None)

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




# =============================================================================
# Regression: user_providers override for private models not listed by /v1/models
# =============================================================================

_REJECTED_VALIDATION = {
    "accepted": False,
    "persist": False,
    "recognized": False,
    "message": "not found",
}


def _run_user_provider_override_case(
    *,
    slug,
    name,
    base_url,
    models,
    raw_input,
):
    """Run ``switch_model`` with a private user provider and a rejected API check.

    The bug in PR #17964 was that ``user_providers`` was treated like a list,
    so private models listed in ``models:`` never triggered the override path.
    These tests keep the validation failure in place and prove the config list
    still wins for both dict- and list-shaped ``models`` entries.
    """
    from unittest.mock import patch

    user_providers = {
        slug: {
            "name": name,
            "api": base_url,
            "discover_models": False,
            "models": models,
        }
    }

    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, provider: model), \
         patch("hermes_cli.models.validate_requested_model", return_value=_REJECTED_VALIDATION), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={"api_key": "***", "base_url": base_url, "api_mode": "anthropic_messages"}):
        return switch_model(
            raw_input=raw_input,
            current_provider=slug,
            current_model="old-model",
            current_base_url=base_url,
            user_providers=user_providers,
            custom_providers=[],
        )




# =============================================================================
# Section 3 no-auth live discovery (PR #29575)
# =============================================================================

def test_section3_probes_no_key_endpoint_without_explicit_models(monkeypatch):
    """A providers: entry with no api_key and no explicit models: list should
    still probe /v1/models for live discovery — mirroring section 4's policy.

    Regression for #29575: local self-hosted backends (llama.cpp, Ollama,
    vLLM) that don't require auth previously showed an empty/minimal model
    list because section 3 gated probing on ``api_url and api_key``.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    probed = {}

    def _fake_fetch(api_key, api_url, **kwargs):
        probed["called"] = True
        probed["api_key"] = api_key
        probed["api_url"] = api_url
        probed["kwargs"] = kwargs
        return ["live-model-1", "live-model-2", "live-model-3"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", _fake_fetch)

    user_providers = {
        "local-llamacpp": {
            "name": "Local llama.cpp",
            "api": "http://localhost:8080/v1",
            # No api_key, no models list — bare local endpoint.
        }
    }

    providers = list_authenticated_providers(
        current_provider="local-llamacpp",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    assert probed.get("called") is True, "no-key bare endpoint should be probed"
    assert probed["api_key"] == ""
    assert probed["kwargs"] == {"timeout": 5.0, "api_mode": None, "headers": None}
    row = next(p for p in providers if p["slug"] == "local-llamacpp")
    assert row["models"] == ["live-model-1", "live-model-2", "live-model-3"]
    assert row["total_models"] == 3


def test_section3_probes_no_key_endpoint_with_singular_default_model(monkeypatch):
    """A providers: entry with no api_key and only a singular ``default_model``
    (no explicit ``models:`` list) must still probe /v1/models — the singular
    field is just the active selection, not the user narrowing the endpoint.

    Regression for #40554 / PR #68984 (@vigilancetech-com): section 3 derived
    ``has_explicit_models`` from the merged models list, so the lone
    ``default_model`` entry suppressed live discovery and the /model picker
    showed a one-line menu for local no-auth endpoints.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    probed = {}

    def _fake_fetch(api_key, api_url, **kwargs):
        probed["called"] = True
        probed["api_key"] = api_key
        return ["live-model-1", "live-model-2", "live-model-3"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", _fake_fetch)

    user_providers = {
        "local-ollama": {
            "name": "Local Ollama",
            "api": "http://localhost:11434/v1",
            "default_model": "llama3",
            # No api_key, no models: list — singular default only.
        }
    }

    providers = list_authenticated_providers(
        current_provider="local-ollama",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    assert probed.get("called") is True, (
        "singular default_model must not suppress live discovery"
    )
    assert probed["api_key"] == ""
    row = next(p for p in providers if p["slug"] == "local-ollama")
    assert row["models"] == ["live-model-1", "live-model-2", "live-model-3"]
    assert row["total_models"] == 3


def test_current_custom_model_is_surfaced_in_builtin_provider_row(monkeypatch):
    """A custom/uncurated model selected via the CLI must appear in its
    provider's picker row.

    Regression: selecting `/model openrouter/<uncurated-name>` left the model
    invisible in every picker (main model picker AND the MoA reference/aggregator
    slot pickers, which read these rows), because the row only carried the
    curated catalog. The current model is now injected at the front of the
    current provider's list.
    """
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    # Pin a small curated catalog so the assertion is deterministic.
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids",
        lambda slug, **kw: ["anthropic/claude-opus-4.8", "openai/gpt-5.5"]
        if slug == "openrouter"
        else [],
    )

    custom = "some-vendor/totally-custom-model-v9"
    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_model=custom,
        user_providers={},
        custom_providers=[],
    )

    row = next(p for p in providers if p["slug"] == "openrouter")
    assert custom in row["models"], row["models"]
    assert row["models"][0] == custom  # injected at the front
    assert row["total_models"] == 3


def test_current_custom_model_not_leaked_into_other_provider_rows(monkeypatch):
    """The current model is only injected into the CURRENT provider's row,
    never into other providers (which can't serve it)."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("NOUS_API_KEY", "sk-test")
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids",
        lambda slug, **kw: ["curated/one"],
    )

    custom = "some-vendor/totally-custom-model-v9"
    providers = list_authenticated_providers(
        current_provider="openrouter",
        current_model=custom,
        user_providers={},
        custom_providers=[],
    )

    for row in providers:
        if row["slug"] != "openrouter" and not row.get("is_current"):
            assert custom not in row.get("models", []), f"leaked into {row['slug']}"


def test_list_authenticated_providers_discover_models_false_keeps_declared_only(monkeypatch):
    """``providers.<built-in>.discover_models: false`` must skip the live
    /v1/models merge and surface ONLY the declared ``models:`` list.

    Regression for #80536: section 1 of list_authenticated_providers merged
    the live catalog via ``cached_provider_model_ids()`` even when config
    pinned ``discover_models: false``, so the /model picker showed models the
    user never declared.
    """
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {"deepseek": {"name": "DeepSeek"}},
    )
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def _boom(slug, *args, **kwargs):
        if slug == "deepseek":
            raise AssertionError("live /v1/models catalog must not be fetched")
        return []

    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", _boom)

    user_providers = {
        "deepseek": {
            "discover_models": False,
            "models": ["deepseek-chat", "deepseek-reasoner"],
        }
    }

    providers = list_authenticated_providers(
        current_provider="deepseek",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    row = next(p for p in providers if p["slug"] == "deepseek")
    assert row["models"] == ["deepseek-chat", "deepseek-reasoner"]
    assert row["total_models"] == 2
