"""Tests for #59560: custom_providers.models list is ignored when api_key is set.

When a `custom_providers` entry has both a `models:` map (explicit list of model
IDs with context_length overrides) AND an `api_key`, the live-discovery path
in `hermes_cli.model_switch` REPLACES the configured list with whatever
`fetch_api_models` returns. This wipes the user's explicit configuration
whenever the live probe succeeds with a different (or partial) list.

Per the issue (#59560), 9router-style aggregators return 136 models via
`GET /v1/models`, but the user's `models:` list in config is the source of
truth for what they want exposed. The two should be MERGED (configured
additions + live discovery), not REPLACED.

Failing-test-first:
  test_user_configured_models_preserved_when_api_key_set — bug detection
  test_live_discovery_falls_back_to_configured_when_probe_errors — safety
  test_live_discovery_still_used_when_no_models_configured — existing behavior
  test_live_models_merged_with_configured_when_discoverable — new behavior
"""

import pytest

import hermes_cli.providers as providers_mod
import hermes_cli.model_switch as model_switch_mod
from hermes_cli.model_switch import list_authenticated_providers


def _patch_no_models_dev(monkeypatch):
    """Skip models.dev for these tests so we isolate custom_providers logic."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})


def _patch_fetch(monkeypatch, return_value):
    """Patch fetch_api_models so tests can control the live probe response.

    The fix in #59560 introduces a module-level alias
    ``hermes_cli.model_switch._fetch_api_models`` (which makes the probe
    patchable from tests). Patch both the source module and the alias so
    tests are robust to either implementation.
    """
    def fake_fetch(*args, **kwargs):
        return return_value

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch)
    monkeypatch.setattr(model_switch_mod, "_fetch_api_models", fake_fetch)


def test_user_configured_models_preserved_when_api_key_set(monkeypatch):
    """The user's explicit `models:` map survives the live-discovery replacement.

    Bug from #59560: when api_key is set, the configured `models:` list is
    wiped and replaced with whatever fetch_api_models returns from the live
    API. The user explicitly configured their model subset; that config is
    authoritative and should not be silently replaced.
    """
    _patch_no_models_dev(monkeypatch)

    # Simulate 9router returning only a subset (e.g. legacy/aliasing glitch
    # or the aggregator just re-listing built-ins).
    _patch_fetch(monkeypatch, ["gpt-4o", "claude-3-5-sonnet"])

    # User configured 3 models in their models: map.
    user_configured = ["my_combo", "router-special-a", "router-special-b"]

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {
                "name": "My_Combo",
                "base_url": "http://localhost:20128/v1",
                "api_key": "sk-test",
                "model": "my_combo",
                "models": {m: {"context_length": 1000000} for m in user_configured},
            }
        ],
    )

    row = next(p for p in providers if "my_combo" in p["slug"])

    # The user's configured models must all be present, not wiped by the
    # live discovery result.
    assert all(m in row["models"] for m in user_configured), (
        f"User-configured models were wiped by live discovery. "
        f"Configured: {user_configured}, Got: {row['models']}"
    )


def test_live_models_merged_with_configured_when_discoverable(monkeypatch):
    """When the user has configured models AND live discovery returns
    additional/different models, both lists appear in the picker.

    The fix is: union of configured + live, deduplicated, configured first
    (so the user's explicit list is preserved at the top).
    """
    _patch_no_models_dev(monkeypatch)

    _patch_fetch(monkeypatch, ["gpt-4o", "claude-3-5-sonnet", "my_combo"])

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {
                "name": "My_Combo",
                "base_url": "http://localhost:20128/v1",
                "api_key": "sk-test",
                "model": "my_combo",
                "models": {
                    "my_combo": {"context_length": 1000000},
                    "router-special": {"context_length": 1000000},
                },
            }
        ],
    )

    row = next(p for p in providers if "my_combo" in p["slug"])

    # All configured models present
    assert "my_combo" in row["models"]
    assert "router-special" in row["models"]
    # Live-only models also present
    assert "gpt-4o" in row["models"]
    assert "claude-3-5-sonnet" in row["models"]


def test_live_discovery_still_used_when_no_models_configured(monkeypatch):
    """When the user has not configured a `models:` map AND no singular
    `model:` field, live discovery is still the source of truth
    (existing behaviour must not regress).
    """
    _patch_no_models_dev(monkeypatch)

    _patch_fetch(monkeypatch, ["llama-3", "qwen-2.5"])

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {
                "name": "Bare Ollama",
                "base_url": "http://localhost:11434/v1",
                # No api_key (local server)
                # No model: field
                # No models: map
            }
        ],
    )

    row = next(p for p in providers if "ollama" in p["slug"])
    # Both live models + the singular model field
    assert "llama-3" in row["models"]
    assert "qwen-2.5" in row["models"]


def test_live_discovery_falls_back_to_configured_when_probe_errors(monkeypatch):
    """If fetch_api_models raises/returns None, the configured models: list
    is still the result the picker shows (don't show empty).
    """
    _patch_no_models_dev(monkeypatch)

    def broken_probe(*a, **k):
        raise ConnectionError("simulated network error")

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", broken_probe)
    monkeypatch.setattr(model_switch_mod, "_fetch_api_models", broken_probe)

    providers = list_authenticated_providers(
        user_providers={},
        custom_providers=[
            {
                "name": "My_Combo",
                "base_url": "http://localhost:20128/v1",
                "api_key": "sk-test",
                "model": "my_combo",
                "models": {
                    "my_combo": {"context_length": 1000000},
                    "router-special": {"context_length": 1000000},
                },
            }
        ],
    )

    row = next(p for p in providers if "my_combo" in p["slug"])
    # Configured list preserved
    assert "my_combo" in row["models"]
    assert "router-special" in row["models"]