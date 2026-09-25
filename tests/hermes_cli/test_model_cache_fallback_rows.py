"""A curated fallback served because the live catalog fetch failed must never be pinned in the
disk cache as if it were the account's real catalog (#107391): a transient Copilot outage wrote
the 17-model static list over the account's 10 live models with a fresh 1h TTL, and every picker
surface served the wrong list until the TTL lapsed.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import hermes_cli.models as mod


@pytest.fixture(autouse=True)
def _reset_swr_state():
    with mod._swr_refresh_lock:
        mod._swr_refresh_inflight.clear()
    yield
    with mod._swr_refresh_lock:
        mod._swr_refresh_inflight.clear()


def _row(models, age_seconds, **extra):
    return {"fp": "fp", "at": time.time() - age_seconds, "models": list(models), **extra}


LIVE = ["claude-opus-5", "kimi-k3", "gpt-5.6-sol"]
STATIC = ["gpt-5.4", "gpt-4o", "claude-sonnet-4.6"]


def test_fallback_does_not_overwrite_the_same_credentials_live_row():
    # 2h-old live row (past the 1h TTL, inside the SWR window) + a live fetch that degrades to the
    # curated list: the account's real catalog must survive, in the cache AND in the return value.
    cache = {"copilot": _row(LIVE, age_seconds=7200)}
    with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
         patch.object(mod, "_credential_fingerprint", return_value="fp"), \
         patch.object(mod, "_save_provider_models_cache") as save, \
         patch.object(mod, "provider_model_ids", return_value=mod.CuratedFallbackModels(STATIC)):
        out = mod.cached_provider_model_ids("copilot", force_refresh=True)

    assert out == LIVE
    assert cache["copilot"]["models"] == LIVE
    save.assert_not_called()


def test_fallback_row_is_recorded_as_fallback_and_never_served_stale():
    # Cold cache: the curated list is served (the picker must not be empty) but recorded as a
    # fallback row, which the SWR stale window must not resurrect — a stale fallback re-probes.
    cache: dict = {}
    with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
         patch.object(mod, "_credential_fingerprint", return_value="fp"), \
         patch.object(mod, "_save_provider_models_cache"), \
         patch.object(mod, "provider_model_ids", return_value=mod.CuratedFallbackModels(STATIC)):
        assert mod.cached_provider_model_ids("copilot") == STATIC
    assert cache["copilot"]["fallback"] is True

    # Same row, now past the fallback TTL: the live fetch runs instead of the stale-serve path,
    # and a real catalog replaces the fallback row.
    cache["copilot"]["at"] = time.time() - mod._PROVIDER_MODELS_FALLBACK_TTL - 1
    with patch.object(mod, "_load_provider_models_cache", return_value=cache), \
         patch.object(mod, "_credential_fingerprint", return_value="fp"), \
         patch.object(mod, "_save_provider_models_cache"), \
         patch.object(mod, "_spawn_swr_refresh") as spawn, \
         patch.object(mod, "provider_model_ids", return_value=list(LIVE)) as live:
        assert mod.cached_provider_model_ids("copilot") == LIVE
    live.assert_called_once()
    spawn.assert_not_called()
    assert cache["copilot"]["models"] == LIVE
    assert "fallback" not in cache["copilot"]


def test_copilot_catalog_marks_the_static_list_as_fallback_on_a_failed_live_fetch():
    with patch.object(mod, "_copilot_acp_session_models", return_value=None), \
         patch.object(mod, "_resolve_copilot_catalog_api_key", return_value="tok"), \
         patch.object(mod, "_fetch_github_models", side_effect=OSError("503")):
        for slug in ("copilot", "copilot-acp"):
            rows = mod.provider_model_ids(slug)
            assert isinstance(rows, mod.CuratedFallbackModels), slug
            assert rows == list(mod._PROVIDER_MODELS["copilot"])


def test_anthropic_catalog_marks_the_curated_list_as_fallback_on_a_failed_live_fetch():
    # _anthropic_catalog used to `return curated` (a plain list) on a failed live fetch, so a
    # transient outage on a proxy base_url cached the generic curated list over the account's/
    # proxy's real catalog for a full hour (same class of bug as #107391, just missed by 3fb8b3f6).
    with patch.object(mod, "_get_model_config_dict", return_value={}), \
         patch.object(mod, "_fetch_anthropic_models", return_value=None):
        rows = mod.provider_model_ids("anthropic")
    assert isinstance(rows, mod.CuratedFallbackModels)
    assert rows == list(mod._PROVIDER_MODELS["anthropic"])


def test_anthropic_catalog_live_success_is_not_flagged_as_fallback():
    with patch.object(mod, "_get_model_config_dict", return_value={}), \
         patch.object(mod, "_fetch_anthropic_models", return_value=["claude-x"]):
        rows = mod.provider_model_ids("anthropic")
    assert not isinstance(rows, mod.CuratedFallbackModels)


def test_codex_catalog_marks_the_fallback_list_as_placeholder_when_token_present_but_live_fails():
    # get_codex_model_ids(token=None) degrades to config.toml/cache/DEFAULT_CODEX_MODELS; a token
    # WAS available, so this is an outage placeholder, not the provider's authoritative catalog.
    with patch("hermes_cli.auth.resolve_codex_runtime_credentials", return_value={"api_key": "tok"}), \
         patch("hermes_cli.auth._codex_access_token_is_expiring", return_value=False), \
         patch("hermes_cli.codex_models._fetch_models_from_api", return_value=[]), \
         patch("hermes_cli.codex_models.get_codex_model_ids", return_value=["gpt-6-sol"]):
        rows = mod.provider_model_ids("openai-codex")
    assert isinstance(rows, mod.CuratedFallbackModels)
    assert rows == ["gpt-6-sol"]


def test_codex_catalog_live_success_is_not_flagged_as_fallback():
    with patch("hermes_cli.auth.resolve_codex_runtime_credentials", return_value={"api_key": "tok"}), \
         patch("hermes_cli.auth._codex_access_token_is_expiring", return_value=False), \
         patch("hermes_cli.codex_models._fetch_models_from_api", return_value=["gpt-6-sol-live"]):
        rows = mod.provider_model_ids("openai-codex")
    assert not isinstance(rows, mod.CuratedFallbackModels)
    assert rows == ["gpt-6-sol-live"]


def test_codex_catalog_no_token_is_not_flagged_as_fallback():
    # No live source at all (no token): the offline catalog IS this provider's catalog for this
    # call, same as a provider with no live fetcher at all — not a placeholder for an outage.
    with patch("hermes_cli.auth.resolve_codex_runtime_credentials", return_value={}), \
         patch("hermes_cli.codex_models.get_codex_model_ids", return_value=["gpt-5.4"]) as get_ids:
        rows = mod.provider_model_ids("openai-codex")
    assert not isinstance(rows, mod.CuratedFallbackModels)
    get_ids.assert_called_once_with(access_token=None)
