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
