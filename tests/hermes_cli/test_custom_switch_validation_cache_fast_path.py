"""Regression tests for #158531: switch validation for custom endpoints fast-paths the
on-disk catalog cache (``cached_fetch_api_models``) before paying a live ``/models`` probe.

The picker persists each custom endpoint's listing to ``provider_models_cache.json``
(fresh + 7-day stale-serve window), so an id the user just picked from the picker is
already known. Validation accepts it from the cache; only a true cache miss -- or a
model absent from the cached listing -- falls through to the live probe, which stays
authoritative for fuzzy matches, suggestions and base-URL correction.
"""

from __future__ import annotations

from unittest.mock import patch

import hermes_cli.models as models_mod
from hermes_cli.models_validate import validate_requested_model

_URL = "https://gw.example.com/v1"
_PROBE_HIT = {
    "models": ["m-cached"],
    "probed_url": _URL + "/models",
    "used_fallback": False,
    "suggested_base_url": None,
    "resolved_base_url": _URL,
}


class TestCustomValidationCacheFastPath:
    def test_cached_catalog_hit_skips_live_probe(self):
        with patch.object(models_mod, "cached_fetch_api_models",
                          return_value=["m-cached", "m-other"]) as cached, \
             patch.object(models_mod, "probe_api_models") as probe:
            out = validate_requested_model(
                "m-cached", "custom:llmgateway", api_key="sk-x", base_url=_URL)
        cached.assert_called_once()
        assert cached.call_args.kwargs["cache_only"] is True
        probe.assert_not_called()
        assert out["accepted"] is True
        assert out["recognized"] is True

    def test_cache_hit_but_model_absent_falls_to_live_probe(self):
        """A cached listing that lacks the id must not block validation: the live probe
        remains the authority (the model may have appeared since the cache was written)."""
        with patch.object(models_mod, "cached_fetch_api_models",
                          return_value=["m-other"]), \
             patch.object(models_mod, "probe_api_models", return_value=dict(_PROBE_HIT)) as probe:
            out = validate_requested_model(
                "m-cached", "custom:llmgateway", api_key="sk-x", base_url=_URL)
        probe.assert_called_once()
        assert out["accepted"] is True

    def test_cache_miss_falls_to_live_probe(self):
        with patch.object(models_mod, "cached_fetch_api_models", return_value=None), \
             patch.object(models_mod, "probe_api_models", return_value=dict(_PROBE_HIT)) as probe:
            out = validate_requested_model(
                "m-cached", "custom:llmgateway", api_key="sk-x", base_url=_URL)
        probe.assert_called_once()
        assert out["accepted"] is True

    def test_no_base_url_skips_cache_lookup(self):
        with patch.object(models_mod, "cached_fetch_api_models") as cached, \
             patch.object(models_mod, "probe_api_models", return_value=dict(_PROBE_HIT)):
            validate_requested_model("m-cached", "custom:llmgateway", api_key="sk-x")
        cached.assert_not_called()

    def test_api_mode_and_headers_forwarded_to_cache_lookup(self):
        """The cache key is credential-fingerprinted (key + api_mode + headers): the lookup
        must pass the same shape the picker's write used, or entries silently miss. ``api_mode``
        is forwarded unconditionally -- a conditional forward (anthropic-only, like the live
        probe's) would silently miss the chat_completions family, which has no GET /models and
        must declare its mode explicitly."""
        with patch.object(models_mod, "cached_fetch_api_models", return_value=[]) as cached, \
             patch.object(models_mod, "probe_api_models", return_value=dict(_PROBE_HIT)):
            validate_requested_model(
                "m-cached", "custom:llmgateway", api_key="sk-x", base_url=_URL,
                api_mode="chat_completions", headers={"x-custom": "1"})
        assert cached.call_args.kwargs.get("api_mode") == "chat_completions"
        assert cached.call_args.kwargs.get("headers") == {"x-custom": "1"}

    def test_chat_completions_mode_cache_hit_skips_live_probe(self):
        """End-to-end shape from the #158531 review: an endpoint that requires
        ``api_mode: chat_completions`` (no GET /models, e.g. DashScope coding plan / Cline /
        MiniMax) has its picker-written cache entry keyed over that mode. The fast path must
        resolve the lookup under the same key, or the probe still fires on every switch."""
        with patch.object(models_mod, "cached_fetch_api_models",
                          return_value=["m-cached"]) as cached, \
             patch.object(models_mod, "probe_api_models") as probe:
            out = validate_requested_model(
                "m-cached", "custom:llmgateway", api_key="sk-x", base_url=_URL,
                api_mode="chat_completions")
        assert cached.call_args.kwargs.get("api_mode") == "chat_completions"
        probe.assert_not_called()
        assert out["accepted"] is True
        assert out["recognized"] is True
