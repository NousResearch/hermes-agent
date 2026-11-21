"""Kimi Coding Plan catalogue + context regression tests (#138375).

``GET api.kimi.com/coding/v1/models`` is the only authority for this provider: the inference
endpoint answers HTTP 200 for any model id, so HTTP-status probes cannot validate it. The
curated ``kimi-coding`` list must mirror the live roster, and ``k3-256k``'s 256 Ki window must
resolve endpoint-scoped instead of falling through to the generic 256,000 default.
"""

from unittest.mock import patch

from agent.model_metadata import get_model_context_length
from hermes_cli.models_catalog_static import _PROVIDER_MODELS

KIMI_CODING_ROSTER = ["k3", "kimi-for-coding", "kimi-for-coding-highspeed", "k3-256k"]
RETIRED_IDS = [
    "kimi-k3", "kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5", "kimi-k2-thinking",
    "kimi-k2-thinking-turbo", "kimi-k2-turbo-preview", "kimi-k2-0905-preview",
]


class TestKimiCodingCatalogue:
    def test_curated_list_matches_live_roster(self):
        assert _PROVIDER_MODELS["kimi-coding"] == KIMI_CODING_ROSTER

    def test_retired_ids_are_dropped(self):
        assert not (set(_PROVIDER_MODELS["kimi-coding"]) & set(RETIRED_IDS))

    def test_cn_roster_untouched(self):
        """The Moonshot CN endpoint serves a different roster; this fix must not touch it."""
        assert "kimi-k2.7-code" in _PROVIDER_MODELS["kimi-coding-cn"]


class TestK3_256kContextIsEndpointScoped:
    def _ctx(self, model: str, base_url: str) -> int:
        with patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.fetch_model_metadata", return_value={}), \
             patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             patch("agent.model_metadata._query_ollama_api_show", return_value=None), \
             patch("agent.models_dev.lookup_models_dev_context", return_value=None):
            return get_model_context_length(model, provider="kimi-coding", base_url=base_url)

    def test_resolves_256k_on_canonical_coding_endpoint(self):
        for base_url in ("https://api.kimi.com/coding", "https://api.kimi.com/coding/v1"):
            assert self._ctx("k3-256k", base_url) == 262_144, base_url

    def test_does_not_leak_off_endpoint(self):
        """Off the canonical host the 256 Ki value must not apply (generic fallback ≠ 262,144)."""
        for base_url in ("https://api.moonshot.ai/v1", "https://api.kimi.com:8443/coding"):
            assert self._ctx("k3-256k", base_url) != 262_144, base_url
