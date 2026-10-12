"""Stale OpenRouter entries for the DeepSeek V4 alias slugs must not mask the 1M catalog."""
from unittest.mock import patch

from agent.model_metadata import get_model_context_length


class TestStaleAliasEntries:
    def test_stale_openrouter_alias_entries_do_not_mask_v4_alias_context(self):
        """Stale pre-V4 OpenRouter entries for the alias slugs must not win over the 1M catalog.

        ``deepseek-chat`` is served by V4 Flash (1M) while the OpenRouter catalogue still carries
        its pre-V4 window — 163,840, verified live against /api/v1/models. The stale number cannot
        be told apart from a legitimate smaller model (163,840 sits above the 128K of the shorter
        ``deepseek`` catch-all), so only the alias identity identifies it. Both call paths must
        reject such an entry: the provider-unaware lookup and the explicit
        ``provider="openrouter"`` lookup. ``deepseek/deepseek-reasoner`` is pinned with a stale
        65,536 entry for the same reason — the vendor-prefixed spelling resolves through the same
        alias slug, so it has to be rejected identically wherever the catalogue entry reappears.
        """
        catalogue = {
            "deepseek-chat": {"context_length": 163_840},
            "deepseek/deepseek-chat": {"context_length": 163_840},
            "deepseek/deepseek-reasoner": {"context_length": 65_536},
            # Non-alias sibling entry: a legitimate smaller window must survive untouched.
            "deepseek/deepseek-v3.2": {"context_length": 65_536},
        }
        with patch("agent.model_metadata.get_cached_context_length", return_value=None), \
             patch("agent.model_metadata.fetch_model_metadata", return_value=catalogue), \
             patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value={}), \
             patch("agent.model_metadata._query_ollama_api_show", return_value=None), \
             patch("agent.models_dev.lookup_models_dev_context", return_value=None):
            for alias in ("deepseek-chat", "deepseek/deepseek-chat", "deepseek/deepseek-reasoner"):
                assert get_model_context_length(alias) == 1_000_000, alias
            # Same stale entries via the explicit OpenRouter provider path (step 5f) — one rule,
            # both call paths, so a fix cannot land on only half of the resolution chain.
            for alias in ("deepseek-chat", "deepseek/deepseek-chat", "deepseek/deepseek-reasoner"):
                assert get_model_context_length(alias, provider="openrouter") == 1_000_000, alias
            # Alias-scoped guard: a versioned id keeps the window the catalogue reports.
            assert get_model_context_length("deepseek/deepseek-v3.2") == 65_536
            assert get_model_context_length("deepseek/deepseek-v3.2", provider="openrouter") == 65_536
