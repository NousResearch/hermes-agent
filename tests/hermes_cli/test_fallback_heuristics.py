"""Unit tests for fallback heuristics and criteria resolution (hermes_cli/fallback_heuristics.py)."""

from __future__ import annotations

import unittest
from hermes_cli.fallback_heuristics import (
    FallbackCriteria,
    ModelCandidateMetadata,
    enrich_model_metadata,
    extract_context_window_from_name,
    extract_date_snapshot,
    extract_param_size_b,
    extract_vendor,
    extract_version_tuple,
    filter_and_rank_candidates,
    is_flash_model,
    is_free_model_slug,
    parse_fallback_criteria,
    resolve_fallback_candidates,
    resolve_fallback_entry,
)
from hermes_cli.fallback_config import get_fallback_chain


class TestParseFallbackCriteria(unittest.TestCase):
    def test_shorthand_strings(self):
        c1 = parse_fallback_criteria("largest_parameter_count_free")
        self.assertEqual(c1.sort_by, "largest_parameter_count")
        self.assertTrue(c1.free_only)
        self.assertTrue(c1.require_tools)

        # Alias "largest_free" maps to largest_parameter_count
        c1_alias = parse_fallback_criteria("largest_free")
        self.assertEqual(c1_alias.sort_by, "largest_parameter_count")

        c2 = parse_fallback_criteria("greatest_context_free")
        self.assertEqual(c2.sort_by, "greatest_context")
        self.assertTrue(c2.free_only)

        c3 = parse_fallback_criteria("smallest_free")
        self.assertEqual(c3.sort_by, "smallest")
        self.assertTrue(c3.free_only)

        c4 = parse_fallback_criteria("latest_flash_free")
        self.assertEqual(c4.sort_by, "latest_flash")
        self.assertTrue(c4.flash_only)
        self.assertTrue(c4.free_only)

        c5 = parse_fallback_criteria("latest")
        self.assertEqual(c5.sort_by, "latest")

    def test_auto_and_heuristic_prefixes(self):
        c1 = parse_fallback_criteria("auto:free:largest_parameter_count")
        self.assertEqual(c1.sort_by, "largest_parameter_count")
        self.assertTrue(c1.free_only)

        c1_alias = parse_fallback_criteria("auto:free:largest")
        self.assertEqual(c1_alias.sort_by, "largest_parameter_count")

        c2 = parse_fallback_criteria("heuristic:greatest_context")
        self.assertEqual(c2.sort_by, "greatest_context")

    def test_dict_criteria(self):
        data = {
            "criteria": {
                "sort_by": "largest_parameter_count",
                "free": True,
                "require_tools": True,
                "flash": True,
                "vendor": "google",
                "filter": "flash-exp",
                "min_context": 64000,
                "max_candidates": 5,
            }
        }
        c = parse_fallback_criteria(data)
        self.assertEqual(c.sort_by, "largest_parameter_count")
        self.assertTrue(c.free_only)
        self.assertTrue(c.require_tools)
        self.assertTrue(c.flash_only)
        self.assertEqual(c.vendor, "google")
        self.assertEqual(c.name_filter, "flash-exp")
        self.assertEqual(c.min_context, 64000)
        self.assertEqual(c.max_candidates, 5)

    def test_describe(self):
        c = FallbackCriteria(sort_by="largest_parameter_count", free_only=True, flash_only=True)
        desc = c.describe()
        self.assertIn("flash", desc)
        self.assertIn("largest parameter count", desc)
        self.assertIn("free", desc)


class TestHeuristicExtractors(unittest.TestCase):
    def test_extract_param_size_b(self):
        # Explicit B sizes
        self.assertEqual(extract_param_size_b("meta-llama/llama-3.3-70b-instruct"), 70.0)
        self.assertEqual(extract_param_size_b("openai/gpt-405b"), 405.0)
        self.assertEqual(extract_param_size_b("qwen/qwen-2.5-0.5b"), 0.5)
        self.assertEqual(extract_param_size_b("google/gemini-1.5b"), 1.5)
        self.assertEqual(extract_param_size_b("qwen/qwen-2.5-coder-32b-instruct:free"), 32.0)

        # MoE notation
        self.assertEqual(extract_param_size_b("mistralai/mixtral-8x7b-instruct"), 56.0)
        self.assertEqual(extract_param_size_b("mistralai/mixtral-8x22b-instruct"), 176.0)

        # Active MoE notation
        self.assertEqual(extract_param_size_b("nvidia/nemotron-3-super-120b-a12b:free"), 120.0)

        # Qualitative tiers
        self.assertEqual(extract_param_size_b("thinkingmachines/inkling-small:free"), 7.0)
        self.assertEqual(extract_param_size_b("poolside/laguna-xs-2.1:free"), 1.0)
        self.assertEqual(extract_param_size_b("anthropic/claude-opus-5"), 100.0)

    def test_extract_context_window(self):
        self.assertEqual(extract_context_window_from_name("llama-3-70b-128k"), 131072)
        self.assertEqual(extract_context_window_from_name("model-32k"), 32768)
        self.assertEqual(extract_context_window_from_name("model-200k"), 200000)
        self.assertEqual(extract_context_window_from_name("gemini-1m"), 1000000)
        self.assertEqual(extract_context_window_from_name("gemini-2m"), 2000000)
        self.assertEqual(extract_context_window_from_name("unknown-model", default=8192), 8192)

    def test_is_flash_model(self):
        self.assertTrue(is_flash_model("google/gemini-2.0-flash-exp:free"))
        self.assertTrue(is_flash_model("deepseek/deepseek-v4.1-flash"))
        self.assertTrue(is_flash_model("qwen/qwen3.8-flash"))
        self.assertFalse(is_flash_model("anthropic/claude-opus-5"))
        self.assertFalse(is_flash_model("meta-llama/llama-3.3-70b-instruct:free"))

    def test_is_free_model_slug(self):
        self.assertTrue(is_free_model_slug("meta-llama/llama-3.3-70b-instruct:free"))
        self.assertTrue(is_free_model_slug("stealth/union-alpha"))
        self.assertFalse(is_free_model_slug("anthropic/claude-opus-5"))

    def test_extract_vendor(self):
        self.assertEqual(extract_vendor("google/gemini-2.5-flash"), "google")
        self.assertEqual(extract_vendor("meta-llama/llama-3.3-70b"), "meta-llama")
        self.assertEqual(extract_vendor("qwen/qwen-2.5-coder"), "qwen")
        self.assertEqual(extract_vendor("bare-model"), "")

    def test_extract_version_tuple(self):
        self.assertEqual(extract_version_tuple("gemini-3.8-flash"), (3, 8))
        self.assertEqual(extract_version_tuple("llama-3.3-70b"), (3, 3))
        self.assertEqual(extract_version_tuple("llama-3.1-70b"), (3, 1))
        self.assertEqual(extract_version_tuple("v4.1"), (4, 1))

    def test_extract_date_snapshot(self):
        self.assertEqual(extract_date_snapshot("deepseek-v4-flash-0731"), 20240731.0)
        self.assertEqual(extract_date_snapshot("model-20241022"), 20241022.0)


class TestFilterAndRankCandidates(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            ModelCandidateMetadata(
                id="qwen/qwen-2.5-coder-32b-instruct:free",
                is_free=True,
                param_size_b=32.0,
                context_window=32768,
                supports_tools=True,
                is_flash=False,
                vendor="qwen",
            ),
            ModelCandidateMetadata(
                id="meta-llama/llama-3.3-70b-instruct:free",
                is_free=True,
                param_size_b=70.0,
                context_window=131072,
                supports_tools=True,
                is_flash=False,
                vendor="meta-llama",
            ),
            ModelCandidateMetadata(
                id="nvidia/nemotron-3-super-120b-a12b:free",
                is_free=True,
                param_size_b=120.0,
                context_window=65536,
                supports_tools=True,
                is_flash=False,
                vendor="nvidia",
            ),
            ModelCandidateMetadata(
                id="google/gemini-2.0-flash-exp:free",
                is_free=True,
                param_size_b=3.0,
                context_window=1048576,
                supports_tools=True,
                is_flash=True,
                vendor="google",
                version_tuple=(2, 0),
            ),
            ModelCandidateMetadata(
                id="google/gemini-1.5-flash:free",
                is_free=True,
                param_size_b=3.0,
                context_window=1048576,
                supports_tools=True,
                is_flash=True,
                vendor="google",
                version_tuple=(1, 5),
            ),
            ModelCandidateMetadata(
                id="paid-large-model/405b",
                is_free=False,
                param_size_b=405.0,
                context_window=131072,
                supports_tools=True,
                vendor="paid",
            ),
            ModelCandidateMetadata(
                id="no-tools-model:free",
                is_free=True,
                param_size_b=70.0,
                context_window=32768,
                supports_tools=False,
            ),
        ]

    def test_filter_free_and_tools(self):
        criteria = FallbackCriteria(sort_by="largest_parameter_count", free_only=True, require_tools=True)
        ranked = filter_and_rank_candidates(self.candidates, criteria)
        ids = [c.id for c in ranked]
        self.assertNotIn("paid-large-model/405b", ids)
        self.assertNotIn("no-tools-model:free", ids)

    def test_rank_largest_parameter_count(self):
        criteria = FallbackCriteria(sort_by="largest_parameter_count", free_only=True, require_tools=True)
        ranked = filter_and_rank_candidates(self.candidates, criteria)
        # nemotron-120b > llama-70b > qwen-32b
        self.assertEqual(ranked[0].id, "nvidia/nemotron-3-super-120b-a12b:free")
        self.assertEqual(ranked[1].id, "meta-llama/llama-3.3-70b-instruct:free")
        self.assertEqual(ranked[2].id, "qwen/qwen-2.5-coder-32b-instruct:free")

    def test_rank_greatest_context(self):
        criteria = FallbackCriteria(sort_by="greatest_context", free_only=True, require_tools=True)
        ranked = filter_and_rank_candidates(self.candidates, criteria)
        # 1M (gemini) > 131k (llama) > 65k (nemotron)
        self.assertIn("gemini", ranked[0].id)
        self.assertEqual(ranked[0].context_window, 1048576)
        self.assertEqual(ranked[2].id, "meta-llama/llama-3.3-70b-instruct:free")

    def test_rank_smallest(self):
        criteria = FallbackCriteria(sort_by="smallest", free_only=True, require_tools=True)
        ranked = filter_and_rank_candidates(self.candidates, criteria)
        # gemini 3b < qwen 32b < llama 70b < nemotron 120b
        self.assertIn("gemini", ranked[0].id)
        self.assertEqual(ranked[-1].id, "nvidia/nemotron-3-super-120b-a12b:free")

    def test_rank_latest_flash(self):
        criteria = FallbackCriteria(sort_by="latest_flash", flash_only=True, free_only=True)
        ranked = filter_and_rank_candidates(self.candidates, criteria)
        self.assertEqual(len(ranked), 2)
        # gemini 2.0 > gemini 1.5
        self.assertEqual(ranked[0].id, "google/gemini-2.0-flash-exp:free")
        self.assertEqual(ranked[1].id, "google/gemini-1.5-flash:free")

    def test_filter_vendor(self):
        criteria = FallbackCriteria(vendor="google", free_only=True)
        ranked = filter_and_rank_candidates(self.candidates, criteria)
        self.assertTrue(all("google" in c.id for c in ranked))


class TestResolveFallbackEntry(unittest.TestCase):
    def test_static_entry_preserved(self):
        entry = {"provider": "anthropic", "model": "claude-sonnet-5"}
        res = resolve_fallback_entry(entry)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["model"], "claude-sonnet-5")

    def test_heuristic_entry_resolves_candidates(self):
        entry = {"provider": "openrouter", "heuristic": "largest_parameter_count_free"}
        res = resolve_fallback_entry(entry, max_candidates=3)
        self.assertTrue(len(res) >= 1)
        self.assertEqual(res[0]["provider"], "openrouter")
        self.assertTrue(res[0]["model"])
        self.assertIn("criteria_rank", res[0])
        self.assertEqual(res[0]["criteria_rank"], 1)

    def test_fallback_chain_integration(self):
        config = {
            "fallback_heuristics": True,
            "fallback_providers": [
                {"provider": "openrouter", "heuristic": "largest_parameter_count_free"},
                {"provider": "anthropic", "model": "claude-sonnet-5"},
            ]
        }
        chain = get_fallback_chain(config)
        self.assertTrue(len(chain) >= 2)
        self.assertEqual(chain[-1]["model"], "claude-sonnet-5")
        self.assertEqual(chain[0]["provider"], "openrouter")

    def test_fallback_heuristics_off_by_default(self):
        config = {
            "fallback_providers": [
                {"heuristic": "largest_parameter_count_free"},
                {"provider": "anthropic", "model": "claude-sonnet-5"},
            ]
        }
        chain = get_fallback_chain(config)
        # Heuristics are OFF by default: only static model is retained
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0]["model"], "claude-sonnet-5")

    def test_word_boundary_qualitative_tiers(self):
        # galaxy-axis-70b has "xs" inside "axis", but should NOT match "xs" (1.0b); it should match 70b
        self.assertEqual(extract_param_size_b("galaxy-axis-70b"), 70.0)
        # prompt-model-mini has "pro" inside "prompt", but should match "mini" (3.0b)
        self.assertEqual(extract_param_size_b("prompt-model-mini"), 3.0)

    def test_candidate_cache_and_clear(self):
        from hermes_cli.fallback_heuristics import clear_candidate_cache, get_candidate_models
        from unittest.mock import patch
        clear_candidate_cache()
        sample = {"cand-1": ModelCandidateMetadata(id="cand-1", provider="openrouter")}
        with patch("hermes_cli.fallback_heuristics._gather_openrouter_candidates", return_value=sample) as mock_gather:
            cands1 = get_candidate_models("openrouter")
            cands2 = get_candidate_models("openrouter")
            self.assertEqual(mock_gather.call_count, 1)
            self.assertEqual([c.id for c in cands1], [c.id for c in cands2])
            clear_candidate_cache()
            cands3 = get_candidate_models("openrouter")
            self.assertEqual(mock_gather.call_count, 2)
            self.assertEqual([c.id for c in cands3], ["cand-1"])

    def test_agent_init_fallback_entries(self):
        from agent.agent_init import _fallback_entries
        entries = _fallback_entries([
            {"provider": "openrouter", "heuristic": "greatest_context_free", "max_candidates": 2, "heuristic_enabled": True},
            {"provider": "anthropic", "model": "claude-sonnet-5"},
        ])
        self.assertIsInstance(entries, list)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["provider"], "openrouter")
        self.assertEqual(entries[-1]["model"], "claude-sonnet-5")

    def test_default_provider_is_nous(self):
        c = FallbackCriteria()
        self.assertEqual(c.provider, "nous")
        self.assertEqual(c.model_provider, "nous")

    def test_model_provider_alias_and_setter(self):
        c = FallbackCriteria()
        c.model_provider = "openrouter"
        self.assertEqual(c.provider, "openrouter")
        self.assertEqual(c.model_provider, "openrouter")

    def test_parse_model_provider_in_dict(self):
        # Top-level model_provider
        c1 = parse_fallback_criteria({"model_provider": "openrouter", "heuristic": "greatest_context"})
        self.assertEqual(c1.provider, "openrouter")

        # Nested in criteria dict
        c2 = parse_fallback_criteria({"criteria": {"model_provider": "openrouter", "sort_by": "latest"}})
        self.assertEqual(c2.provider, "openrouter")

        # Shorthand string with provider: / model_provider:
        c3 = parse_fallback_criteria("model_provider:openrouter, largest_parameter_count")
        self.assertEqual(c3.provider, "openrouter")

        c4 = parse_fallback_criteria("provider=openrouter, greatest_context")
        self.assertEqual(c4.provider, "openrouter")

        c5 = parse_fallback_criteria("nous:latest_flash_free")
        self.assertEqual(c5.provider, "nous")
        self.assertTrue(c5.flash_only)

    def test_resolve_fallback_entry_defaults_to_nous(self):
        # When no provider is specified on a heuristic entry, it must default to Nous Research
        entry = {"heuristic": "largest_parameter_count_free"}
        res = resolve_fallback_entry(entry, max_candidates=2)
        self.assertTrue(len(res) >= 1)
        self.assertEqual(res[0]["provider"], "nous")
        self.assertTrue(res[0]["model"])
        self.assertIn("largest parameter count", res[0]["criteria_matched"])

    def test_resolve_fallback_entry_with_model_provider(self):
        # model_provider used instead of provider
        entry = {"model_provider": "openrouter", "heuristic": "largest_parameter_count_free"}
        res = resolve_fallback_entry(entry, max_candidates=2)
        self.assertTrue(len(res) >= 1)
        self.assertEqual(res[0]["provider"], "openrouter")
        self.assertTrue(res[0]["model"])

    def test_fallback_chain_with_default_provider(self):
        # Chain without explicit provider resolves to Nous when heuristics are enabled
        config = {
            "fallback_heuristics": True,
            "fallback_providers": [
                {"heuristic": "largest_parameter_count_free"},
                {"model_provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"},
            ]
        }
        chain = get_fallback_chain(config)
        self.assertTrue(len(chain) >= 2)
        self.assertEqual(chain[0]["provider"], "nous")
        self.assertEqual(chain[-1]["provider"], "openrouter")

    def test_fallback_criteria_reasoning_effort(self):
        c = FallbackCriteria()
        self.assertIsNone(c.reasoning_effort)

        c1 = parse_fallback_criteria({"reasoning_effort": "low", "heuristic": "latest_flash"})
        self.assertEqual(c1.reasoning_effort, "low")
        self.assertIn("reasoning:low", c1.describe())

        c2 = parse_fallback_criteria("thinking:none, largest_parameter_count_free")
        self.assertEqual(c2.reasoning_effort, "none")
        self.assertIn("reasoning:none", c2.describe())

        c3 = parse_fallback_criteria("reasoning:default, greatest_context")
        self.assertEqual(c3.reasoning_effort, "default")
        self.assertNotIn("reasoning:default", c3.describe())

    def test_reresolve_fallback_reasoning_config_precedence(self):
        from types import SimpleNamespace
        from agent.chat_completion_helpers import _reresolve_fallback_reasoning_config

        # 1. Fallback entry with heuristic provenance and no reasoning setting defaults to native default ({"native": True})
        agent = SimpleNamespace(model="stepfun/step-3.7-flash:free", reasoning_config={"enabled": True, "effort": "high"})
        _reresolve_fallback_reasoning_config(agent, {"_is_heuristic": True})
        self.assertEqual(agent.reasoning_config, {"native": True})

        # 2. Fallback entry with explicit reasoning_effort: low
        _reresolve_fallback_reasoning_config(agent, {"reasoning_effort": "low"})
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "low"})

        # 3. Fallback entry with explicit reasoning_effort: none
        _reresolve_fallback_reasoning_config(agent, {"reasoning_effort": "none"})
        self.assertEqual(agent.reasoning_config, {"enabled": False})

        # 4. Fallback entry with explicit reasoning_effort: default maps to {"native": True}
        _reresolve_fallback_reasoning_config(agent, {"reasoning_effort": "default"})
        self.assertEqual(agent.reasoning_config, {"native": True})

        # 5. Nested in criteria
        _reresolve_fallback_reasoning_config(agent, {"criteria": {"reasoning_effort": "medium"}})
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "medium"})


class TestFallbackHeuristicsToggle(unittest.TestCase):
    """Test that heuristics are disabled by default and can be toggled on/off."""

    def test_disabled_by_default(self):
        from hermes_cli.fallback_config import is_fallback_heuristics_enabled
        # None / empty config defaults to False
        self.assertFalse(is_fallback_heuristics_enabled(None))
        self.assertFalse(is_fallback_heuristics_enabled({}))
        self.assertFalse(is_fallback_heuristics_enabled({"fallback_providers": []}))

    def test_enabled_via_config_flags(self):
        from hermes_cli.fallback_config import is_fallback_heuristics_enabled
        # Top-level fallback_heuristics
        self.assertTrue(is_fallback_heuristics_enabled({"fallback_heuristics": True}))
        self.assertFalse(is_fallback_heuristics_enabled({"fallback_heuristics": False}))

        # Top-level fallback_heuristics_enabled
        self.assertTrue(is_fallback_heuristics_enabled({"fallback_heuristics_enabled": True}))

        # agent.fallback_heuristics
        self.assertTrue(is_fallback_heuristics_enabled({"agent": {"fallback_heuristics": True}}))

        # fallback.heuristics
        self.assertTrue(is_fallback_heuristics_enabled({"fallback": {"heuristics": True}}))

    def test_entry_level_override(self):
        from hermes_cli.fallback_config import is_fallback_heuristics_enabled
        # Entry heuristic_enabled: True wins over global False
        entry_on = {"heuristic_enabled": True}
        self.assertTrue(is_fallback_heuristics_enabled({"fallback_heuristics": False}, entry=entry_on))

        # Entry heuristic_enabled: False wins over global True
        entry_off = {"heuristic_enabled": False}
        self.assertFalse(is_fallback_heuristics_enabled({"fallback_heuristics": True}, entry=entry_off))

    def test_chain_filtering_when_disabled_vs_enabled(self):
        # When disabled (default): heuristic entries with no static model are skipped
        config_disabled = {
            "fallback_heuristics": False,
            "fallback_providers": [
                {"heuristic": "largest_parameter_count_free"},
                {"provider": "anthropic", "model": "claude-sonnet-5"},
            ]
        }
        chain = get_fallback_chain(config_disabled)
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0]["provider"], "anthropic")
        self.assertEqual(chain[0]["model"], "claude-sonnet-5")

        # When enabled: heuristic entries resolve to candidates
        config_enabled = {
            "fallback_heuristics": True,
            "fallback_providers": [
                {"heuristic": "largest_parameter_count_free", "max_candidates": 2},
                {"provider": "anthropic", "model": "claude-sonnet-5"},
            ]
        }
        chain_enabled = get_fallback_chain(config_enabled)
        self.assertEqual(len(chain_enabled), 3)
        self.assertEqual(chain_enabled[0]["provider"], "nous")
        self.assertTrue(chain_enabled[0]["model"])
        self.assertEqual(chain_enabled[-1]["provider"], "anthropic")

    def test_chain_preserves_static_model_on_entry_when_heuristics_disabled(self):
        # An entry with both a static model and a heuristic preserves the static model when heuristics are off
        config = {
            "fallback_heuristics": False,
            "fallback_providers": [
                {
                    "provider": "openrouter",
                    "model": "qwen/qwen-2.5-72b-instruct",
                    "heuristic": "largest_parameter_count_free",
                }
            ]
        }
        chain = get_fallback_chain(config)
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0]["provider"], "openrouter")
        self.assertEqual(chain[0]["model"], "qwen/qwen-2.5-72b-instruct")

    def test_cmd_fallback_heuristics(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from hermes_cli.fallback_cmd import cmd_fallback_heuristics

        saved_config = {}

        def mock_load_config():
            return dict(saved_config)

        def mock_save_config(cfg):
            saved_config.clear()
            saved_config.update(cfg)

        with patch("hermes_cli.config.load_config", side_effect=mock_load_config), \
             patch("hermes_cli.config.save_config", side_effect=mock_save_config):

            # 1. Turn on
            cmd_fallback_heuristics(SimpleNamespace(heuristics_action="on"))
            self.assertTrue(saved_config.get("fallback_heuristics"))

            # 2. Turn off
            cmd_fallback_heuristics(SimpleNamespace(heuristics_action="off"))
            self.assertFalse(saved_config.get("fallback_heuristics"))


if __name__ == "__main__":
    unittest.main()


