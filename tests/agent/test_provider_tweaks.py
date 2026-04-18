"""Unit tests for agent.provider_tweaks."""

import pytest

from agent.provider_tweaks import get_provider_tweaks, merge_provider_tweaks


# ── get_provider_tweaks ────────────────────────────────────────────────────


class TestGetProviderTweaks:
    def test_returns_empty_for_non_openrouter_base_url(self):
        assert get_provider_tweaks("minimax/minimax-m2.7", "https://api.minimax.io/v1") == {}

    def test_returns_empty_for_missing_base_url(self):
        assert get_provider_tweaks("minimax/minimax-m2.7", None) == {}
        assert get_provider_tweaks("minimax/minimax-m2.7", "") == {}

    def test_returns_empty_for_missing_model(self):
        assert get_provider_tweaks(None, "https://openrouter.ai/api/v1") == {}
        assert get_provider_tweaks("", "https://openrouter.ai/api/v1") == {}

    def test_returns_empty_for_unmatched_model_on_openrouter(self):
        assert get_provider_tweaks("anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1") == {}
        assert get_provider_tweaks("openai/gpt-5.4", "https://openrouter.ai/api/v1") == {}
        assert get_provider_tweaks("deepseek/deepseek-chat", "https://openrouter.ai/api/v1") == {}

    def test_minimax_m27_on_openrouter_gets_ignore_and_order(self):
        t = get_provider_tweaks("minimax/minimax-m2.7", "https://openrouter.ai/api/v1")
        assert t != {}
        assert t["ignore"] == ["minimax"]
        assert "fireworks" in t["order"]
        assert "novitaai" in t["order"]
        assert t["ref"]
        assert t["reason"]

    def test_minimax_m2_base_also_matches(self):
        t = get_provider_tweaks("minimax/minimax-m2", "https://openrouter.ai/api/v1")
        assert t["ignore"] == ["minimax"]

    def test_minimax_m21_also_matches(self):
        t = get_provider_tweaks("minimax/minimax-m2.1", "https://openrouter.ai/api/v1")
        assert t["ignore"] == ["minimax"]

    def test_case_insensitive_model_match(self):
        t = get_provider_tweaks("MiniMax/MiniMax-M2.7", "https://openrouter.ai/api/v1")
        assert t["ignore"] == ["minimax"]

    def test_case_insensitive_base_url_match(self):
        t = get_provider_tweaks("minimax/minimax-m2.7", "https://OpenRouter.AI/api/v1")
        assert t["ignore"] == ["minimax"]

    def test_openrouter_subpath_still_matched(self):
        # Proxied OpenRouter deployments still route through openrouter.ai
        t = get_provider_tweaks("minimax/minimax-m2.7", "https://proxy.example.com/openrouter.ai/api/v1")
        assert t["ignore"] == ["minimax"]

    def test_returned_lists_are_copies_not_references(self):
        t1 = get_provider_tweaks("minimax/minimax-m2.7", "https://openrouter.ai/api/v1")
        t1["ignore"].append("pwned")
        t2 = get_provider_tweaks("minimax/minimax-m2.7", "https://openrouter.ai/api/v1")
        assert "pwned" not in t2["ignore"]


# ── merge_provider_tweaks ──────────────────────────────────────────────────


class TestMergeProviderTweaks:
    def test_empty_tweaks_returns_input_unchanged(self):
        assert merge_provider_tweaks({"order": ["x"]}, {}) == {"order": ["x"]}
        assert merge_provider_tweaks(None, {}) == {}
        assert merge_provider_tweaks({}, {}) == {}

    def test_empty_preferences_gets_tweak_defaults(self):
        t = {"ignore": ["minimax"], "order": ["fireworks"], "reason": "x", "ref": "y"}
        merged = merge_provider_tweaks({}, t)
        assert merged["ignore"] == ["minimax"]
        assert merged["order"] == ["fireworks"]

    def test_none_preferences_gets_tweak_defaults(self):
        t = {"ignore": ["minimax"], "order": ["fireworks"], "reason": "x", "ref": "y"}
        merged = merge_provider_tweaks(None, t)
        assert merged["ignore"] == ["minimax"]
        assert merged["order"] == ["fireworks"]

    def test_user_ignore_wins_over_tweaks(self):
        t = {"ignore": ["minimax"], "order": ["fireworks"], "reason": "x", "ref": "y"}
        merged = merge_provider_tweaks({"ignore": ["together"]}, t)
        assert merged["ignore"] == ["together"]
        # But order still filled from tweaks since user didn't set it
        assert merged["order"] == ["fireworks"]

    def test_user_order_wins_over_tweaks(self):
        t = {"ignore": ["minimax"], "order": ["fireworks"], "reason": "x", "ref": "y"}
        merged = merge_provider_tweaks({"order": ["novitaai"]}, t)
        assert merged["order"] == ["novitaai"]
        # But ignore still filled from tweaks
        assert merged["ignore"] == ["minimax"]

    def test_user_only_disables_all_tweaks(self):
        """When user whitelists specific providers, don't layer ignore/order on top."""
        t = {"ignore": ["minimax"], "order": ["fireworks"], "reason": "x", "ref": "y"}
        merged = merge_provider_tweaks({"only": ["minimax"]}, t)
        assert merged == {"only": ["minimax"]}
        assert "ignore" not in merged
        assert "order" not in merged

    def test_does_not_mutate_input_preferences(self):
        prefs = {"order": ["novitaai"]}
        t = {"ignore": ["minimax"], "order": ["fireworks"], "reason": "x", "ref": "y"}
        merge_provider_tweaks(prefs, t)
        # Input wasn't modified
        assert prefs == {"order": ["novitaai"]}
        assert "ignore" not in prefs

    def test_does_not_mutate_input_tweaks(self):
        prefs = {}
        t = {"ignore": ["minimax"], "order": ["fireworks"], "reason": "x", "ref": "y"}
        merged = merge_provider_tweaks(prefs, t)
        merged["ignore"].append("pwned")
        assert t["ignore"] == ["minimax"]

    def test_preserves_unrelated_preference_keys(self):
        t = {"ignore": ["minimax"], "order": ["fireworks"], "reason": "x", "ref": "y"}
        merged = merge_provider_tweaks(
            {"sort": "throughput", "data_collection": "deny"},
            t,
        )
        assert merged["sort"] == "throughput"
        assert merged["data_collection"] == "deny"
        assert merged["ignore"] == ["minimax"]
        assert merged["order"] == ["fireworks"]


# ── Integration: the concrete MiniMax case from PR #12072 ──────────────────


class TestMinimaxConcreteCase:
    """End-to-end behaviour for the exact scenario that motivated this module.

    MiniMax direct OpenRouter endpoint has documented non-terminating streams
    on tool-calling workloads (MiniMax-M2 issue #109).  The tweaks module
    must automatically route `minimax/*` requests away from that endpoint
    on OpenRouter, while leaving user-supplied preferences untouched.
    """

    def test_fresh_minimax_request_gets_fireworks_routing(self):
        t = get_provider_tweaks("minimax/minimax-m2.7", "https://openrouter.ai/api/v1")
        merged = merge_provider_tweaks({}, t)
        assert "minimax" in merged["ignore"]
        # Fireworks is the empirically-best provider for m2.7 (confirmed
        # 2026-04-18: 99% uptime, 75 tok/s p50, clean tool-call streams)
        assert merged["order"][0] == "fireworks"

    def test_user_force_minimax_still_honoured(self):
        """User who explicitly wants to test the broken endpoint gets it."""
        t = get_provider_tweaks("minimax/minimax-m2.7", "https://openrouter.ai/api/v1")
        merged = merge_provider_tweaks({"only": ["minimax"]}, t)
        # User's explicit whitelist wins, even though we "know better".
        assert merged == {"only": ["minimax"]}

    def test_anthropic_model_on_openrouter_unaffected(self):
        """Tweaks only trigger on the specific buggy model family."""
        t = get_provider_tweaks("anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1")
        assert t == {}
        merged = merge_provider_tweaks({}, t)
        assert merged == {}
