"""Tests for the hermes_cli models module."""

from unittest.mock import patch, MagicMock

from hermes_cli.models import (
    OPENROUTER_MODELS, menu_labels, model_ids, detect_provider_for_model,
    filter_nous_free_models, _NOUS_ALLOWED_FREE_MODELS,
    is_nous_free_tier, partition_nous_models_by_tier,
    check_nous_free_tier, clear_nous_free_tier_cache,
    _FREE_TIER_CACHE_TTL,
)
import hermes_cli.models as _models_mod


class TestModelIds:
    def test_returns_non_empty_list(self):
        ids = model_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_ids_match_models_list(self):
        ids = model_ids()
        expected = [mid for mid, _ in OPENROUTER_MODELS]
        assert ids == expected

    def test_all_ids_contain_provider_slash(self):
        """Model IDs should follow the provider/model format."""
        for mid in model_ids():
            assert "/" in mid, f"Model ID '{mid}' missing provider/ prefix"

    def test_no_duplicate_ids(self):
        ids = model_ids()
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"


class TestMenuLabels:
    def test_same_length_as_model_ids(self):
        assert len(menu_labels()) == len(model_ids())

    def test_first_label_marked_recommended(self):
        labels = menu_labels()
        assert "recommended" in labels[0].lower()

    def test_each_label_contains_its_model_id(self):
        for label, mid in zip(menu_labels(), model_ids()):
            assert mid in label, f"Label '{label}' doesn't contain model ID '{mid}'"

    def test_non_recommended_labels_have_no_tag(self):
        """Only the first model should have (recommended)."""
        labels = menu_labels()
        for label in labels[1:]:
            assert "recommended" not in label.lower(), f"Unexpected 'recommended' in '{label}'"


class TestOpenRouterModels:
    def test_structure_is_list_of_tuples(self):
        for entry in OPENROUTER_MODELS:
            assert isinstance(entry, tuple) and len(entry) == 2
            mid, desc = entry
            assert isinstance(mid, str) and len(mid) > 0
            assert isinstance(desc, str)

    def test_at_least_5_models(self):
        """Sanity check that the models list hasn't been accidentally truncated."""
        assert len(OPENROUTER_MODELS) >= 5


class TestFindOpenrouterSlug:
    def test_exact_match(self):
        from hermes_cli.models import _find_openrouter_slug
        assert _find_openrouter_slug("anthropic/claude-opus-4.6") == "anthropic/claude-opus-4.6"

    def test_bare_name_match(self):
        from hermes_cli.models import _find_openrouter_slug
        result = _find_openrouter_slug("claude-opus-4.6")
        assert result == "anthropic/claude-opus-4.6"

    def test_case_insensitive(self):
        from hermes_cli.models import _find_openrouter_slug
        result = _find_openrouter_slug("Anthropic/Claude-Opus-4.6")
        assert result is not None

    def test_unknown_returns_none(self):
        from hermes_cli.models import _find_openrouter_slug
        assert _find_openrouter_slug("totally-fake-model-xyz") is None


class TestDetectProviderForModel:
    def test_anthropic_model_detected(self):
        """claude-opus-4-6 should resolve to anthropic provider."""
        result = detect_provider_for_model("claude-opus-4-6", "openai-codex")
        assert result is not None
        assert result[0] == "anthropic"

    def test_deepseek_model_detected(self):
        """deepseek-chat should resolve to deepseek provider."""
        result = detect_provider_for_model("deepseek-chat", "openai-codex")
        assert result is not None
        # Provider is deepseek (direct) or openrouter (fallback) depending on creds
        assert result[0] in ("deepseek", "openrouter")

    def test_current_provider_model_returns_none(self):
        """Models belonging to the current provider should not trigger a switch."""
        assert detect_provider_for_model("gpt-5.3-codex", "openai-codex") is None

    def test_openrouter_slug_match(self):
        """Models in the OpenRouter catalog should be found."""
        result = detect_provider_for_model("anthropic/claude-opus-4.6", "openai-codex")
        assert result is not None
        assert result[0] == "openrouter"
        assert result[1] == "anthropic/claude-opus-4.6"

    def test_bare_name_gets_openrouter_slug(self, monkeypatch):
        for env_var in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ):
            monkeypatch.delenv(env_var, raising=False)
        """Bare model names should get mapped to full OpenRouter slugs."""
        result = detect_provider_for_model("claude-opus-4.6", "openai-codex")
        assert result is not None
        # Should find it on OpenRouter with full slug
        assert result[1] == "anthropic/claude-opus-4.6"

    def test_unknown_model_returns_none(self):
        """Completely unknown model names should return None."""
        assert detect_provider_for_model("nonexistent-model-xyz", "openai-codex") is None

    def test_aggregator_not_suggested(self):
        """nous/openrouter should never be auto-suggested as target provider."""
        result = detect_provider_for_model("claude-opus-4-6", "openai-codex")
        assert result is not None
        assert result[0] not in ("nous",)  # nous has claude models but shouldn't be suggested


class TestFilterNousFreeModels:
    """Tests for filter_nous_free_models — Nous Portal free-model policy."""

    _PAID = {"prompt": "0.000003", "completion": "0.000015"}
    _FREE = {"prompt": "0", "completion": "0"}

    def test_paid_models_kept(self):
        """Regular paid models pass through unchanged."""
        models = ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]
        pricing = {m: self._PAID for m in models}
        assert filter_nous_free_models(models, pricing) == models

    def test_free_non_allowlist_models_removed(self):
        """Free models NOT in the allowlist are filtered out."""
        models = ["anthropic/claude-opus-4.6", "arcee-ai/trinity-large-preview:free"]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "arcee-ai/trinity-large-preview:free": self._FREE,
        }
        result = filter_nous_free_models(models, pricing)
        assert result == ["anthropic/claude-opus-4.6"]

    def test_allowlist_model_kept_when_free(self):
        """Allowlist models are kept when they report as free."""
        models = ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro"]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "xiaomi/mimo-v2-pro": self._FREE,
        }
        result = filter_nous_free_models(models, pricing)
        assert result == ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro"]

    def test_allowlist_model_removed_when_paid(self):
        """Allowlist models are removed when they are NOT free."""
        models = ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro"]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "xiaomi/mimo-v2-pro": self._PAID,
        }
        result = filter_nous_free_models(models, pricing)
        assert result == ["anthropic/claude-opus-4.6"]

    def test_no_pricing_returns_all(self):
        """When pricing data is unavailable, all models pass through."""
        models = ["anthropic/claude-opus-4.6", "nvidia/nemotron-3-super-120b-a12b:free"]
        assert filter_nous_free_models(models, {}) == models

    def test_model_with_no_pricing_entry_treated_as_paid(self):
        """A model missing from the pricing dict is kept (assumed paid)."""
        models = ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]
        pricing = {"anthropic/claude-opus-4.6": self._PAID}  # gpt-5.4 not in pricing
        result = filter_nous_free_models(models, pricing)
        assert result == models

    def test_mixed_scenario(self):
        """End-to-end: mix of paid, free-allowed, free-disallowed, allowlist-not-free."""
        models = [
            "anthropic/claude-opus-4.6",       # paid, not allowlist → keep
            "nvidia/nemotron-3-super-120b-a12b:free",  # free, not allowlist → drop
            "xiaomi/mimo-v2-pro",              # free, allowlist → keep
            "xiaomi/mimo-v2-omni",             # paid, allowlist → drop
            "openai/gpt-5.4",                  # paid, not allowlist → keep
        ]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "nvidia/nemotron-3-super-120b-a12b:free": self._FREE,
            "xiaomi/mimo-v2-pro": self._FREE,
            "xiaomi/mimo-v2-omni": self._PAID,
            "openai/gpt-5.4": self._PAID,
        }
        result = filter_nous_free_models(models, pricing)
        assert result == [
            "anthropic/claude-opus-4.6",
            "xiaomi/mimo-v2-pro",
            "openai/gpt-5.4",
        ]

    def test_allowlist_contains_expected_models(self):
        """Sanity: the allowlist has the models we expect."""
        assert "xiaomi/mimo-v2-pro" in _NOUS_ALLOWED_FREE_MODELS
        assert "xiaomi/mimo-v2-omni" in _NOUS_ALLOWED_FREE_MODELS


class TestIsNousFreeTier:
    """Tests for is_nous_free_tier — account tier detection."""

    def test_paid_plus_tier(self):
        assert is_nous_free_tier({"subscription": {"plan": "Plus", "tier": 2, "monthly_charge": 20}}) is False

    def test_free_tier_by_charge(self):
        assert is_nous_free_tier({"subscription": {"plan": "Free", "tier": 0, "monthly_charge": 0}}) is True

    def test_no_charge_field_not_free(self):
        """Missing monthly_charge defaults to not-free (don't block users)."""
        assert is_nous_free_tier({"subscription": {"plan": "Free", "tier": 0}}) is False

    def test_plan_name_alone_not_free(self):
        """Plan name alone is not enough — monthly_charge is required."""
        assert is_nous_free_tier({"subscription": {"plan": "free"}}) is False

    def test_empty_subscription_not_free(self):
        """Empty subscription dict defaults to not-free (don't block users)."""
        assert is_nous_free_tier({"subscription": {}}) is False

    def test_no_subscription_not_free(self):
        """Missing subscription key returns False."""
        assert is_nous_free_tier({}) is False

    def test_empty_response_not_free(self):
        """Completely empty response defaults to not-free."""
        assert is_nous_free_tier({}) is False


class TestPartitionNousModelsByTier:
    """Tests for partition_nous_models_by_tier — free vs paid tier model split."""

    _PAID = {"prompt": "0.000003", "completion": "0.000015"}
    _FREE = {"prompt": "0", "completion": "0"}

    def test_paid_tier_all_selectable(self):
        """Paid users get all models as selectable, none unavailable."""
        models = ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro"]
        pricing = {"anthropic/claude-opus-4.6": self._PAID, "xiaomi/mimo-v2-pro": self._FREE}
        sel, unav = partition_nous_models_by_tier(models, pricing, free_tier=False)
        assert sel == models
        assert unav == []

    def test_free_tier_splits_correctly(self):
        """Free users see only free models; paid ones are unavailable."""
        models = ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro", "openai/gpt-5.4"]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "xiaomi/mimo-v2-pro": self._FREE,
            "openai/gpt-5.4": self._PAID,
        }
        sel, unav = partition_nous_models_by_tier(models, pricing, free_tier=True)
        assert sel == ["xiaomi/mimo-v2-pro"]
        assert unav == ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]

    def test_no_pricing_returns_all(self):
        """Without pricing data, all models are selectable."""
        models = ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]
        sel, unav = partition_nous_models_by_tier(models, {}, free_tier=True)
        assert sel == models
        assert unav == []

    def test_all_free_models(self):
        """When all models are free, free-tier users can select all."""
        models = ["xiaomi/mimo-v2-pro", "xiaomi/mimo-v2-omni"]
        pricing = {m: self._FREE for m in models}
        sel, unav = partition_nous_models_by_tier(models, pricing, free_tier=True)
        assert sel == models
        assert unav == []

    def test_all_paid_models(self):
        """When all models are paid, free-tier users have none selectable."""
        models = ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]
        pricing = {m: self._PAID for m in models}
        sel, unav = partition_nous_models_by_tier(models, pricing, free_tier=True)
        assert sel == []
        assert unav == models


class TestCheckNousFreeTierCache:
    """Tests for the TTL cache on check_nous_free_tier()."""

    def setup_method(self):
        """Reset cache before each test."""
        clear_nous_free_tier_cache()

    def teardown_method(self):
        """Reset cache after each test."""
        clear_nous_free_tier_cache()

    @patch("hermes_cli.models.fetch_nous_account_tier")
    @patch("hermes_cli.models.is_nous_free_tier", return_value=True)
    def test_result_is_cached(self, mock_is_free, mock_fetch):
        """Second call within TTL returns cached result without API call."""
        mock_fetch.return_value = {"subscription": {"monthly_charge": 0}}
        with patch("hermes_cli.auth.get_provider_auth_state", return_value={"access_token": "tok"}), \
             patch("hermes_cli.auth.resolve_nous_runtime_credentials"):
            result1 = check_nous_free_tier()
            result2 = check_nous_free_tier()

        assert result1 is True
        assert result2 is True
        # fetch_nous_account_tier should only be called once (cached on second call)
        assert mock_fetch.call_count == 1

    @patch("hermes_cli.models.fetch_nous_account_tier")
    @patch("hermes_cli.models.is_nous_free_tier", return_value=False)
    def test_cache_expires_after_ttl(self, mock_is_free, mock_fetch):
        """After TTL expires, the API is called again."""
        mock_fetch.return_value = {"subscription": {"monthly_charge": 20}}
        with patch("hermes_cli.auth.get_provider_auth_state", return_value={"access_token": "tok"}), \
             patch("hermes_cli.auth.resolve_nous_runtime_credentials"):
            result1 = check_nous_free_tier()
            assert mock_fetch.call_count == 1

            # Simulate TTL expiry by backdating the cache timestamp
            cached_result, cached_at = _models_mod._free_tier_cache
            _models_mod._free_tier_cache = (cached_result, cached_at - _FREE_TIER_CACHE_TTL - 1)

            result2 = check_nous_free_tier()
            assert mock_fetch.call_count == 2

        assert result1 is False
        assert result2 is False

    def test_clear_cache_forces_refresh(self):
        """clear_nous_free_tier_cache() invalidates the cached result."""
        # Manually seed the cache
        import time
        _models_mod._free_tier_cache = (True, time.monotonic())

        clear_nous_free_tier_cache()
        assert _models_mod._free_tier_cache is None

    def test_cache_ttl_is_short(self):
        """TTL should be short enough to catch upgrades quickly (<=5 min)."""
        assert _FREE_TIER_CACHE_TTL <= 300
