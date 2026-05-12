"""Tests for the hermes_cli models module."""

from unittest.mock import patch, MagicMock

from hermes_cli.models import (
    OPENROUTER_MODELS, fetch_openrouter_models, model_ids, detect_provider_for_model,
    is_nous_free_tier, partition_nous_models_by_tier,
    check_nous_free_tier, _FREE_TIER_CACHE_TTL,
    union_with_portal_free_recommendations,
)
import hermes_cli.models as _models_mod

LIVE_OPENROUTER_MODELS = [
    ("anthropic/claude-opus-4.6", "recommended"),
    ("qwen/qwen3.6-plus", ""),
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
]



class TestModelIds:
    def test_returns_non_empty_list(self):
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            ids = model_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_ids_match_fetched_catalog(self):
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            ids = model_ids()
        expected = [mid for mid, _ in LIVE_OPENROUTER_MODELS]
        assert ids == expected

    def test_all_ids_contain_provider_slash(self):
        """Model IDs should follow the provider/model format."""
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            for mid in model_ids():
                assert "/" in mid, f"Model ID '{mid}' missing provider/ prefix"

    def test_no_duplicate_ids(self):
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            ids = model_ids()
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"





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


class TestFetchOpenRouterModels:
    def test_live_fetch_recomputes_free_tags(self, monkeypatch):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data":[{"id":"anthropic/claude-opus-4.6","pricing":{"prompt":"0.000015","completion":"0.000075"}},{"id":"qwen/qwen3.6-plus","pricing":{"prompt":"0.000000325","completion":"0.00000195"}},{"id":"nvidia/nemotron-3-super-120b-a12b:free","pricing":{"prompt":"0","completion":"0"}}]}'

        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        with patch("hermes_cli.models.urllib.request.urlopen", return_value=_Resp()):
            models = fetch_openrouter_models(force_refresh=True)

        assert models == [
            ("anthropic/claude-opus-4.6", "recommended"),
            ("qwen/qwen3.6-plus", ""),
            ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
        ]

    def test_falls_back_to_static_snapshot_on_fetch_failure(self, monkeypatch):
        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        with patch("hermes_cli.models.urllib.request.urlopen", side_effect=OSError("boom")):
            models = fetch_openrouter_models(force_refresh=True)

        assert models == OPENROUTER_MODELS

    def test_filters_out_models_without_tool_support(self, monkeypatch):
        """Models whose supported_parameters omits 'tools' must not appear in the picker.

        hermes-agent is tool-calling-first — surfacing a non-tool model leads to
        immediate runtime failures when the user selects it. Ported from
        Kilo-Org/kilocode#9068.
        """
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                # opus-4.6 advertises tools → kept
                # nano-image has explicit supported_parameters that OMITS tools → dropped
                # qwen3.6-plus advertises tools → kept
                return (
                    b'{"data":['
                    b'{"id":"anthropic/claude-opus-4.6","pricing":{"prompt":"0.000015","completion":"0.000075"},'
                    b'"supported_parameters":["temperature","tools","tool_choice"]},'
                    b'{"id":"google/gemini-3-pro-image-preview","pricing":{"prompt":"0.00001","completion":"0.00003"},'
                    b'"supported_parameters":["temperature","response_format"]},'
                    b'{"id":"qwen/qwen3.6-plus","pricing":{"prompt":"0.000000325","completion":"0.00000195"},'
                    b'"supported_parameters":["tools","temperature"]}'
                    b']}'
                )

        # Include the image-only id in the curated list so it has a chance to be surfaced.
        monkeypatch.setattr(
            _models_mod,
            "OPENROUTER_MODELS",
            [
                ("anthropic/claude-opus-4.6", ""),
                ("google/gemini-3-pro-image-preview", ""),
                ("qwen/qwen3.6-plus", ""),
            ],
        )
        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        with patch("hermes_cli.models.urllib.request.urlopen", return_value=_Resp()):
            models = fetch_openrouter_models(force_refresh=True)

        ids = [mid for mid, _ in models]
        assert "anthropic/claude-opus-4.6" in ids
        assert "qwen/qwen3.6-plus" in ids
        # Image-only model advertised supported_parameters WITHOUT tools → must be dropped.
        assert "google/gemini-3-pro-image-preview" not in ids

    def test_permissive_when_supported_parameters_missing(self, monkeypatch):
        """Models missing the supported_parameters field keep appearing in the picker.

        Some OpenRouter-compatible gateways (Nous Portal, private mirrors, older
        catalog snapshots) don't populate supported_parameters. Treating missing
        as 'unknown → allow' prevents the picker from silently emptying on
        those gateways.
        """
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                # No supported_parameters field at all on either entry.
                return (
                    b'{"data":['
                    b'{"id":"anthropic/claude-opus-4.6","pricing":{"prompt":"0.000015","completion":"0.000075"}},'
                    b'{"id":"qwen/qwen3.6-plus","pricing":{"prompt":"0.000000325","completion":"0.00000195"}}'
                    b']}'
                )

        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        with patch("hermes_cli.models.urllib.request.urlopen", return_value=_Resp()):
            models = fetch_openrouter_models(force_refresh=True)

        ids = [mid for mid, _ in models]
        assert "anthropic/claude-opus-4.6" in ids
        assert "qwen/qwen3.6-plus" in ids


class TestOpenRouterToolSupportHelper:
    """Unit tests for _openrouter_model_supports_tools (Kilo port #9068)."""

    def test_tools_in_supported_parameters(self):
        from hermes_cli.models import _openrouter_model_supports_tools
        assert _openrouter_model_supports_tools(
            {"id": "x", "supported_parameters": ["temperature", "tools"]}
        ) is True

    def test_tools_missing_from_supported_parameters(self):
        from hermes_cli.models import _openrouter_model_supports_tools
        assert _openrouter_model_supports_tools(
            {"id": "x", "supported_parameters": ["temperature", "response_format"]}
        ) is False

    def test_supported_parameters_absent_is_permissive(self):
        """Missing field → allow (so older / non-OR gateways still work)."""
        from hermes_cli.models import _openrouter_model_supports_tools
        assert _openrouter_model_supports_tools({"id": "x"}) is True

    def test_supported_parameters_none_is_permissive(self):
        from hermes_cli.models import _openrouter_model_supports_tools
        assert _openrouter_model_supports_tools({"id": "x", "supported_parameters": None}) is True

    def test_supported_parameters_malformed_is_permissive(self):
        """Malformed (non-list) value → allow rather than silently drop."""
        from hermes_cli.models import _openrouter_model_supports_tools
        assert _openrouter_model_supports_tools(
            {"id": "x", "supported_parameters": "tools,temperature"}
        ) is True

    def test_non_dict_item_is_permissive(self):
        from hermes_cli.models import _openrouter_model_supports_tools
        assert _openrouter_model_supports_tools(None) is True
        assert _openrouter_model_supports_tools("anthropic/claude-opus-4.6") is True

    def test_empty_supported_parameters_list_drops_model(self):
        """Explicit empty list → no tools → drop."""
        from hermes_cli.models import _openrouter_model_supports_tools
        assert _openrouter_model_supports_tools(
            {"id": "x", "supported_parameters": []}
        ) is False


class TestFindOpenrouterSlug:
    def test_exact_match(self):
        from hermes_cli.models import _find_openrouter_slug
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            assert _find_openrouter_slug("anthropic/claude-opus-4.6") == "anthropic/claude-opus-4.6"

    def test_bare_name_match(self):
        from hermes_cli.models import _find_openrouter_slug
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            result = _find_openrouter_slug("claude-opus-4.6")
        assert result == "anthropic/claude-opus-4.6"

    def test_case_insensitive(self):
        from hermes_cli.models import _find_openrouter_slug
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            result = _find_openrouter_slug("Anthropic/Claude-Opus-4.6")
        assert result is not None

    def test_unknown_returns_none(self):
        from hermes_cli.models import _find_openrouter_slug
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            assert _find_openrouter_slug("totally-fake-model-xyz") is None


class TestDetectProviderForModel:
    def test_anthropic_model_detected(self):
        """claude-opus-4-6 should resolve to anthropic provider."""
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
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

    def test_short_alias_resolves_to_static_model(self):
        """Short aliases (e.g. sonnet) should resolve without network lookups."""
        with patch(
            "hermes_cli.models.fetch_openrouter_models",
            side_effect=AssertionError("network lookup should not run"),
        ):
            result = detect_provider_for_model("sonnet", "auto")
        assert result is not None
        assert result[0] == "anthropic"
        assert result[1].startswith("claude-sonnet")

    def test_openrouter_slug_match(self):
        """Models in the OpenRouter catalog should be found."""
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
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
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            result = detect_provider_for_model("claude-opus-4.6", "openai-codex")
        assert result is not None
        # Should find it on OpenRouter with full slug
        assert result[1] == "anthropic/claude-opus-4.6"

    def test_unknown_model_returns_none(self):
        """Completely unknown model names should return None."""
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            assert detect_provider_for_model("nonexistent-model-xyz", "openai-codex") is None

    def test_aggregator_not_suggested(self):
        """nous/openrouter should never be auto-suggested as target provider."""
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            result = detect_provider_for_model("claude-opus-4-6", "openai-codex")
        assert result is not None
        assert result[0] not in ("nous",)  # nous has claude models but shouldn't be suggested


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


class TestUnionWithPortalFreeRecommendations:
    """Tests for union_with_portal_free_recommendations.

    The Portal's freeRecommendedModels endpoint is the source of truth for
    what's free *right now* — the in-repo curated list and docs-hosted
    manifest can lag. This helper guarantees the picker still surfaces
    Portal-flagged free models even when the rest of the catalog is stale.
    """

    _PAID = {"prompt": "0.000003", "completion": "0.000015"}
    _FREE = {"prompt": "0", "completion": "0"}

    def _payload(self, free_models: list[str]) -> dict:
        return {
            "freeRecommendedModels": [
                {"modelName": mid, "displayName": mid} for mid in free_models
            ],
        }

    def test_adds_portal_free_model_missing_from_curated(self):
        """A Portal-advertised free model not in curated is prepended + priced free."""
        curated = ["anthropic/claude-opus-4.6"]
        pricing = {"anthropic/claude-opus-4.6": self._PAID}
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value=self._payload(["qwen/qwen3.6-plus"]),
        ):
            ids, p = union_with_portal_free_recommendations(curated, pricing, "")

        assert ids[0] == "qwen/qwen3.6-plus"  # prepended
        assert "anthropic/claude-opus-4.6" in ids
        # Synthetic free pricing entry created
        assert p["qwen/qwen3.6-plus"] == self._FREE
        # Existing pricing untouched
        assert p["anthropic/claude-opus-4.6"] == self._PAID

    def test_does_not_duplicate_curated_entries(self):
        """A Portal free model already in curated is not duplicated."""
        curated = ["qwen/qwen3.6-plus", "anthropic/claude-opus-4.6"]
        pricing = {
            "qwen/qwen3.6-plus": self._FREE,
            "anthropic/claude-opus-4.6": self._PAID,
        }
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value=self._payload(["qwen/qwen3.6-plus"]),
        ):
            ids, p = union_with_portal_free_recommendations(curated, pricing, "")

        assert ids == curated
        assert p == pricing

    def test_then_partition_keeps_portal_free_model(self):
        """End-to-end: Portal-flagged free model survives partition."""
        # Simulate the broken-state-before-this-fix: in-repo curated list
        # contains qwen/qwen3.6-plus (because new builds shipped it) but
        # live pricing endpoint hasn't published its zero-cost entry yet.
        # The Portal's freeRecommendedModels still flags it as free.
        curated = ["qwen/qwen3.6-plus", "anthropic/claude-opus-4.6"]
        pricing = {"anthropic/claude-opus-4.6": self._PAID}  # qwen missing!
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value=self._payload(["qwen/qwen3.6-plus"]),
        ):
            ids, p = union_with_portal_free_recommendations(curated, pricing, "")
        sel, unav = partition_nous_models_by_tier(ids, p, free_tier=True)
        assert "qwen/qwen3.6-plus" in sel
        assert "anthropic/claude-opus-4.6" in unav

    def test_empty_payload_returns_inputs_unchanged(self):
        """Empty Portal response leaves curated + pricing untouched."""
        curated = ["a", "b"]
        pricing = {"a": self._PAID}
        with patch("hermes_cli.models.fetch_nous_recommended_models", return_value={}):
            ids, p = union_with_portal_free_recommendations(curated, pricing, "")
        assert ids == curated
        assert p == pricing

    def test_missing_freeRecommendedModels_key(self):
        """Portal payload without freeRecommendedModels degrades gracefully."""
        curated = ["a"]
        pricing = {"a": self._PAID}
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value={"paidRecommendedModels": [{"modelName": "x"}]},
        ):
            ids, p = union_with_portal_free_recommendations(curated, pricing, "")
        assert ids == curated
        assert p == pricing

    def test_fetch_failure_returns_inputs(self):
        """Network failures don't blow up the picker."""
        curated = ["a"]
        pricing = {"a": self._PAID}
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            side_effect=RuntimeError("network down"),
        ):
            ids, p = union_with_portal_free_recommendations(curated, pricing, "")
        assert ids == curated
        assert p == pricing

    def test_invalid_entries_skipped(self):
        """Non-dict / missing-modelName entries are filtered out."""
        curated = ["a"]
        pricing = {"a": self._PAID}
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value={
                "freeRecommendedModels": [
                    "not-a-dict",
                    {"displayName": "no-modelName"},
                    {"modelName": ""},
                    {"modelName": "qwen/qwen3.6-plus"},
                ]
            },
        ):
            ids, p = union_with_portal_free_recommendations(curated, pricing, "")
        assert ids == ["qwen/qwen3.6-plus", "a"]
        assert p["qwen/qwen3.6-plus"] == self._FREE


class TestCheckNousFreeTierCache:
    """Tests for the TTL cache on check_nous_free_tier()."""

    def setup_method(self):
        _models_mod._free_tier_cache = None

    def teardown_method(self):
        _models_mod._free_tier_cache = None

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

            cached_result, cached_at = _models_mod._free_tier_cache
            _models_mod._free_tier_cache = (cached_result, cached_at - _FREE_TIER_CACHE_TTL - 1)

            result2 = check_nous_free_tier()
            assert mock_fetch.call_count == 2

        assert result1 is False
        assert result2 is False

    def test_cache_ttl_is_short(self):
        """TTL should be short enough to catch upgrades quickly (<=5 min)."""
        assert _FREE_TIER_CACHE_TTL <= 300


class TestNousRecommendedModels:
    """Tests for fetch_nous_recommended_models + get_nous_recommended_aux_model."""

    _SAMPLE_PAYLOAD = {
        "paidRecommendedModels": [],
        "freeRecommendedModels": [],
        "paidRecommendedCompactionModel": None,
        "paidRecommendedVisionModel": None,
        "freeRecommendedCompactionModel": {
            "modelName": "google/gemini-3-flash-preview",
            "displayName": "Google: Gemini 3 Flash Preview",
        },
        "freeRecommendedVisionModel": {
            "modelName": "google/gemini-3-flash-preview",
            "displayName": "Google: Gemini 3 Flash Preview",
        },
    }

    def setup_method(self):
        _models_mod._nous_recommended_cache.clear()

    def teardown_method(self):
        _models_mod._nous_recommended_cache.clear()

    def _mock_urlopen(self, payload):
        """Return a context-manager mock mimicking urllib.request.urlopen()."""
        import json as _json
        response = MagicMock()
        response.read.return_value = _json.dumps(payload).encode()
        cm = MagicMock()
        cm.__enter__.return_value = response
        cm.__exit__.return_value = False
        return cm

    def test_fetch_caches_per_portal_url(self):
        from hermes_cli.models import fetch_nous_recommended_models
        mock_cm = self._mock_urlopen(self._SAMPLE_PAYLOAD)
        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            a = fetch_nous_recommended_models("https://portal.example.com")
            b = fetch_nous_recommended_models("https://portal.example.com")
        assert a == self._SAMPLE_PAYLOAD
        assert b == self._SAMPLE_PAYLOAD
        assert mock_urlopen.call_count == 1  # second call served from cache

    def test_fetch_cache_is_keyed_per_portal(self):
        from hermes_cli.models import fetch_nous_recommended_models
        mock_cm = self._mock_urlopen(self._SAMPLE_PAYLOAD)
        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            fetch_nous_recommended_models("https://portal.example.com")
            fetch_nous_recommended_models("https://portal.staging-nousresearch.com")
        assert mock_urlopen.call_count == 2  # different portals → separate fetches

    def test_fetch_returns_empty_on_network_failure(self):
        from hermes_cli.models import fetch_nous_recommended_models
        with patch("urllib.request.urlopen", side_effect=OSError("boom")):
            result = fetch_nous_recommended_models("https://portal.example.com")
        assert result == {}

    def test_fetch_force_refresh_bypasses_cache(self):
        from hermes_cli.models import fetch_nous_recommended_models
        mock_cm = self._mock_urlopen(self._SAMPLE_PAYLOAD)
        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_urlopen:
            fetch_nous_recommended_models("https://portal.example.com")
            fetch_nous_recommended_models("https://portal.example.com", force_refresh=True)
        assert mock_urlopen.call_count == 2

    def test_get_aux_model_returns_vision_recommendation(self):
        from hermes_cli.models import get_nous_recommended_aux_model
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value=self._SAMPLE_PAYLOAD,
        ):
            # Free tier → free vision recommendation.
            model = get_nous_recommended_aux_model(vision=True, free_tier=True)
        assert model == "google/gemini-3-flash-preview"

    def test_get_aux_model_returns_compaction_recommendation(self):
        from hermes_cli.models import get_nous_recommended_aux_model
        payload = dict(self._SAMPLE_PAYLOAD)
        payload["freeRecommendedCompactionModel"] = {"modelName": "minimax/minimax-m2.7"}
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value=payload,
        ):
            model = get_nous_recommended_aux_model(vision=False, free_tier=True)
        assert model == "minimax/minimax-m2.7"

    def test_get_aux_model_returns_none_when_field_null(self):
        from hermes_cli.models import get_nous_recommended_aux_model
        payload = dict(self._SAMPLE_PAYLOAD)
        payload["freeRecommendedCompactionModel"] = None
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value=payload,
        ):
            model = get_nous_recommended_aux_model(vision=False, free_tier=True)
        assert model is None

    def test_get_aux_model_returns_none_on_empty_payload(self):
        from hermes_cli.models import get_nous_recommended_aux_model
        with patch("hermes_cli.models.fetch_nous_recommended_models", return_value={}):
            assert get_nous_recommended_aux_model(vision=False, free_tier=True) is None
            assert get_nous_recommended_aux_model(vision=True, free_tier=False) is None

    def test_get_aux_model_returns_none_when_modelname_blank(self):
        from hermes_cli.models import get_nous_recommended_aux_model
        payload = {"freeRecommendedCompactionModel": {"modelName": "  "}}
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value=payload,
        ):
            assert get_nous_recommended_aux_model(vision=False, free_tier=True) is None

    def test_paid_tier_prefers_paid_recommendation(self):
        """Paid-tier users should get the paid model when it's populated."""
        from hermes_cli.models import get_nous_recommended_aux_model
        payload = {
            "paidRecommendedCompactionModel": {"modelName": "anthropic/claude-opus-4.7"},
            "freeRecommendedCompactionModel": {"modelName": "google/gemini-3-flash-preview"},
            "paidRecommendedVisionModel": {"modelName": "openai/gpt-5.4"},
            "freeRecommendedVisionModel": {"modelName": "google/gemini-3-flash-preview"},
        }
        with patch("hermes_cli.models.fetch_nous_recommended_models", return_value=payload):
            text = get_nous_recommended_aux_model(vision=False, free_tier=False)
            vision = get_nous_recommended_aux_model(vision=True, free_tier=False)
        assert text == "anthropic/claude-opus-4.7"
        assert vision == "openai/gpt-5.4"

    def test_paid_tier_falls_back_to_free_when_paid_is_null(self):
        """If the Portal returns null for the paid field, fall back to free."""
        from hermes_cli.models import get_nous_recommended_aux_model
        payload = {
            "paidRecommendedCompactionModel": None,
            "freeRecommendedCompactionModel": {"modelName": "google/gemini-3-flash-preview"},
            "paidRecommendedVisionModel": None,
            "freeRecommendedVisionModel": {"modelName": "google/gemini-3-flash-preview"},
        }
        with patch("hermes_cli.models.fetch_nous_recommended_models", return_value=payload):
            text = get_nous_recommended_aux_model(vision=False, free_tier=False)
            vision = get_nous_recommended_aux_model(vision=True, free_tier=False)
        assert text == "google/gemini-3-flash-preview"
        assert vision == "google/gemini-3-flash-preview"

    def test_free_tier_never_uses_paid_recommendation(self):
        """Free-tier users must not get paid-only recommendations."""
        from hermes_cli.models import get_nous_recommended_aux_model
        payload = {
            "paidRecommendedCompactionModel": {"modelName": "anthropic/claude-opus-4.7"},
            "freeRecommendedCompactionModel": None,  # no free recommendation
        }
        with patch("hermes_cli.models.fetch_nous_recommended_models", return_value=payload):
            model = get_nous_recommended_aux_model(vision=False, free_tier=True)
        # Free tier must return None — never leak the paid model.
        assert model is None

    def test_auto_detects_tier_when_not_supplied(self):
        """Default behaviour: call check_nous_free_tier() to pick the tier."""
        from hermes_cli.models import get_nous_recommended_aux_model
        payload = {
            "paidRecommendedCompactionModel": {"modelName": "paid-model"},
            "freeRecommendedCompactionModel": {"modelName": "free-model"},
        }
        with (
            patch("hermes_cli.models.fetch_nous_recommended_models", return_value=payload),
            patch("hermes_cli.models.check_nous_free_tier", return_value=True),
        ):
            assert get_nous_recommended_aux_model(vision=False) == "free-model"
        with (
            patch("hermes_cli.models.fetch_nous_recommended_models", return_value=payload),
            patch("hermes_cli.models.check_nous_free_tier", return_value=False),
        ):
            assert get_nous_recommended_aux_model(vision=False) == "paid-model"

    def test_tier_detection_error_defaults_to_paid(self):
        """If tier detection raises, assume paid so we don't downgrade silently."""
        from hermes_cli.models import get_nous_recommended_aux_model
        payload = {
            "paidRecommendedCompactionModel": {"modelName": "paid-model"},
            "freeRecommendedCompactionModel": {"modelName": "free-model"},
        }
        with (
            patch("hermes_cli.models.fetch_nous_recommended_models", return_value=payload),
            patch("hermes_cli.models.check_nous_free_tier", side_effect=RuntimeError("boom")),
        ):
            assert get_nous_recommended_aux_model(vision=False) == "paid-model"
