"""Tests for the hermes_cli models module."""

from unittest.mock import patch, MagicMock

from hermes_cli.nous_account import NousPortalAccountInfo
from hermes_cli.models import (
    OPENROUTER_MODELS, fetch_openrouter_models, model_ids, detect_provider_for_model,
    is_nous_free_tier, partition_nous_models_by_tier,
    check_nous_free_tier, _FREE_TIER_CACHE_TTL,
    union_with_portal_free_recommendations,
    union_with_portal_paid_recommendations,
)
import hermes_cli.models as _models_mod

LIVE_OPENROUTER_MODELS = [
    ("anthropic/claude-opus-4.6", "recommended"),
    ("qwen/qwen3.7-max", ""),
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
]


class TestModelIds:
    def test_returns_non_empty_list(self):
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            ids = model_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0


class TestOpenRouterModels:
    def test_structure_is_list_of_tuples(self):
        for entry in OPENROUTER_MODELS:
            assert isinstance(entry, tuple) and len(entry) == 2
            mid, desc = entry
            assert isinstance(mid, str) and len(mid) > 0
            assert isinstance(desc, str)


class TestFetchOpenRouterModels:


    def test_falls_back_to_static_snapshot_on_fetch_failure(self, monkeypatch):
        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        # Pin the remote manifest out too — otherwise the fallback silently
        # depends on whatever the deployed catalog currently contains.
        with patch("hermes_cli.model_catalog.get_curated_openrouter_models", return_value=None), \
             patch("hermes_cli.models._urlopen_model_catalog_request", side_effect=OSError("boom")):
            models = fetch_openrouter_models(force_refresh=True)

        assert models == OPENROUTER_MODELS

    def test_fetches_extra_models_from_config(self, monkeypatch):
        """User-configured openrouter.extra_models are appended to the picker list."""
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"data":['
                    b'{"id":"anthropic/claude-opus-4.6","pricing":{"prompt":"0.000015","completion":"0.000075"},'
                    b'"supported_parameters":["temperature","tools"]},'
                    b'{"id":"google/gemini-2.5-flash","pricing":{"prompt":"0.00001","completion":"0.00003"},'
                    b'"supported_parameters":["temperature","tools"]}'
                    b']}'
                )

        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        mock_config = {"openrouter": {"extra_models": ["google/gemini-2.5-flash"]}}
        with (
            patch("hermes_cli.model_catalog.get_curated_openrouter_models", return_value=[("anthropic/claude-opus-4.6", "")]),
            patch("hermes_cli.config.load_config", return_value=mock_config),
            patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_Resp()),
        ):
            models = fetch_openrouter_models(force_refresh=True)

        ids = [mid for mid, _ in models]
        assert "anthropic/claude-opus-4.6" in ids
        assert "google/gemini-2.5-flash" in ids
        assert ("google/gemini-2.5-flash", "") in models

    def test_extra_models_in_fallback_when_offline(self, monkeypatch):
        """User-configured openrouter.extra_models remain in fallback list during fetch failure."""
        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        mock_config = {"openrouter": {"extra_models": ["google/gemini-2.5-flash"]}}
        with (
            patch("hermes_cli.model_catalog.get_curated_openrouter_models", return_value=[("anthropic/claude-opus-4.6", "")]),
            patch("hermes_cli.config.load_config", return_value=mock_config),
            patch("hermes_cli.models._urlopen_model_catalog_request", side_effect=OSError("offline")),
        ):
            models = fetch_openrouter_models(force_refresh=True)

        ids = [mid for mid, _ in models]
        assert "anthropic/claude-opus-4.6" in ids
        assert "google/gemini-2.5-flash" in ids

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
                # qwen3.7-max advertises tools → kept
                return (
                    b'{"data":['
                    b'{"id":"anthropic/claude-opus-4.6","pricing":{"prompt":"0.000015","completion":"0.000075"},'
                    b'"supported_parameters":["temperature","tools","tool_choice"]},'
                    b'{"id":"google/gemini-3-pro-image-preview","pricing":{"prompt":"0.00001","completion":"0.00003"},'
                    b'"supported_parameters":["temperature","response_format"]},'
                    b'{"id":"qwen/qwen3.7-max","pricing":{"prompt":"0.000000325","completion":"0.00000195"},'
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
                ("qwen/qwen3.7-max", ""),
            ],
        )
        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        with (
            patch("hermes_cli.model_catalog.get_curated_openrouter_models", return_value=[]),
            patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_Resp()),
        ):
            models = fetch_openrouter_models(force_refresh=True)

        ids = [mid for mid, _ in models]
        assert "anthropic/claude-opus-4.6" in ids
        assert "qwen/qwen3.7-max" in ids
        # Image-only model advertised supported_parameters WITHOUT tools → must be dropped.
        assert "google/gemini-3-pro-image-preview" not in ids



class TestOpenRouterToolSupportHelper:
    """Unit tests for _openrouter_model_supports_tools (Kilo port #9068)."""

    def test_tools_in_supported_parameters(self):
        from hermes_cli.models import _openrouter_model_supports_tools
        assert _openrouter_model_supports_tools(
            {"id": "x", "supported_parameters": ["temperature", "tools"]}
        ) is True


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


class TestDetectProviderForModel:



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





    def test_custom_provider_not_overridden_by_static_catalog(self):
        """When current provider is custom:*, a static-catalog match must NOT
        override it — otherwise a model served by the user's own endpoint gets
        misattributed to a native provider, rewriting model.provider (#48305).

        `gpt-5.4` is in the static openai catalog; with current=custom:foo,
        detection must return None instead of switching to openai.
        """
        assert detect_provider_for_model("gpt-5.4", "custom:foo") is None




class TestIsNousFreeTier:
    """Tests for is_nous_free_tier — account tier detection."""

    def test_paid_service_access_allowed_true_is_not_free(self):
        assert is_nous_free_tier({"paid_service_access": {"allowed": True}}) is False


    def test_empty_subscription_not_free(self):
        """Empty subscription dict defaults to not-free (don't block users)."""
        assert is_nous_free_tier({"subscription": {}}) is False


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
        """A Portal-advertised free model not in curated is appended + priced free."""
        curated = ["anthropic/claude-opus-4.6"]
        pricing = {"anthropic/claude-opus-4.6": self._PAID}
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value=self._payload(["qwen/qwen3.6-plus"]),
        ):
            ids, p = union_with_portal_free_recommendations(curated, pricing, "")

        # Curated ("HA") models stay first; Portal-only picks follow.
        assert ids[0] == "anthropic/claude-opus-4.6"
        assert ids[-1] == "qwen/qwen3.6-plus"  # appended
        # Synthetic free pricing entry created
        assert p["qwen/qwen3.6-plus"] == self._FREE
        # Existing pricing untouched
        assert p["anthropic/claude-opus-4.6"] == self._PAID




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


class TestUnionWithPortalPaidRecommendations:
    """Tests for union_with_portal_paid_recommendations.

    Mirror of TestUnionWithPortalFreeRecommendations: the Portal's
    paidRecommendedModels endpoint is the source of truth for what's a
    blessed paid model *right now*. The in-repo curated list and
    docs-hosted manifest can lag — this helper guarantees newly-launched
    paid models surface in the picker for paid-tier users without a CLI
    release.
    """

    _PAID = {"prompt": "0.000003", "completion": "0.000015"}
    _FREE = {"prompt": "0", "completion": "0"}

    def _payload(self, paid_models: list[str]) -> dict:
        return {
            "paidRecommendedModels": [
                {"modelName": mid, "displayName": mid} for mid in paid_models
            ],
        }


    def test_preserves_relative_order_of_new_paid_models(self):
        """Multiple new paid models are appended in payload order, after curated."""
        curated = ["anthropic/claude-opus-4.6"]
        pricing = {"anthropic/claude-opus-4.6": self._PAID}
        with patch(
            "hermes_cli.models.fetch_nous_recommended_models",
            return_value=self._payload(["openai/gpt-5.4", "openai/gpt-5.5"]),
        ):
            ids, _ = union_with_portal_paid_recommendations(curated, pricing, "")
        assert ids == [
            "anthropic/claude-opus-4.6",
            "openai/gpt-5.4",
            "openai/gpt-5.5",
        ]


class TestCheckNousFreeTierCache:
    """Tests for the TTL cache on check_nous_free_tier()."""

    def setup_method(self):
        _models_mod._free_tier_cache = None

    def teardown_method(self):
        _models_mod._free_tier_cache = None

    @patch("hermes_cli.nous_account.get_nous_portal_account_info")
    def test_result_is_cached(self, mock_account):
        """Second call within TTL returns cached result without account lookup."""
        mock_account.return_value = NousPortalAccountInfo(
            logged_in=True,
            source="jwt",
            fresh=False,
            paid_service_access=False,
        )
        result1 = check_nous_free_tier()
        result2 = check_nous_free_tier()

        assert result1 is True
        assert result2 is True
        assert mock_account.call_count == 1


    @patch("hermes_cli.nous_account.get_nous_portal_account_info")
    def test_force_fresh_bypasses_cache(self, mock_account):
        mock_account.return_value = NousPortalAccountInfo(
            logged_in=True,
            source="account_api",
            fresh=True,
            paid_service_access=True,
        )

        assert check_nous_free_tier() is False
        assert check_nous_free_tier(force_fresh=True) is False

        assert mock_account.call_count == 2
        mock_account.assert_called_with(force_fresh=True)



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
        with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=mock_cm) as mock_urlopen:
            a = fetch_nous_recommended_models("https://portal.example.com")
            b = fetch_nous_recommended_models("https://portal.example.com")
        assert a == self._SAMPLE_PAYLOAD
        assert b == self._SAMPLE_PAYLOAD
        assert mock_urlopen.call_count == 1  # second call served from cache







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


class TestCodexSoftAcceptPlausibilityGate:
    """#45006 kernel (b): the openai-codex / xai-oauth hidden-model soft-accept
    (#16172 / #19729) must only accept slugs that plausibly belong to that
    provider's family. An undeclared, unrelated typed name (e.g. a local model
    name) must be REJECTED with actionable --provider guidance instead of being
    fake-accepted as a hidden Codex/Grok model (which would 400 on the next turn
    and mislabel the provider as 'OpenAI Codex')."""

    def test_unrelated_name_rejected_on_openai_codex(self):
        from hermes_cli.models import validate_requested_model
        r = validate_requested_model("qwen3.5-4b", "openai-codex")
        assert r["accepted"] is False
        assert r["persist"] is False
        assert "--provider" in (r["message"] or "")


    def test_real_catalog_model_unaffected(self):
        from hermes_cli.models import validate_requested_model
        r = validate_requested_model("gpt-5.5", "openai-codex")
        assert r["accepted"] is True
        assert r["recognized"] is True


class TestClaudeSonnet5InCuratedLists:
    """Regression: Claude Sonnet 5 must appear in curated model lists (#55846)."""

    def test_anthropic_native_list_includes_sonnet_5(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "claude-sonnet-5" in _PROVIDER_MODELS["anthropic"]




class TestFormatPricePerMtok:
    """_format_price_per_mtok: sub-cent prices must not collapse to 'free'/'$0.00'."""

    def test_standard_prices_keep_two_decimals(self):
        from hermes_cli.models import _format_price_per_mtok
        assert _format_price_per_mtok("0.000003") == "$3.00"
        assert _format_price_per_mtok("0.00003") == "$30.00"
        assert _format_price_per_mtok("0.00000015") == "$0.15"
        assert _format_price_per_mtok("0.00018") == "$180.00"

    def test_zero_is_free(self):
        from hermes_cli.models import _format_price_per_mtok
        assert _format_price_per_mtok("0") == "free"
        assert _format_price_per_mtok("0.0") == "free"

    def test_invalid_is_question_mark(self):
        from hermes_cli.models import _format_price_per_mtok
        assert _format_price_per_mtok("garbage") == "?"
        assert _format_price_per_mtok(None) == "?"

    def test_sub_cent_price_extends_precision(self):
        from hermes_cli.models import _format_price_per_mtok
        # DeepSeek V4 Flash 0731 promo cache-hit rate: $0.0018/Mtok.
        assert _format_price_per_mtok("0.0000000018") == "$0.0018"
        assert _format_price_per_mtok("0.000000001") == "$0.001"
        assert _format_price_per_mtok("0.0000000049") == "$0.0049"
        assert _format_price_per_mtok("0.000000005") == "$0.005"
        # Tiny but non-zero must never render as free or $0.00.
        assert _format_price_per_mtok("0.00000000001") == "$0.00001"

    def test_one_cent_boundary_stays_two_decimals(self):
        from hermes_cli.models import _format_price_per_mtok
        assert _format_price_per_mtok("0.00000001") == "$0.01"
