"""Tests for the hermes_cli models module."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
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



    def test_nous_list_includes_sonnet_5(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "anthropic/claude-sonnet-5" in _PROVIDER_MODELS["nous"]


class _FakeOllamaTagsHandler(BaseHTTPRequestHandler):
    """Serve Ollama-native /api/tags while rejecting OpenAI /v1/models."""

    models_payload = [
        {"name": "qwen3:1.7b", "model": "qwen3:1.7b"},
        {"name": "llama3.2:1b", "model": "llama3.2:1b"},
    ]
    paths_seen: list[str] = []

    def do_GET(self):
        type(self).paths_seen.append(self.path)
        if self.path.rstrip("/") == "/api/tags":
            body = json.dumps({"models": type(self).models_payload}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.rstrip("/") == "/v1/models":
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def _start_fake_ollama_server(models=None):
    _FakeOllamaTagsHandler.models_payload = (
        models
        if models is not None
        else [
            {"name": "qwen3:1.7b", "model": "qwen3:1.7b"},
            {"name": "llama3.2:1b", "model": "llama3.2:1b"},
        ]
    )
    _FakeOllamaTagsHandler.paths_seen = []
    server = HTTPServer(("127.0.0.1", 0), _FakeOllamaTagsHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


class TestLocalOllamaModelDiscovery:
    def test_provider_model_ids_uses_ollama_api_tags_from_provider_config(self):
        """Local Ollama discovery should use /api/tags from providers.ollama.base_url."""
        from hermes_cli.models import provider_model_ids

        server, port = _start_fake_ollama_server()
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": f"http://127.0.0.1:{port}"}}},
            ):
                assert provider_model_ids("ollama", force_refresh=True) == [
                    "qwen3:1.7b",
                    "llama3.2:1b",
                ]
        finally:
            server.shutdown()

    def test_provider_model_ids_ollama_force_refresh_clears_native_tags_cache(self):
        from hermes_cli.models import provider_model_ids

        server, port = _start_fake_ollama_server(models=[{"name": "old-model"}])
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": f"http://127.0.0.1:{port}"}}},
            ):
                assert provider_model_ids("ollama", force_refresh=True) == ["old-model"]
                _FakeOllamaTagsHandler.models_payload = [{"name": "new-model"}]
                assert provider_model_ids("ollama", force_refresh=True) == ["new-model"]
        finally:
            server.shutdown()

    def test_native_tags_cache_expires(self, monkeypatch):
        from hermes_cli.models import fetch_ollama_local_models

        server, port = _start_fake_ollama_server(models=[{"name": "old-model"}])
        try:
            base_url = f"http://127.0.0.1:{port}"
            assert fetch_ollama_local_models(base_url) == ["old-model"]
            _FakeOllamaTagsHandler.models_payload = [{"name": "new-model"}]
            root = _models_mod._root_for_ollama_native_api(base_url)
            cached_models, _ = _models_mod._OLLAMA_LOCAL_MODELS_CACHE[root]
            _models_mod._OLLAMA_LOCAL_MODELS_CACHE[root] = (cached_models, 0.0)
            monkeypatch.setattr("hermes_cli.models.time.monotonic", lambda: 301.0)
            assert fetch_ollama_local_models(base_url) == ["new-model"]
        finally:
            server.shutdown()

    def test_ollama_has_no_static_default_model(self):
        from hermes_cli.models import get_default_model_for_provider

        assert get_default_model_for_provider("ollama") == ""

    def test_fetch_ollama_models_accepts_base_url_without_scheme(self):
        """OLLAMA_HOST commonly omits http://; discovery should normalize it."""
        from hermes_cli.models import fetch_ollama_local_models

        server, port = _start_fake_ollama_server(models=[{"name": "qwen2.5:1.5b"}])
        try:
            assert fetch_ollama_local_models(f"127.0.0.1:{port}/v1") == ["qwen2.5:1.5b"]
        finally:
            server.shutdown()

    def test_fetch_ollama_models_accepts_full_models_url(self):
        """Pasted OpenAI-style /v1/models URLs should normalize to the native root."""
        from hermes_cli.models import fetch_ollama_local_models

        server, port = _start_fake_ollama_server(models=[{"name": "qwen2.5:1.5b"}])
        try:
            assert fetch_ollama_local_models(f"127.0.0.1:{port}/v1/models") == [
                "qwen2.5:1.5b"
            ]
        finally:
            server.shutdown()
        assert "/api/tags" in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models/api/tags" not in _FakeOllamaTagsHandler.paths_seen

    def test_runtime_error_from_config_load_does_not_escape_ollama_helpers(self):
        """Managed-mode config failures should degrade to defaults, not crash pickers."""
        from hermes_cli.models import _get_ollama_base_url, should_use_ollama_native_catalog

        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("bad home")), patch(
            "hermes_cli.models.probe_ollama_local_models",
            return_value=None,
        ):
            assert _get_ollama_base_url() == "http://localhost:11434"
            assert should_use_ollama_native_catalog("custom", "127.0.0.1:11434/v1") is False

    def test_probe_ollama_models_malformed_base_url_returns_none(self):
        """Malformed user-configured URLs should behave like probe failures, not crashes."""
        from hermes_cli.models import probe_ollama_local_models

        assert probe_ollama_local_models("http://127.0.0.1:bad-port/v1") is None

    def test_fetch_ollama_models_preserves_probe_failure(self):
        from hermes_cli.models import fetch_ollama_local_models

        with patch("hermes_cli.models.probe_ollama_local_models", return_value=None):
            assert fetch_ollama_local_models("http://127.0.0.1:11434") is None

    def test_ollama_port_detection_requires_working_api_tags(self):
        from hermes_cli.models import should_use_ollama_native_catalog

        with patch("hermes_cli.models.probe_ollama_local_models", return_value=["qwen3:1.7b"]):
            assert should_use_ollama_native_catalog("custom", "192.168.1.5:11434/v1") is True
        with patch("hermes_cli.models.probe_ollama_local_models", return_value=None):
            assert should_use_ollama_native_catalog("custom", "192.168.1.5:11434/v1") is False

    def test_provider_model_ids_ollama_cloud_config_uses_generic_catalog(self):
        from hermes_cli.models import provider_model_ids

        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "providers": {
                    "ollama": {
                        "base_url": "https://ollama.com/v1",
                        "api_key": "cloud-key",
                    }
                }
            },
        ), patch("hermes_cli.models.fetch_ollama_local_models") as fetch_local, patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["qwen3:1.7b"],
        ) as fetch_generic:
            assert provider_model_ids("ollama", force_refresh=True) == ["qwen3:1.7b"]
        fetch_local.assert_not_called()
        fetch_generic.assert_called_once_with(
            "cloud-key",
            "https://ollama.com/v1",
            headers={"Authorization": "Bearer cloud-key"},
        )

    def test_native_ollama_catalog_uses_configured_key_env(self, monkeypatch):
        from hermes_cli.models import _get_ollama_request_headers

        monkeypatch.setenv("TEST_OLLAMA_API_KEY", "env-key")
        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "providers": {
                    "ollama": {
                        "base_url": "https://ollama.internal/v1",
                        "key_env": "TEST_OLLAMA_API_KEY",
                    }
                }
            },
        ):
            assert _get_ollama_request_headers() == {
                "Authorization": "Bearer env-key"
            }

    def test_native_ollama_catalog_uses_api_key_env_alias(self, monkeypatch):
        from hermes_cli.models import _get_ollama_request_headers

        monkeypatch.setenv("TEST_OLLAMA_API_KEY_ALIAS", "alias-key")
        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "providers": {
                    "ollama": {
                        "base_url": "https://ollama.internal/v1",
                        "api_key_env": "TEST_OLLAMA_API_KEY_ALIAS",
                    }
                }
            },
        ):
            assert _get_ollama_request_headers() == {
                "Authorization": "Bearer alias-key"
            }

    def test_provider_model_ids_ignores_active_non_ollama_custom_endpoint(self):
        from hermes_cli.models import provider_model_ids

        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "custom",
                    "base_url": "https://custom.example/v1",
                }
            },
        ), patch(
            "hermes_cli.models.fetch_ollama_local_models",
            return_value=["qwen3:1.7b"],
        ) as fetch_local:
            assert provider_model_ids("ollama", force_refresh=True) == ["qwen3:1.7b"]
        fetch_local.assert_called_once_with("http://localhost:11434")

    def test_ollama_cache_fingerprint_does_not_probe_custom_endpoint(self):
        from hermes_cli.models import _credential_fingerprint

        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "custom",
                    "base_url": "http://127.0.0.1:11434/v1",
                }
            },
        ), patch("hermes_cli.models.probe_ollama_local_models") as probe_ollama:
            assert _credential_fingerprint("ollama")
        probe_ollama.assert_not_called()

    def test_ollama_cache_fingerprint_changes_when_configured_api_key_changes(self):
        from hermes_cli.models import _credential_fingerprint

        provider_config = {
            "base_url": "http://127.0.0.1:11434",
            "api_key": "ollama-key-a",
        }
        with patch(
            "hermes_cli.config.load_config",
            return_value={"providers": {"ollama": provider_config}},
        ):
            first = _credential_fingerprint("ollama")
            provider_config["api_key"] = "ollama-key-b"
            second = _credential_fingerprint("ollama")

        assert first != second

    def test_ollama_cache_fingerprint_changes_when_key_env_value_changes(self, monkeypatch):
        from hermes_cli.models import _credential_fingerprint

        monkeypatch.setenv("TEST_OLLAMA_API_KEY", "ollama-key-a")
        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "providers": {
                    "ollama": {
                        "base_url": "http://127.0.0.1:11434",
                        "key_env": "TEST_OLLAMA_API_KEY",
                    }
                }
            },
        ):
            first = _credential_fingerprint("ollama")
            monkeypatch.setenv("TEST_OLLAMA_API_KEY", "ollama-key-b")
            second = _credential_fingerprint("ollama")

        assert first != second

    def test_clear_provider_models_cache_clears_ollama_native_tags_cache(self):
        import hermes_cli.models as models

        cache = getattr(models, "_OLLAMA_LOCAL_MODELS_CACHE")
        cache["http://127.0.0.1:11434"] = ("old-model",)
        models.clear_provider_models_cache("ollama")
        assert cache == {}

    def test_clear_provider_models_cache_custom_clears_native_tags_cache(self):
        import hermes_cli.models as models

        cache = getattr(models, "_OLLAMA_LOCAL_MODELS_CACHE")
        cache["http://127.0.0.1:11434"] = ("old-model",)
        models.clear_provider_models_cache("custom")
        assert cache == {}

    def test_clear_provider_models_cache_does_not_remove_custom_disk_cache(self):
        import hermes_cli.models as models

        disk_cache = {
            "custom": {"models": ["custom-model"]},
            "ollama": {"models": ["ollama-model"]},
        }
        with patch.object(models, "_load_provider_models_cache", return_value=disk_cache), patch.object(
            models, "_save_provider_models_cache"
        ) as save:
            models.clear_provider_models_cache("ollama")
        save.assert_called_once_with({"custom": {"models": ["custom-model"]}})


    def test_ollama_cloud_urls_do_not_use_native_local_catalog(self):
        from hermes_cli.models import should_use_ollama_native_catalog

        assert should_use_ollama_native_catalog("ollama-cloud", "https://ollama.com/v1") is False
        assert should_use_ollama_native_catalog("ollama", "https://ollama.com/v1") is False

    def test_non_ollama_custom_endpoint_uses_generic_catalog_path(self):
        from hermes_cli.models import should_use_ollama_native_catalog

        assert should_use_ollama_native_catalog("custom", "https://example.test/v1") is False
        assert should_use_ollama_native_catalog("openrouter", "http://localhost:11434/v1") is False

    def test_picker_user_provider_row_discovers_ollama_api_tags(self):
        """providers.ollama with only base_url should still show local Ollama models."""
        from hermes_cli.model_switch import list_authenticated_providers

        server, port = _start_fake_ollama_server()
        try:
            rows = list_authenticated_providers(
                user_providers={"ollama": {"base_url": f"http://127.0.0.1:{port}"}},
                custom_providers=[],
                max_models=10,
            )
        finally:
            server.shutdown()

        ollama_row = next(row for row in rows if row["slug"] == "ollama")
        assert ollama_row["models"] == ["qwen3:1.7b", "llama3.2:1b"]
        assert ollama_row["total_models"] == 2

    def test_picker_non_ollama_user_provider_uses_configured_ollama_root(self):
        """A custom-named providers: entry at the configured Ollama root should use /api/tags."""
        from hermes_cli.model_switch import list_authenticated_providers

        server, port = _start_fake_ollama_server()
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": base_url}}},
            ):
                rows = list_authenticated_providers(
                    user_providers={"local-llm": {"base_url": base_url}},
                    custom_providers=[],
                    max_models=10,
                )
        finally:
            server.shutdown()

        row = next(row for row in rows if row["slug"] == "local-llm")
        assert row["models"] == ["qwen3:1.7b", "llama3.2:1b"]
        assert "/api/tags" in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_picker_user_provider_verifies_ambiguous_ollama_port(self):
        """A custom-named providers: entry on :11434 should use /api/tags after verification."""
        from hermes_cli.model_switch import list_authenticated_providers

        with patch("hermes_cli.config.load_config", return_value={"providers": {}}), patch(
            "hermes_cli.models.should_use_ollama_native_catalog",
            return_value=True,
        ), patch(
            "hermes_cli.models.fetch_ollama_local_models",
            return_value=["qwen3:1.7b"],
        ), patch("hermes_cli.models.fetch_api_models", return_value=[]) as fetch_api:
            rows = list_authenticated_providers(
                user_providers={"local-llm": {"base_url": "http://127.0.0.1:11434/v1"}},
                custom_providers=[],
                max_models=10,
            )

        row = next(row for row in rows if row["slug"] == "local-llm")
        assert row["models"] == ["qwen3:1.7b"]
        fetch_api.assert_not_called()

    def test_picker_bare_custom_model_config_discovers_ollama_api_tags(self):
        """The documented model.provider=custom shape should use native tags."""
        from hermes_cli.model_switch import list_authenticated_providers

        server, port = _start_fake_ollama_server()
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": base_url}}},
            ):
                rows = list_authenticated_providers(
                    current_provider="custom",
                    current_base_url=base_url,
                    current_model="qwen3:1.7b",
                    user_providers={},
                    custom_providers=[],
                    probe_custom_providers=False,
                    probe_current_custom_provider=True,
                    max_models=10,
                )
        finally:
            server.shutdown()

        row = next(row for row in rows if row["slug"] == "custom")
        assert row["models"] == ["qwen3:1.7b", "llama3.2:1b"]
        assert "/api/tags" in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_named_custom_model_flow_discovers_ollama_api_tags(self):
        """Interactive named-custom setup should use tags for a local Ollama root."""
        from hermes_cli.main import _model_flow_named_custom

        server, port = _start_fake_ollama_server()
        base_url = f"http://127.0.0.1:{port}/v1"
        config = {"providers": {"ollama": {"base_url": base_url}}}
        menu_items: list[str] = []

        def cancel_after_capturing_models(_title, items, **_kwargs):
            menu_items.extend(items)
            return -1

        try:
            with patch("hermes_cli.config.load_config", return_value=config), patch(
                "hermes_cli.config.save_config"
            ), patch("hermes_cli.auth._save_model_choice"), patch(
                "hermes_cli.auth.deactivate_provider"
            ), patch("hermes_cli.main._save_custom_provider"), patch(
                "hermes_cli.curses_ui.curses_radiolist",
                side_effect=cancel_after_capturing_models,
            ), patch("builtins.input", return_value="manual-fallback"), patch(
                "builtins.print"
            ):
                _model_flow_named_custom(
                    config,
                    {"name": "Local Ollama", "base_url": base_url},
                )
        finally:
            server.shutdown()

        assert menu_items == ["qwen3:1.7b", "llama3.2:1b", "Cancel"]
        assert "/api/tags" in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_named_custom_model_flow_preserves_explicit_ollama_models(self):
        """An explicit named-custom models list should skip live native tags."""
        from hermes_cli.main import _model_flow_named_custom

        server, port = _start_fake_ollama_server()
        base_url = f"http://127.0.0.1:{port}/v1"
        config = {"providers": {"ollama": {"base_url": base_url}}}
        menu_items: list[str] = []

        def cancel_after_capturing_models(_title, items, **_kwargs):
            menu_items.extend(items)
            return -1

        try:
            with patch("hermes_cli.config.load_config", return_value=config), patch(
                "hermes_cli.curses_ui.curses_radiolist",
                side_effect=cancel_after_capturing_models,
            ), patch("builtins.print"):
                _model_flow_named_custom(
                    config,
                    {
                        "name": "Local Ollama",
                        "base_url": base_url,
                        "models": ["curated-only"],
                    },
                )
        finally:
            server.shutdown()

        assert menu_items == ["curated-only", "Cancel"]
        assert "/api/tags" not in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_picker_user_provider_preserves_explicit_models_for_ollama_root(self):
        """providers: entries should not replace an explicit model list with /api/tags."""
        from hermes_cli.model_switch import list_authenticated_providers

        server, port = _start_fake_ollama_server()
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": base_url}}},
            ):
                rows = list_authenticated_providers(
                    user_providers={
                        "local-llm": {
                            "base_url": base_url,
                            "api_key": "no-key-required",
                            "models": ["curated-only"],
                        }
                    },
                    custom_providers=[],
                    max_models=10,
                )
        finally:
            server.shutdown()

        row = next(row for row in rows if row["slug"] == "local-llm")
        assert row["models"] == ["curated-only"]
        assert "/api/tags" not in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_picker_custom_provider_group_discovers_configured_ollama_root(self):
        """custom_providers entries with no explicit model list should use /api/tags for Ollama roots."""
        from hermes_cli.model_switch import list_authenticated_providers

        server, port = _start_fake_ollama_server()
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": base_url}}},
            ):
                rows = list_authenticated_providers(
                    user_providers={},
                    custom_providers=[{"name": "Local Ollama", "base_url": base_url}],
                    max_models=10,
                )
        finally:
            server.shutdown()

        row = next(row for row in rows if row["name"] == "Local Ollama")
        assert row["models"] == ["qwen3:1.7b", "llama3.2:1b"]
        assert "/api/tags" in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_picker_custom_provider_saved_model_still_discovers_ollama_tags(self):
        """Singular model: is an active choice, not a native-catalog restriction."""
        from hermes_cli.model_switch import list_authenticated_providers

        server, port = _start_fake_ollama_server()
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": base_url}}},
            ):
                rows = list_authenticated_providers(
                    user_providers={},
                    custom_providers=[
                        {
                            "name": "Local Ollama",
                            "base_url": base_url,
                            "model": "qwen3:1.7b",
                        }
                    ],
                    max_models=10,
                )
        finally:
            server.shutdown()

        row = next(row for row in rows if row["name"] == "Local Ollama")
        assert row["models"] == ["qwen3:1.7b", "llama3.2:1b"]
        assert "/api/tags" in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_picker_custom_provider_group_verifies_ambiguous_ollama_port(self):
        """Generated custom provider slugs should not block verified :11434 /api/tags discovery."""
        from hermes_cli.model_switch import list_authenticated_providers

        with patch("hermes_cli.config.load_config", return_value={"providers": {}}), patch(
            "hermes_cli.models.probe_ollama_local_models",
            return_value=["qwen3:1.7b"],
        ), patch("hermes_cli.models.fetch_api_models", return_value=[]) as fetch_api:
            rows = list_authenticated_providers(
                user_providers={},
                custom_providers=[{"name": "Local Ollama", "base_url": "http://127.0.0.1:11434/v1"}],
                max_models=10,
            )

        row = next(row for row in rows if row["name"] == "Local Ollama")
        assert row["models"] == ["qwen3:1.7b"]
        fetch_api.assert_not_called()

    def test_model_validation_uses_ollama_api_tags_for_ollama_provider(self):
        """`/model` validation for provider=ollama should not probe `/models`."""
        from hermes_cli.models import validate_requested_model

        server, port = _start_fake_ollama_server()
        try:
            result = validate_requested_model(
                "qwen3:1.7b",
                "ollama",
                base_url=f"http://127.0.0.1:{port}",
            )
        finally:
            server.shutdown()

        assert result == {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": None,
        }
        assert "/api/tags" in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_model_validation_ollama_cloud_config_does_not_use_local_tags(self):
        """provider=ollama with a cloud base URL should not fall into local /api/tags."""
        from hermes_cli.models import validate_requested_model

        with patch(
            "hermes_cli.config.load_config",
            return_value={"providers": {"ollama": {"base_url": "https://ollama.com/v1"}}},
        ), patch("hermes_cli.models.probe_ollama_local_models") as probe_ollama, patch(
            "hermes_cli.models.probe_api_models",
            return_value={
                "models": ["qwen3:1.7b"],
                "probed_url": "https://ollama.com/v1/models",
            },
        ):
            result = validate_requested_model("qwen3:1.7b", "ollama")

        probe_ollama.assert_not_called()
        assert result == {
            "accepted": True,
            "persist": True,
            "recognized": True,
            "message": None,
        }

    def test_model_validation_uses_ollama_api_tags_for_matching_custom_endpoint(self):
        """Current-provider `custom` on the configured Ollama URL should use `/api/tags`."""
        from hermes_cli.models import validate_requested_model

        server, port = _start_fake_ollama_server()
        base_url = f"http://127.0.0.1:{port}"
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": base_url}}},
            ):
                result = validate_requested_model(
                    "llama3.2:1b",
                    "custom",
                    base_url=base_url,
                )
        finally:
            server.shutdown()

        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None
        assert "/api/tags" in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_model_validation_empty_ollama_tags_does_not_fall_back_to_models(self):
        """Reachable but empty /api/tags should not produce a misleading /models warning."""
        from hermes_cli.models import validate_requested_model

        server, port = _start_fake_ollama_server(models=[])
        base_url = f"http://127.0.0.1:{port}"
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": base_url}}},
            ):
                result = validate_requested_model(
                    "qwen3:1.7b",
                    "custom",
                    base_url=base_url,
                )
        finally:
            server.shutdown()

        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is False
        assert "/api/tags" in result["message"]
        assert "/models" not in result["message"]
        assert "/api/tags" in _FakeOllamaTagsHandler.paths_seen
        assert "/v1/models" not in _FakeOllamaTagsHandler.paths_seen

    def test_switch_model_on_current_ollama_custom_endpoint_keeps_base_url(self):
        """Mid-session `/model` on local Ollama must not re-resolve custom to another provider."""
        from hermes_cli.model_switch import switch_model

        server, port = _start_fake_ollama_server()
        base_url = f"http://127.0.0.1:{port}"
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"providers": {"ollama": {"base_url": base_url}}},
            ), patch(
                "hermes_cli.model_switch.get_model_info",
                return_value=None,
            ):
                result = switch_model(
                    raw_input="llama3.2:1b",
                    current_provider="custom",
                    current_model="qwen3:1.7b",
                    current_base_url=base_url,
                    current_api_key="no-key-required",
                    user_providers={"ollama": {"base_url": base_url}},
                    custom_providers=[],
                )
        finally:
            server.shutdown()

        assert result.success is True
        assert result.target_provider == "custom"
        assert result.new_model == "llama3.2:1b"
        assert result.base_url == base_url
        assert result.warning_message == ""

    def test_switch_model_on_non_ollama_custom_endpoint_still_resolves_runtime(self):
        """The Ollama base-url preservation path must not change ordinary custom endpoints."""
        from hermes_cli.model_switch import switch_model

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "new-key",
                "base_url": "https://custom.example/v1",
                "api_mode": "chat_completions",
            },
        ), patch(
            "hermes_cli.models.validate_requested_model",
            return_value={
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            },
        ), patch(
            "hermes_cli.model_switch.get_model_info",
            return_value=None,
        ):
            result = switch_model(
                raw_input="my-model",
                current_provider="custom",
                current_model="old-model",
                current_base_url="https://old-custom.example/v1",
                current_api_key="old-key",
                user_providers={},
                custom_providers=[],
            )

        assert result.success is True
        assert result.base_url == "https://custom.example/v1"
        assert result.api_key == "new-key"

    def test_switch_model_ollama_precheck_runtime_error_falls_back_to_runtime_resolution(self):
        """A native-catalog precheck failure should not abort ordinary /model switching."""
        from hermes_cli.model_switch import switch_model

        with patch(
            "hermes_cli.models.should_use_ollama_native_catalog",
            side_effect=RuntimeError("config unavailable"),
        ), patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "new-key",
                "base_url": "https://custom.example/v1",
                "api_mode": "chat_completions",
            },
        ), patch(
            "hermes_cli.models.validate_requested_model",
            return_value={
                "accepted": True,
                "persist": True,
                "recognized": True,
                "message": None,
            },
        ), patch(
            "hermes_cli.model_switch.get_model_info",
            return_value=None,
        ):
            result = switch_model(
                raw_input="my-model",
                current_provider="custom",
                current_model="old-model",
                current_base_url="http://127.0.0.1:11434/v1",
                current_api_key="old-key",
                user_providers={},
                custom_providers=[],
            )

        assert result.success is True
        assert result.base_url == "https://custom.example/v1"
        assert result.api_key == "new-key"

    def test_ollama_root_matching_is_case_insensitive_for_hostnames(self):
        from hermes_cli.models import _same_ollama_native_root

        assert _same_ollama_native_root(
            "HTTP://OLLAMA.EXAMPLE:11434/v1",
            "http://ollama.example:11434",
        ) is True

    def test_ollama_host_environment_forms_are_normalized(self, monkeypatch):
        from hermes_cli.models import _get_ollama_base_url, _root_for_ollama_native_api

        monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
        assert _root_for_ollama_native_api(_get_ollama_base_url()) == "http://0.0.0.0:11434"
        monkeypatch.setenv("OLLAMA_HOST", ":22434")
        assert _root_for_ollama_native_api(_get_ollama_base_url()) == "http://127.0.0.1:22434"
        monkeypatch.setenv("OLLAMA_HOST", "::1")
        assert _root_for_ollama_native_api(_get_ollama_base_url()) == "http://[::1]:11434"
        monkeypatch.setenv("OLLAMA_HOST", "[::1]")
        assert _root_for_ollama_native_api(_get_ollama_base_url()) == "http://[::1]:11434"
        monkeypatch.setenv("OLLAMA_HOST", "http://ollama.example")
        assert _get_ollama_base_url() == "http://ollama.example:11434"
        monkeypatch.setenv("OLLAMA_HOST", "http://user:pass@ollama.example")
        assert _get_ollama_base_url() == "http://user:pass@ollama.example:11434"
        assert _root_for_ollama_native_api("http://ollama.example/api/tags") == "http://ollama.example"

    def test_ollama_failed_probe_is_cached_briefly(self):
        import hermes_cli.models as models

        models._OLLAMA_LOCAL_MODELS_CACHE.clear()
        models._OLLAMA_LOCAL_PROBE_FAILURE_CACHE.clear()
        with patch(
            "hermes_cli.models._urlopen_model_catalog_request",
            side_effect=OSError("offline"),
        ) as request:
            assert models.probe_ollama_local_models("http://127.0.0.1:19999") is None
            assert models.probe_ollama_local_models("http://127.0.0.1:19999") is None
        request.assert_called_once()

    def test_empty_ollama_catalog_does_not_resurrect_stale_disk_models(self):
        import hermes_cli.models as models

        base_url = "http://127.0.0.1:11434"
        probe_key = models._ollama_probe_cache_key(base_url, None)
        models._OLLAMA_LOCAL_PROBE_REACHABLE[probe_key] = True
        try:
            with patch.object(
                models,
                "_load_provider_models_cache",
                return_value={"ollama": {"fp": "same", "at": 0, "models": ["stale:model"]}},
            ), patch.object(models, "_save_provider_models_cache"), patch.object(
                models, "_credential_fingerprint", return_value="same"
            ), patch.object(models, "provider_model_ids", return_value=[]), patch.object(
                models, "_get_ollama_base_url", return_value=base_url
            ), patch.object(models, "_get_ollama_request_headers", return_value={}):
                assert models.cached_provider_model_ids("ollama") == []
        finally:
            models._OLLAMA_LOCAL_PROBE_REACHABLE.pop(probe_key, None)

    def test_failed_ollama_catalog_preserves_stale_disk_models(self):
        import hermes_cli.models as models

        base_url = "http://127.0.0.1:11434"
        probe_key = models._ollama_probe_cache_key(base_url, None)
        models._OLLAMA_LOCAL_PROBE_REACHABLE[probe_key] = False
        try:
            with patch.object(
                models,
                "_load_provider_models_cache",
                return_value={"ollama": {"fp": "same", "at": 0, "models": ["stale:model"]}},
            ), patch.object(models, "_save_provider_models_cache"), patch.object(
                models, "_credential_fingerprint", return_value="same"
            ), patch.object(models, "provider_model_ids", return_value=[]), patch.object(
                models, "_get_ollama_base_url", return_value=base_url
            ), patch.object(models, "_get_ollama_request_headers", return_value={}):
                assert models.cached_provider_model_ids("ollama") == ["stale:model"]
        finally:
            models._OLLAMA_LOCAL_PROBE_REACHABLE.pop(probe_key, None)

    def test_ollama_native_request_uses_redirect_safe_catalog_helper(self):
        import hermes_cli.models as models

        response = MagicMock()
        response.read.return_value = b'{"models": [{"name": "qwen3:1.7b"}]}'
        response.__enter__.return_value = response
        with patch.object(
            models, "_urlopen_model_catalog_request", return_value=response
        ) as request:
            assert models.fetch_ollama_local_models("http://127.0.0.1:11434") == [
                "qwen3:1.7b"
            ]
        request.assert_called_once()

    def test_validation_with_nonmatching_ollama_root_does_not_forward_config_headers(self):
        import hermes_cli.models as models

        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "providers": {
                    "ollama": {
                        "base_url": "https://ollama.internal/v1",
                        "extra_headers": {"Authorization": "Bearer secret"},
                    }
                }
            },
        ), patch.object(models, "should_use_ollama_native_catalog", return_value=True), patch.object(
            models, "probe_ollama_local_models", return_value=[]
        ) as probe:
            models.validate_requested_model(
                "qwen3:1.7b",
                "ollama",
                base_url="https://other.internal/v1",
            )
            assert probe.call_args.kwargs["headers"] == {}
            models.validate_requested_model(
                "qwen3:1.7b",
                "ollama",
                base_url="https://other.internal/v1",
                headers={"X-Endpoint-Token": "explicit"},
            )
            assert probe.call_args.kwargs["headers"] == {"X-Endpoint-Token": "explicit"}



    def test_switch_model_direct_ollama_alias_preserves_matching_origin_api_key(self):
        import hermes_cli.model_switch as model_switch

        base_url = "https://ollama.internal/v1"
        original_aliases = dict(model_switch.DIRECT_ALIASES)
        model_switch.DIRECT_ALIASES.clear()
        model_switch.DIRECT_ALIASES["remote-qwen"] = model_switch.DirectAlias(
            model="qwen3:1.7b", provider="ollama", base_url=base_url
        )
        try:
            with patch.object(model_switch, "get_model_info", return_value=None), patch(
                "hermes_cli.models._get_provider_config_dict",
                return_value={"base_url": base_url, "api_key": "secret"},
            ) as config_provider, patch(
                "hermes_cli.models.validate_requested_model",
                return_value={"accepted": True, "persist": True, "recognized": True, "message": ""},
            ):
                result = model_switch.switch_model(
                    raw_input="remote-qwen",
                    current_provider="openrouter",
                    current_model="old-model",
                    current_api_key="stale-other-endpoint-key",
                    user_providers={"ollama": {"base_url": base_url, "api_key": "secret"}},
                    custom_providers=[],
                )
        finally:
            model_switch.DIRECT_ALIASES.clear()
            model_switch.DIRECT_ALIASES.update(original_aliases)

        assert config_provider.call_args is not None, result
        assert config_provider.call_args.args == ("ollama",), config_provider.call_args
        assert result.success is True
        assert result.api_key == "secret", result

    def test_switch_model_direct_ollama_alias_clears_different_origin_api_key(self):
        import hermes_cli.model_switch as model_switch

        original_aliases = dict(model_switch.DIRECT_ALIASES)
        model_switch.DIRECT_ALIASES.clear()
        model_switch.DIRECT_ALIASES["other-qwen"] = model_switch.DirectAlias(
            model="qwen3:1.7b", provider="ollama", base_url="https://other.internal/v1"
        )
        try:
            with patch.object(model_switch, "get_model_info", return_value=None), patch(
                "hermes_cli.models.validate_requested_model",
                return_value={"accepted": True, "persist": True, "recognized": True, "message": ""},
            ):
                result = model_switch.switch_model(
                    raw_input="other-qwen",
                    current_provider="custom",
                    current_model="old-model",
                    user_providers={
                        "ollama": {
                            "base_url": "https://ollama.internal/v1",
                            "api_key": "secret",
                        }
                    },
                    custom_providers=[],
                )
        finally:
            model_switch.DIRECT_ALIASES.clear()
            model_switch.DIRECT_ALIASES.update(original_aliases)

        assert result.success is True
        assert result.api_key == "no-key-required"
