"""Regression tests for null and malformed OpenAI-style model catalogs."""

import json
import os
from unittest.mock import MagicMock, patch

from hermes_cli import models
from hermes_cli import models_pricing
from hermes_cli import runtime_provider


class _JsonResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _fetch_opencode_free(payload):
    models._opencode_free_live_memo = None
    try:
        with patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            return_value=_JsonResponse(payload),
        ):
            return models._fetch_opencode_free_models(force_refresh=True)
    finally:
        models._opencode_free_live_memo = None


def _fetch_pricing(payload):
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()
    with (
        patch.object(models_pricing, "_get_json", return_value=payload),
        patch.object(models_pricing, "_seed_reasoning_caps"),
    ):
        return models_pricing.fetch_models_with_pricing(
            base_url="https://example.test",
            force_refresh=True,
        )


def _auto_detect(payload):
    response = MagicMock(ok=True)
    response.json.return_value = payload
    with patch("requests.get", return_value=response):
        return runtime_provider._auto_detect_local_model("http://localhost:1234")


# _fetch_live_catalog_index: 4 cases


def test_live_catalog_index_accepts_null_data():
    with patch.object(models, "_get_json", return_value={"data": None}):
        assert models._fetch_live_catalog_index(
            "https://example.test/models", 1, None
        ) == ([], {})


def test_live_catalog_index_accepts_missing_data():
    with patch.object(models, "_get_json", return_value={}):
        assert models._fetch_live_catalog_index(
            "https://example.test/models", 1, None
        ) == ([], {})


def test_live_catalog_index_accepts_non_dict_payload():
    with patch.object(models, "_get_json", return_value=[]):
        assert models._fetch_live_catalog_index(
            "https://example.test/models", 1, None
        ) == ([], {})


def test_live_catalog_index_ignores_non_dict_items():
    valid = {"id": "model-a"}
    payload = {"data": [None, "bad", valid]}
    with patch.object(models, "_get_json", return_value=payload):
        result = models._fetch_live_catalog_index(
            "https://example.test/models", 1, None
        )
    assert result is not None
    items, by_id = result
    assert items == payload["data"]
    assert by_id == {"model-a": valid}


# _fetch_anthropic_models: 4 cases


def test_anthropic_models_accepts_null_data():
    with patch.object(models, "_get_json", return_value={"data": None}):
        assert models._fetch_anthropic_models(api_key="test") == []


def test_anthropic_models_accepts_missing_data():
    with patch.object(models, "_get_json", return_value={}):
        assert models._fetch_anthropic_models(api_key="test") == []


def test_anthropic_models_accepts_non_dict_payload():
    with patch.object(models, "_get_json", return_value=[]):
        assert models._fetch_anthropic_models(api_key="test") == []


def test_anthropic_models_ignores_non_dict_items():
    payload = {"data": [None, "bad", {"id": "claude-haiku-test"}]}
    with patch.object(models, "_get_json", return_value=payload):
        assert models._fetch_anthropic_models(api_key="test") == ["claude-haiku-test"]


# _fetch_opencode_free_models: 3 cases


def test_opencode_free_models_accepts_null_data():
    assert _fetch_opencode_free({"data": None}) is None


def test_opencode_free_models_ignores_non_dict_list_items():
    assert _fetch_opencode_free([None, "bad", 7, {"id": "alpha-free"}]) == [
        "alpha-free"
    ]


def test_opencode_free_models_accepts_list_payload_with_free_items():
    payload = [
        {"id": "alpha-free"},
        {"id": "paid-model"},
        {"id": "beta-free"},
    ]
    assert _fetch_opencode_free(payload) == ["alpha-free", "beta-free"]


# _fetch_ai_gateway_models: 4 cases


def _fetch_ai_gateway(payload):
    env = {
        "AI_GATEWAY_API_KEY": "test",
        "AI_GATEWAY_BASE_URL": "https://example.test/v1",
    }
    with (
        patch.dict(os.environ, env, clear=False),
        patch.object(models, "_get_json", return_value=payload),
    ):
        return models._fetch_ai_gateway_models()


def test_ai_gateway_models_accepts_null_data():
    assert _fetch_ai_gateway({"data": None}) == []


def test_ai_gateway_models_accepts_missing_data():
    assert _fetch_ai_gateway({}) == []


def test_ai_gateway_models_accepts_non_dict_payload():
    assert _fetch_ai_gateway([]) == []


def test_ai_gateway_models_ignores_non_dict_items():
    payload = {
        "data": [
            None,
            "bad",
            {"id": "model-a", "type": "language", "tags": ["tool-use"]},
        ]
    }
    assert _fetch_ai_gateway(payload) == ["model-a"]


# fetch_models_with_pricing: 4 cases


def test_models_with_pricing_accepts_null_data():
    assert _fetch_pricing({"data": None}) == {}


def test_models_with_pricing_accepts_missing_data():
    assert _fetch_pricing({}) == {}


def test_models_with_pricing_accepts_non_dict_payload():
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()
    with (
        patch.object(models_pricing, "_get_json", return_value=[]),
        patch.object(models_pricing, "_seed_reasoning_caps") as seed,
    ):
        result = models_pricing.fetch_models_with_pricing(
            base_url="https://example.test",
            force_refresh=True,
        )

    assert result == {}
    seed.assert_called_once_with("https://example.test/v1/models", None)


def test_models_with_pricing_ignores_non_dict_items():
    payload = {
        "data": [
            None,
            "bad",
            {"id": "model-a", "pricing": {"prompt": "1", "completion": "2"}},
        ]
    }
    assert _fetch_pricing(payload) == {"model-a": {"prompt": "1", "completion": "2"}}


# _auto_detect_local_model: 5 cases


def test_auto_detect_local_model_accepts_null_data():
    assert _auto_detect({"data": None}) == ""


def test_auto_detect_local_model_accepts_non_dict_payload():
    assert _auto_detect([]) == ""


def test_auto_detect_local_model_accepts_empty_data():
    assert _auto_detect({"data": []}) == ""


def test_auto_detect_local_model_ignores_non_dict_item():
    assert _auto_detect({"data": ["bad"]}) == ""


def test_auto_detect_local_model_returns_valid_single_model():
    assert _auto_detect({"data": [{"id": "local-model"}]}) == "local-model"
