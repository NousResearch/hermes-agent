"""AI Gateway model list and pricing translation.

Vercel AI Gateway exposes ``/v1/models`` with a richer shape than OpenAI's
spec (type, tags, pricing). The pricing object uses ``input`` / ``output``
where hermes's shared picker expects ``prompt`` / ``completion``; these tests
pin the translation and the curated-list filtering.
"""
import json
from unittest.mock import patch, MagicMock

from hermes_cli import models as models_module
from hermes_cli.models import (
    AI_GATEWAY_MODELS,
    _ai_gateway_model_is_free,
    fetch_ai_gateway_models,
    fetch_ai_gateway_pricing,
)


def _mock_urlopen(payload):
    """Build a urlopen() context manager mock returning the given payload."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


def _reset_caches():
    models_module._ai_gateway_catalog_cache = None
    models_module._pricing_cache.clear()


def test_ai_gateway_pricing_translates_input_output_to_prompt_completion():
    _reset_caches()
    payload = {
        "data": [
            {
                "id": "moonshotai/kimi-k2.5",
                "type": "language",
                "pricing": {
                    "input": "0.0000006",
                    "output": "0.0000025",
                    "input_cache_read": "0.00000015",
                    "input_cache_write": "0.0000006",
                },
            }
        ]
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_pricing(force_refresh=True)

    entry = result["moonshotai/kimi-k2.5"]
    assert entry["prompt"] == "0.0000006"
    assert entry["completion"] == "0.0000025"
    assert entry["input_cache_read"] == "0.00000015"
    assert entry["input_cache_write"] == "0.0000006"


def test_ai_gateway_pricing_returns_empty_on_fetch_failure():
    _reset_caches()
    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        result = fetch_ai_gateway_pricing(force_refresh=True)
    assert result == {}


def test_ai_gateway_pricing_skips_entries_without_pricing_dict():
    _reset_caches()
    payload = {
        "data": [
            {"id": "x/y", "pricing": None},
            {"id": "a/b", "pricing": {"input": "0", "output": "0"}},
        ]
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_pricing(force_refresh=True)
    assert "x/y" not in result
    assert result["a/b"] == {"prompt": "0", "completion": "0"}


def test_ai_gateway_free_detector():
    assert _ai_gateway_model_is_free({"input": "0", "output": "0"}) is True
    assert _ai_gateway_model_is_free({"input": "0", "output": "0.01"}) is False
    assert _ai_gateway_model_is_free({"input": "0.01", "output": "0"}) is False
    assert _ai_gateway_model_is_free(None) is False
    assert _ai_gateway_model_is_free({"input": "not a number"}) is False


def test_fetch_ai_gateway_models_filters_against_live_catalog():
    _reset_caches()
    preferred = [mid for mid, _ in AI_GATEWAY_MODELS]
    live_ids = preferred[:3]  # only first three exist live
    payload = {
        "data": [
            {"id": mid, "pricing": {"input": "0.001", "output": "0.002"}}
            for mid in live_ids
        ]
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_models(force_refresh=True)

    assert [mid for mid, _ in result] == live_ids
    assert result[0][1] == "recommended"


def test_fetch_ai_gateway_models_tags_free_models():
    _reset_caches()
    first_id = AI_GATEWAY_MODELS[0][0]
    second_id = AI_GATEWAY_MODELS[1][0]
    payload = {
        "data": [
            {"id": first_id, "pricing": {"input": "0.001", "output": "0.002"}},
            {"id": second_id, "pricing": {"input": "0", "output": "0"}},
        ]
    }
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_models(force_refresh=True)

    by_id = dict(result)
    assert by_id[first_id] == "recommended"
    assert by_id[second_id] == "free"


def test_fetch_ai_gateway_models_falls_back_on_error():
    _reset_caches()
    with patch("urllib.request.urlopen", side_effect=OSError("network")):
        result = fetch_ai_gateway_models(force_refresh=True)
    assert result == list(AI_GATEWAY_MODELS)
