"""Sale UI pricing helpers: gateway pricing.original → discount chrome."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import hermes_cli.models as models_mod
from hermes_cli.models import (
    compute_sale_discount,
    fetch_models_with_pricing,
)






def test_fetch_models_with_pricing_copies_nested_original(monkeypatch):
    models_mod._pricing_cache.clear()
    payload = {
        "data": [
            {
                "id": "anthropic/claude-sonnet-5",
                "pricing": {
                    "prompt": "0.0000016",
                    "completion": "0.000008",
                    "input_cache_read": "0.00000016",
                    "original": {
                        "prompt": "0.000002",
                        "completion": "0.00001",
                        "input_cache_read": "0.0000002",
                    },
                },
            },
            {
                "id": "free/model",
                "pricing": {"prompt": "0", "completion": "0"},
            },
        ]
    }
    body = json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda *a: False

    monkeypatch.setattr(
        models_mod,
        "_urlopen_model_catalog_request",
        lambda req, timeout=8.0: resp,
    )

    # Nous Portal opts in via include_sale_original=True.
    result = fetch_models_with_pricing(
        api_key="sk-test",
        base_url="https://example.test",
        force_refresh=True,
        include_sale_original=True,
    )
    paid = result["anthropic/claude-sonnet-5"]
    assert paid["prompt"] == "0.0000016"
    assert paid["completion"] == "0.000008"
    assert paid["original"] == {
        "prompt": "0.000002",
        "completion": "0.00001",
        "input_cache_read": "0.0000002",
    }
    assert "original" not in result["free/model"]




def test_resolve_nous_pricing_credentials_honors_inference_env_override(monkeypatch):
    """Staging profiles set NOUS_INFERENCE_BASE_URL — pricing must follow it.

    Without this, anonymous/failed-auth fallback hits prod and sale
    ``pricing.original`` never reaches Desktop/CLI pickers.
    """
    monkeypatch.setenv(
        "NOUS_INFERENCE_BASE_URL",
        "https://stg-inference-api.nousresearch.com/v1",
    )
    # Auth resolution fails / returns nothing — the env override must still win.
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        lambda: None,
    )
    api_key, base_url = models_mod._resolve_nous_pricing_credentials()
    assert api_key == ""
    assert base_url == "https://stg-inference-api.nousresearch.com/v1"


