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
    # The bare origin, whichever form the override was written in: callers
    # append their own path (``/v1/models``), so a suffix here would double up.
    assert base_url == "https://stg-inference-api.nousresearch.com"


def test_resolve_nous_pricing_credentials_normalizes_either_suffix(monkeypatch):
    """``/v1`` on the override is optional and must not change the result."""
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials", lambda: None
    )
    for override in (
        "https://stg-inference-api.nousresearch.com",
        "https://stg-inference-api.nousresearch.com/",
        "https://stg-inference-api.nousresearch.com/v1",
        "https://stg-inference-api.nousresearch.com/v1/",
    ):
        monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", override)
        assert models_mod._resolve_nous_pricing_credentials()[1] == (
            "https://stg-inference-api.nousresearch.com"
        )


def test_a_failed_catalog_fetch_is_not_cached_forever(monkeypatch):
    """A blip must not disable live model discovery for the whole process.

    The empty result is cached so a dead endpoint isn't re-dialed on every
    call, but it expires — the processes that read this run for weeks, and
    every caller silently falls back to a curated list meanwhile.
    """
    models_mod._pricing_cache.clear()
    models_mod._pricing_cache_retry_after.clear()

    calls = []

    def _fail(req, timeout=8.0):
        calls.append(req)
        raise OSError("connection refused")

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", _fail)

    assert fetch_models_with_pricing(base_url="https://example.test") == {}
    # Inside the window the failure is cached: no second dial.
    assert fetch_models_with_pricing(base_url="https://example.test") == {}
    assert len(calls) == 1

    now = models_mod.time.monotonic()
    monkeypatch.setattr(
        models_mod.time,
        "monotonic",
        lambda: now + models_mod._FAILED_CATALOG_TTL_SECONDS + 1,
    )
    assert fetch_models_with_pricing(base_url="https://example.test") == {}
    assert len(calls) == 2


def test_a_successful_catalog_fetch_stays_cached(monkeypatch):
    """Only the failures expire; a real catalog is still fetched once."""
    models_mod._pricing_cache.clear()
    models_mod._pricing_cache_retry_after.clear()

    calls = []
    body = json.dumps(
        {"data": [{"id": "a/b", "pricing": {"prompt": "1", "completion": "2"}}]}
    ).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda *a: False

    def _ok(req, timeout=8.0):
        calls.append(req)
        return resp

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", _ok)

    assert "a/b" in fetch_models_with_pricing(base_url="https://example.test")
    now = models_mod.time.monotonic()
    monkeypatch.setattr(
        models_mod.time,
        "monotonic",
        lambda: now + models_mod._FAILED_CATALOG_TTL_SECONDS + 1,
    )
    assert "a/b" in fetch_models_with_pricing(base_url="https://example.test")
    assert len(calls) == 1

