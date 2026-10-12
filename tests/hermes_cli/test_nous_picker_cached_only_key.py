"""The GUI/in-chat picker's on-sale union reads what the prewarm actually cached.

The Nous fetcher registers its provider cache key WITHOUT the credential fingerprint
while fetch_models_with_pricing WRITES under base + fingerprint — so for any logged-in
account (api key present) the picker's cached_only read missed the rows the prewarm
filled and the on-sale union saw an empty dict (review on #132015). This exercises
the real path: serve /v1/models once through the real fetcher with a key present, then
ask the picker for ids WITHOUT stubbing the pricing layer.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import hermes_cli.models as models_mod
import hermes_cli.models_pricing as mp
from hermes_cli import model_switch_providers as msp

_BASE = "https://inference.example.test/v1"
_LIST = {"prompt": "0.000002", "completion": "0.00001"}


def _serve_nous_catalog(monkeypatch, payload: dict) -> None:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda *a: False
    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", lambda req, timeout=8.0: resp)


def test_picker_on_sale_union_reads_the_authenticated_prewarm(monkeypatch):
    """Red on the unfixed key registration: cached_only must find the fingerprinted entry."""
    mp._pricing_cache.clear()
    mp._pricing_cache_retry_after.clear()
    mp._pricing_provider_cache_keys.clear()

    _serve_nous_catalog(monkeypatch, {"data": [
        {"id": "nous/curated", "pricing": dict(_LIST)},
        {"id": "sale/deep", "supported_parameters": ["tools"],
         "pricing": {"prompt": "0.0000004", "completion": "0.000002",
                     "original": dict(_LIST)}},  # 80% off, absent from the curated list
    ]})

    # The prewarm path: a logged-in account resolves an api key, so the catalog is written
    # under base + auth fingerprint. Anonymous/base-only stubs hide the mismatch.
    monkeypatch.setattr(mp, "_resolve_nous_pricing_credentials", lambda: ("sk-logged-in", _BASE))
    warmed = mp._fetch_nous_pricing_for_provider()
    assert "sale/deep" in warmed

    # The picker path: NO stub on get_pricing_for_provider/_cached_only_pricing — the real
    # cached_only read must resolve the same entry the prewarm filled.
    monkeypatch.setattr("hermes_cli.models.check_nous_free_tier", lambda **kw: False)
    monkeypatch.setattr("hermes_cli.models.fetch_nous_recommended_models", lambda *a, **kw: None)
    monkeypatch.setattr(mp, "nous_policy_allowed_ids", lambda **kw: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.get_pricing_for_provider",
        mp.get_pricing_for_provider,
        raising=False,
    )

    ids = msp._nous_picker_model_ids({"nous": ["nous/curated"]}, False)
    assert ids == ["nous/curated", "sale/deep"]


def test_anonymous_reads_still_resolve(monkeypatch):
    """No key → empty fingerprint → registered key equals written key, as before."""
    mp._pricing_cache.clear()
    mp._pricing_cache_retry_after.clear()
    mp._pricing_provider_cache_keys.clear()

    _serve_nous_catalog(monkeypatch, {"data": [{"id": "nous/curated", "pricing": dict(_LIST)}]})
    monkeypatch.setattr(mp, "_resolve_nous_pricing_credentials", lambda: ("", _BASE))
    warmed = mp._fetch_nous_pricing_for_provider()
    assert warmed == {"nous/curated": dict(_LIST)}

    assert mp.get_pricing_for_provider("nous", cached_only=True) == warmed
