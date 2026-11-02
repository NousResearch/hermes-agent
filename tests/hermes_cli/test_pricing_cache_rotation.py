"""Rotating credentials do not grow the per-credential pricing catalog cache without bound.

A gateway up for three days held 139 copies of the same Nous catalog (about 26 MB).
"""
from hermes_cli import models_pricing as mp


def test_the_pricing_cache_keeps_the_newest_credentials_per_endpoint():
    mp._pricing_cache.clear()
    mp._pricing_cache_retry_after.clear()
    other = "https://other.example" + mp._pricing_auth_fingerprint("keep-me")
    mp._cache_catalog(other, {"m": {"prompt": "1"}})
    keys = ["https://api.example" + mp._pricing_auth_fingerprint(f"token-{i}") for i in range(20)]
    for key in keys:
        mp._cache_catalog(key, {"m": {"prompt": "1"}})
    kept = [key for key in mp._pricing_cache if key.startswith("https://api.example")]
    assert kept == keys[-mp._AUTHED_CATALOGS_PER_ENDPOINT:]
    assert other in mp._pricing_cache, "another endpoint's catalog is not touched"
    assert mp.peek_cached_pricing("https://api.example/v1") is mp._pricing_cache[keys[-1]]
