"""Rotating credentials do not grow the ``/models`` metadata memo without bound.

The memo is keyed per credential fingerprint and minted credentials rotate: a gateway up for
three days held 141 copies of the same Nous catalog (about 70 MB).
"""
from agent import model_metadata as mm


def test_the_metadata_memo_keeps_only_credentials_used_within_its_ttl(monkeypatch):
    from agent.model_metadata_endpoint_memo import remember_endpoint_models
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    clock = [1000.0]
    monkeypatch.setattr("agent.model_metadata_endpoint_memo.time.time", lambda: clock[0])
    for minute in range(30):  # a credential rotated every minute for half an hour
        clock[0] = 1000.0 + minute * 60
        remember_endpoint_models(("https://api.example/v1", f"sha256:{minute}"), {"m": {}})
    fresh = {key for key, at in mm._endpoint_model_metadata_cache_time.items()
             if clock[0] - at < mm._ENDPOINT_MODEL_CACHE_TTL}
    assert set(mm._endpoint_model_metadata_cache) == fresh == set(mm._endpoint_model_metadata_cache_time)
    assert ("https://api.example/v1", "sha256:29") in fresh
