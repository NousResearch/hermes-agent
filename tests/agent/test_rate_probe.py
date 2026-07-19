"""Tests for the Arc status-bar rate/credit probe (agent/rate_probe.py)."""
import io
import json
import time
from unittest import mock

import agent.rate_probe as rp


def _fake_response(payload: dict):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    return _Resp()


def setup_function(_fn):
    # Isolate module-level caches between tests.
    rp._cache.clear()
    rp._inflight.clear()


def test_non_openrouter_provider_returns_none():
    assert rp.get_rate_limits("anthropic", "sk-ant-xxx") is None
    assert rp.get_rate_limits("openrouter", "") is None
    assert rp.get_rate_limits("", "key") is None


def test_openrouter_balance_parsed_with_limit():
    payload = {"data": {"usage": 6.0, "limit": 20.0, "limit_remaining": 14.0}}
    with mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        # First call kicks off async fetch, returns None (no data yet).
        assert rp.get_rate_limits("openrouter", "sk-or-abcdefgh") is None
        # Wait for the daemon thread to populate the cache.
        for _ in range(50):
            if rp._cache:
                break
            time.sleep(0.02)
        out = rp.get_rate_limits("openrouter", "sk-or-abcdefgh")
    assert out is not None
    assert out["source"] == "openrouter"
    c = out["credits"]
    assert c["remaining_usd"] == 14.0
    assert c["limit_usd"] == 20.0
    assert c["used_percentage"] == 30  # 6/20


def test_openrouter_no_limit_still_reports_balance():
    payload = {"data": {"usage": 3.5, "limit": None, "limit_remaining": None}}
    with mock.patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        entry = rp._openrouter_key_balance("sk-or-x")
    # usage present, remaining absent → still surfaces (usage-only) but no gauge
    # since there is no limit to divide against.
    assert entry is not None
    assert "used_percentage" not in entry["credits"]


def test_fetch_fails_open_on_exception():
    with mock.patch("urllib.request.urlopen", side_effect=OSError("network down")):
        assert rp._fetch("openrouter", "sk-or-x") is None
