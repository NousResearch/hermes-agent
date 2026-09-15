"""Nous options integration with real cache readers and enrichment."""
import json
import os
import socket
import threading
import time
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("tier", [None, True, False])
@pytest.mark.parametrize("status", ["exhausted", "dead"])
def test_nous_options_entitlement(tmp_path, monkeypatch, refresh, tier, status):
    from agent import models_dev
    from hermes_cli import auth, inventory, models, models_pricing, anon_auth
    from hermes_cli.providers import HERMES_OVERLAYS
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "alpha"
    profile.mkdir(parents=True)
    home = tmp_path / "fakehome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("CODEX_HOME", str(home / "codex"))
    from hermes_constants import get_default_hermes_root
    assert get_default_hermes_root() == root
    assert "PYTEST_CURRENT_TEST" in os.environ
    target = profile / "auth.json"
    target.write_text(json.dumps({"providers": {"nous": {"tokens": {"access_token": "synthetic"}}},
        "credential_pool": {"nous": [{"id": "synthetic", "source": "device_code", "auth_type": "oauth",
            "access_token": "synthetic", "last_status": status, "last_error_reset_at": 4102444800}]}}))
    before = target.read_bytes()
    assert not anon_auth.has_guest()
    ids = ["free-model", "paid-model"]
    monkeypatch.setitem(models._PROVIDER_MODELS, "nous", ids)
    monkeypatch.setattr(models_dev, "_models_dev_cache", {"nous": {"models": {}}})
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", time.time())
    key = models._pricing_profile_key()
    monkeypatch.setattr(models, "_free_tier_cache", {} if tier is None else {key: (tier, time.monotonic())})
    endpoint = "https://synthetic.invalid"
    monkeypatch.setattr(models_pricing, "_pricing_provider_cache_keys", {(key, "nous"): endpoint})
    monkeypatch.setattr(models_pricing, "_pricing_cache", {endpoint: {
        "free-model": {"prompt": "0", "completion": "0"},
        "paid-model": {"prompt": "0.000001", "completion": "0.000002"}}})
    monkeypatch.setattr(models_pricing, "_pricing_cache_retry_after", {})
    # This matrix isolates pricing/entitlement with an already populated reasoning
    # cache. Cold Nous capability hydration has its own pre-existing credential
    # resolution path, outside the pricing repair (documented in S-followup).
    monkeypatch.setattr(models, "_nous_reasoning_caps_cache", {mid: {"supports_reasoning": True} for mid in ids})
    transport = Mock(side_effect=OSError("synthetic transport"))
    import requests, httpx, urllib.request
    monkeypatch.setattr(socket.socket, "connect", transport)
    monkeypatch.setattr(requests.sessions.Session, "request", transport)
    monkeypatch.setattr(httpx.Client, "send", transport)
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", transport)
    writes = Mock(side_effect=AssertionError("no auth writes"))
    monkeypatch.setattr(auth, "atomic_json_write", writes)
    # Observe forbidden authorization/entitlement boundaries without replacing the
    # actual cache reader, discovery or any enrichment collaborator.
    live_tier = Mock(side_effect=AssertionError("no live tier"))
    monkeypatch.setattr(models, "check_nous_free_tier", live_tier)
    resolve = Mock(side_effect=AssertionError("no pricing credential resolution"))
    monkeypatch.setattr(models_pricing, "_resolve_nous_pricing_credentials", resolve)
    excluded = sorted((set(auth.PROVIDER_REGISTRY) | set(HERMES_OVERLAYS)
        | {cp.slug for cp in models.CANONICAL_PROVIDERS}) - {"nous"})
    existing = set(threading.enumerate())
    payload = inventory.build_model_options_payload(inventory.ConfigContext("nous", "", "", {}, [], excluded), refresh=refresh)
    for thread in set(threading.enumerate()) - existing:
        thread.join(timeout=10)
        assert not thread.is_alive()
    row = next(row for row in payload["providers"] if row["slug"] == "nous")
    assert row["models"] == ids
    assert row["availability_source"] == "recorded_pool"
    assert row["auth_state"] == ("invalid" if status == "dead" else "present")
    assert row.get("free_tier_pending", False) is (tier is None)
    assert row["unavailable_models"] == (ids if tier is None else ["paid-model"] if tier else [])
    assert "not checked" in row["warning"]
    assert "capabilities" in row and "featured_models" in row
    live_tier.assert_not_called()
    resolve.assert_not_called()
    writes.assert_not_called()
    assert target.read_bytes() == before
