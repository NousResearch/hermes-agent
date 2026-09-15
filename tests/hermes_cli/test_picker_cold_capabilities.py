"""Recorded Nous rows must not resolve auth to enrich a cold capability catalog.

Derived from the independent S-final-review comparative fixture. Discovery,
pricing, capabilities and OAuth resolution stay real; only final transports and
persistence are denied. PYTEST_CURRENT_TEST and its production guards stay set.
"""
import json
import socket
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import pytest


@pytest.mark.parametrize("singleton", ["none", "global", "local"])
@pytest.mark.parametrize("expires", [1, 4102444800], ids=["expired", "valid"])
@pytest.mark.parametrize("hot", [False, True], ids=["cold", "hot"])
@pytest.mark.parametrize("refresh", [False, True], ids=["open", "refresh"])
@pytest.mark.parametrize("status", ["exhausted", "dead"])
@pytest.mark.parametrize("tier", [None, True, False], ids=["unknown", "free", "paid"])
def test_recorded_caps_no_auth(tmp_path, monkeypatch, singleton, expires, hot, refresh, status, tier):
    from agent import models_dev
    from hermes_cli import auth, inventory, models, models_pricing, anon_auth
    from hermes_cli.providers import HERMES_OVERLAYS

    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "alpha"
    profile.mkdir(parents=True)
    home = tmp_path / "fakehome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("CODEX_HOME", str(home / "codex"))
    assert "PYTEST_CURRENT_TEST" in __import__("os").environ
    target = profile / "auth.json"
    store = {"providers": {}, "credential_pool": {"nous": [{
        "id": "synthetic", "source": "device_code", "auth_type": "oauth",
        "access_token": "synthetic", "last_status": status,
        "last_error_reset_at": 4102444800,
    }]}}
    credentials = {"access_token": "synthetic", "refresh_token": "synthetic-refresh", "expires_at": expires}
    if singleton == "local":
        store["providers"]["nous"] = credentials
    elif singleton == "global":
        (root / "auth.json").write_text(json.dumps({"providers": {"nous": credentials}}))
    target.write_text(json.dumps(store))
    before = target.read_bytes()
    global_before = (root / "auth.json").read_bytes() if singleton == "global" else None
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
        "paid-model": {"prompt": "0.000001", "completion": "0.000002"},
    }})
    monkeypatch.setattr(models_pricing, "_pricing_cache_retry_after", {})
    monkeypatch.setattr(models, "_nous_reasoning_caps_cache", {
        "free-model": {"supports_reasoning": False},
        "paid-model": {"supports_reasoning": True, "mandatory": True},
    } if hot else None)
    for name, value in [("_nous_caps_disk_checked", False), ("_nous_caps_warm_started", False),
                        ("_nous_reasoning_caps_failed_at", None)]:
        monkeypatch.setattr(models, name, value)
    counts = Counter()
    stacks = []

    def boundary(name):
        def deny(*args, **kwargs):
            counts[name] += 1
            if name == "urlopen":
                counts["static_catalog" if "_fetch_manifest" in [f.name for f in traceback.extract_stack()]
                       else "other_urlopen"] += 1
            if name == "httpx":
                stacks.append({"functions": [f.name for f in traceback.extract_stack()],
                               "method": args[1].method, "url": str(args[1].url)})
            raise OSError("synthetic boundary " + name)
        return deny

    import requests
    import httpx
    import urllib.request
    monkeypatch.setattr(socket.socket, "connect", boundary("socket"))
    monkeypatch.setattr(requests.sessions.Session, "request", boundary("requests"))
    monkeypatch.setattr(httpx.Client, "send", boundary("httpx"))
    monkeypatch.setattr(urllib.request, "urlopen", boundary("urlopen"))
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boundary("opener"))
    monkeypatch.setattr(auth, "_save_auth_store", boundary("auth_write"))
    monkeypatch.setattr(auth, "atomic_json_write", boundary("atomic_auth_write"))
    monkeypatch.setattr(models, "_write_json_cache", boundary("cache_write"))
    original_resolve = auth.resolve_nous_runtime_credentials
    original_open = Path.open

    def tracked_open(path, mode="r", *args, **kwargs):
        if path.name == "auth.lock" and any(flag in mode for flag in ("a", "w", "+")):
            counts["auth_lock_open"] += 1
            if "nous_catalog_url" in [f.name for f in traceback.extract_stack()]:
                counts["caps_auth_lock_open"] += 1
        return original_open(path, mode, *args, **kwargs)

    def resolve(*args, **kwargs):
        names = [f.name for f in traceback.extract_stack()]
        counts["caps_resolve" if "nous_catalog_url" in names else "other_resolve"] += 1
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(auth, "resolve_nous_runtime_credentials", resolve)
    excluded = sorted((set(auth.PROVIDER_REGISTRY) | set(HERMES_OVERLAYS)
                       | {cp.slug for cp in models.CANONICAL_PROVIDERS}) - {"nous"})
    existing = set(threading.enumerate())
    payload = inventory.build_model_options_payload(
        inventory.ConfigContext("", "", "", {}, [], excluded), refresh=refresh)
    for thread in set(threading.enumerate()) - existing:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert target.read_bytes() == before
    assert ((root / "auth.json").read_bytes() if singleton == "global" else None) == global_before
    assert counts["socket"] == 0
    # These are final counters, not exceptions swallowed by the production path.
    assert counts["httpx"] == 0, stacks
    assert counts["caps_resolve"] == counts["caps_auth_lock_open"] == 0
    # Historical static model-catalog attempts are compared separately to the base;
    # they are not capability/OAuth enrichment and remain intercepted.
    assert counts["urlopen"] == counts["static_catalog"]
    assert all(counts[name] == 0 for name in ("requests", "other_urlopen", "opener", "cache_write"))
    row = next(row for row in payload["providers"] if row["slug"] == "nous")
    assert row["models"] == ids
    assert row["availability_source"] == "recorded_pool"
    assert row["warning"] and "not checked" in row["warning"]
    assert row.get("free_tier_pending", False) is (tier is None)
    assert row["unavailable_models"] == (ids if tier is None else (["paid-model"] if tier else []))
    if hot:
        assert row["capabilities"]["free-model"]["reasoning"] is False
        assert row["capabilities"]["paid-model"]["can_disable_reasoning"] is False
    else:
        assert "can_disable_reasoning" not in row["capabilities"]["paid-model"]
    # UI reads must not poison hydrate/warm guards used by ordinary runtime callers.
    assert models._nous_caps_disk_checked is False
    assert models._nous_caps_warm_started is False


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("hot", [False, True])
def test_memory_peek_does_not_poison_normal_refresh(tmp_path, monkeypatch, routed, hot):
    """A UI peek cannot use another profile or disable later normal OAuth resolution."""
    import httpx
    from hermes_cli import auth, inventory, models, models_profile_cache, models_reasoning_caps as caps
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    home = tmp_path / "home"
    profile = home / "hermes-root" / "profiles" / "alpha"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("CODEX_HOME", str(home / "codex"))
    global_auth = profile.parents[1] / "auth.json"
    global_auth.write_text(json.dumps({"providers": {"nous": {
        "access_token": "synthetic", "refresh_token": "synthetic-refresh", "expires_at": 1}}}))
    before = global_auth.read_bytes()
    # A disk catalog is not permission to resolve credentials to name its URL.
    (profile / "cache").mkdir()
    disk = profile / "cache" / "reasoning_caps.json"
    disk.write_text(json.dumps({"https://synthetic.invalid/v1/models": {
        "ts": time.time(), "caps": {"model": {"supports_reasoning": False}}}}))
    disk_before = disk.read_bytes()
    monkeypatch.setattr(models_profile_cache, "_SLOTS_BY_HOME", {})
    monkeypatch.setattr(models, "_nous_reasoning_caps_cache", {"model": {"supports_reasoning": False}})
    for name, value in [("_nous_caps_disk_checked", False), ("_nous_caps_warm_started", False),
                        ("_nous_reasoning_caps_failed_at", None)]:
        monkeypatch.setattr(models, name, value)
    seen = Counter()

    def deny(name):
        def stop(*args, **kwargs):
            seen[name] += 1
            raise OSError("synthetic boundary " + name)
        return stop

    monkeypatch.setattr(httpx.Client, "send", deny("oauth"))
    monkeypatch.setattr(socket.socket, "connect", deny("socket"))
    monkeypatch.setattr(auth, "_save_auth_store", deny("auth_write"))
    monkeypatch.setattr(auth, "atomic_json_write", deny("atomic_write"))
    monkeypatch.setattr(models, "_write_json_cache", deny("cache_write"))
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", deny("urlopen"))
    token = set_hermes_home_override(profile) if routed else None
    try:
        if not routed:
            monkeypatch.setattr(models, "_nous_reasoning_caps_cache", None)
        assert caps._NOUS_CAPS.get("cache") is None  # not the launch profile's hot value
        if hot:
            caps._NOUS_CAPS.set("cache", {"model": {"supports_reasoning": True, "mandatory": True}})
        result = caps.nous_model_reasoning_capabilities(" model ", memory_only=True, allow_fetch=True)
        assert result == ({"supports_reasoning": True, "mandatory": True} if hot else None)
        assert not seen
        assert caps._NOUS_CAPS.get("disk_checked") is False
        assert caps._NOUS_CAPS.get("warm_started") is False
        # A normal (non-warning) reader still follows its original cold path.
        caps._NOUS_CAPS.set("cache", None)
        inventory._reasoning_catalog_reader("nous")
        assert seen["oauth"] == 1
        assert seen["socket"] == seen["cache_write"] == seen["urlopen"] == 0
        assert global_auth.read_bytes() == before
        assert disk.read_bytes() == disk_before
    finally:
        if token is not None:
            reset_hermes_home_override(token)
