"""Comparative S integration: real discovery, prefetch, cache and enrichment.

Only transport and persistent-write boundaries are replaced. PYTEST guards remain
active: this does not certify the network paths those guards intentionally skip.
"""
import json
import os
from collections import Counter
from pathlib import Path
import socket
import threading
import time

import pytest


@pytest.mark.parametrize("singleton", [False, True])
@pytest.mark.parametrize("hot", [False, True])
@pytest.mark.parametrize("refresh", [False, True])
def test_real_picker_chain(tmp_path, monkeypatch, singleton, hot, refresh):
    from agent import models_dev
    from hermes_cli import auth, inventory, models
    from hermes_cli.models import CANONICAL_PROVIDERS
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
    # More than three eligible providers, plus the exhausted Codex pool.
    slugs = ["openrouter", "deepseek", "zai", "xai", "openai-codex"]
    for key in ["OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GLM_API_KEY", "XAI_API_KEY"]:
        monkeypatch.setenv(key, "synthetic-test-key")
    pool = [{"id": "synthetic-account", "source": "device_code", "auth_type": "oauth",
             "access_token": "synthetic-access", "refresh_token": "synthetic-refresh",
             "last_status": "exhausted", "last_status_at": 2000000000,
             "last_error_reset_at": 4102444800}]
    target = root / "auth.json"
    target.write_text(json.dumps({"credential_pool": {"openai-codex": pool},
        "providers": {"openai-codex": {"tokens": {"access_token": "synthetic-access",
            "refresh_token": "synthetic-refresh", "expires_at": 4102444800}}} if singleton else {}}))
    before = target.read_bytes()
    data = {slug: {"id": slug, "name": slug, "env": [key], "models": {
        "synthetic-model": {"id": "synthetic-model", "name": "Synthetic", "reasoning": True}}}
        for slug, key in zip(slugs, ["OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GLM_API_KEY", "XAI_API_KEY"])}
    # Seed data, not replacement fetch/enrichment functions.
    monkeypatch.setattr(models_dev, "_models_dev_cache", data)
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", time.time())
    if hot:
        models._save_provider_models_cache({slug: {"fp": models._credential_fingerprint(slug),
            "at": time.time(), "models": ["synthetic-model"]} for slug in slugs})
    counts = Counter()
    lock = threading.Lock()
    def boundary(name):
        def deny(*args, **kwargs):
            with lock:
                counts[name] += 1
            raise OSError("synthetic boundary: " + name)
        return deny
    monkeypatch.setattr(socket.socket, "connect", boundary("socket"))
    import requests, httpx, urllib.request
    monkeypatch.setattr(requests.sessions.Session, "request", boundary("requests"))
    monkeypatch.setattr(httpx.Client, "send", boundary("httpx"))
    monkeypatch.setattr(urllib.request, "urlopen", boundary("urlopen"))
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boundary("urllib_opener"))
    # Final persistence boundaries, never load_pool/discovery/prefetch/enrichment.
    monkeypatch.setattr(auth, "_save_auth_store", boundary("auth_write"))
    monkeypatch.setattr(auth, "_write_private_file_atomic", boundary("atomic_auth_write"))
    monkeypatch.setattr(models, "_write_json_cache", boundary("catalog_write"))
    excluded = sorted((set(auth.PROVIDER_REGISTRY) | set(HERMES_OVERLAYS)
                       | {cp.slug for cp in CANONICAL_PROVIDERS}) - set(slugs))
    ctx = inventory.ConfigContext("", "", "", {}, [], excluded)
    existing = set(threading.enumerate())
    import sys
    flows = Counter()
    watched = {"_collect_authed_provider_slugs", "_prefetch_provider_models_parallel",
               "_build_curated_lists", "_apply_capabilities", "_apply_featured", "_prewarm_pricing_async"}
    def observe(frame, event, arg):
        if event == "call" and frame.f_code.co_name in watched:
            flows[frame.f_code.co_name] += 1
    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        payload = inventory.build_model_options_payload(ctx, refresh=refresh)
    finally:
        sys.setprofile(previous)
    assert flows["_build_curated_lists"] == 1
    assert flows["_apply_capabilities"] == flows["_apply_featured"] == 1
    assert flows["_collect_authed_provider_slugs"] == (0 if refresh else 1)
    assert flows["_prefetch_provider_models_parallel"] == (0 if refresh else 1)
    assert flows["_prewarm_pricing_async"] == (0 if refresh else 1)
    # Drain real background prewarm before measuring or removing boundaries.
    for thread in set(threading.enumerate()) - existing:
        thread.join(timeout=10)
        assert not thread.is_alive(), thread.name
    assert target.read_bytes() == before
    assert not (profile / "auth.json").exists()
    assert counts["auth_write"] == 0
    assert counts["socket"] == 0  # all transport attempts intercepted earlier
    baseline = json.loads((Path(__file__).parent / "fixtures/picker_chain_base_counts.json").read_text())
    case = f"{singleton}-{hot}-{refresh}"
    expected = baseline["cases"][case]
    # This is a no-ADDITIONAL-effects assertion, not a claim of zero historical
    # I/O. The original archive was executed under these same boundaries.
    assert all(value <= expected.get(name, 0) for name, value in counts.items()), (counts, expected)
    rows = payload["providers"]
    assert set(slugs[:4]) <= {row["slug"] for row in rows}
    for row in rows:
        assert "capabilities" in row and "featured_models" in row
    # These counters are compared against the original pre-S archive by the
    # evidence harness; do not mistake pre-existing attempts for new effects.
    evidence = Path(__file__).resolve().parents[2] / ".local-evidence" / "chain-counts"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / f"{singleton}-{hot}-{refresh}.json").write_text(json.dumps({
        "singleton": singleton, "hot": hot, "refresh": refresh,
        "flows": dict(sorted(flows.items())), "calls": dict(sorted(counts.items())), "providers": [r["slug"] for r in rows]}, sort_keys=True))
