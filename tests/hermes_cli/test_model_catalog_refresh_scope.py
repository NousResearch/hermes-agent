"""Background catalog writes must retain the requesting profile's home."""

import json
import os
import threading
import time

from hermes_cli import model_catalog, models
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


def test_stale_catalog_refresh_stays_in_requesting_profile(tmp_path, monkeypatch):
    default_home = tmp_path / "hermes"
    profile_home = default_home / "profiles" / "worker"
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(model_catalog, "_catalog_swr_inflight", False)
    model_catalog.reset_cache()
    old = {"version": 1, "providers": {"nous": {"models": [{"id": "old-model"}]}}}
    fresh = {"version": 1, "providers": {"nous": {"models": [{"id": "new-model"}]}}}
    paths = [home / "cache" / "model_catalog.json" for home in (default_home, profile_home)]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(old), encoding="utf-8")
        os.utime(path, (1, 1))
    default_before = paths[0].read_bytes()
    release_fetch = threading.Event()
    write_finished = threading.Event()
    write_disk_cache = model_catalog._write_disk_cache

    def fetch(*args):
        assert release_fetch.wait(5), "caller did not release the refresh"
        return fresh

    def write(data):
        try:
            write_disk_cache(data)
        finally:
            write_finished.set()

    # Only replace the HTTP boundary; use a real worker thread and disk I/O.
    monkeypatch.setattr(model_catalog, "_fetch_manifest_with_fallback", fetch)
    monkeypatch.setattr(model_catalog, "_write_disk_cache", write)
    token = set_hermes_home_override(profile_home)
    try:
        assert model_catalog.get_catalog() == old
    finally:
        reset_hermes_home_override(token)
        release_fetch.set()
    assert write_finished.wait(5), "background catalog write did not finish"

    assert json.loads(paths[1].read_text(encoding="utf-8")) == fresh
    assert paths[0].read_bytes() == default_before


def test_provider_refresh_reads_and_writes_in_requesting_profile(tmp_path, monkeypatch):
    default_home = tmp_path / "hermes"
    profile_home = default_home / "profiles" / "worker"
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(models, "_swr_refresh_inflight", set())
    provider = "opencode-free"  # Keyless, so the test needs no credentials.
    row = {"fp": models._credential_fingerprint(provider), "at": time.time() - 7200,
           "models": ["old-model"]}
    paths = [home / "provider_models_cache.json" for home in (default_home, profile_home)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({provider: row}), encoding="utf-8")
    default_before = paths[0].read_bytes()
    release_fetch = threading.Event()
    write_finished = threading.Event()
    fetch_homes = []
    save = models._save_provider_models_cache

    def fetch(*args, **kwargs):
        assert release_fetch.wait(5), "caller did not release the refresh"
        fetch_homes.append(get_hermes_home())
        return ["new-model"]

    def write(data):
        try:
            save(data)
        finally:
            write_finished.set()

    monkeypatch.setattr(models, "provider_model_ids", fetch)
    monkeypatch.setattr(models, "_save_provider_models_cache", write)
    token = set_hermes_home_override(profile_home)
    try:
        assert models.cached_provider_model_ids(provider) == ["old-model"]
    finally:
        reset_hermes_home_override(token)
        release_fetch.set()
    assert write_finished.wait(5), "background provider write did not finish"

    assert fetch_homes == [profile_home]
    assert json.loads(paths[1].read_text(encoding="utf-8"))[provider]["models"] == ["new-model"]
    assert paths[0].read_bytes() == default_before
