"""Tests for the get_nous_auth_status() process-level cache.

The cache avoids re-validating Nous credentials on every menu paint —
`hermes tools` → "All Platforms" used to fire ~31 OAuth refresh POSTs
against portal.nousresearch.com during one render. The cache is keyed
on auth.json mtime so login/logout flows invalidate naturally; tests
and other writers can also call invalidate_nous_auth_status_cache().
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch


def _seed_auth_file(tmp_path):
    """Drop a placeholder auth.json into the test HERMES_HOME.

    The exact content doesn't matter for cache-key purposes — only that
    the file exists and we can mutate it to bump mtime.
    """
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    return auth


def test_get_nous_auth_status_caches_consecutive_calls(tmp_path, monkeypatch):
    """A second call within the TTL skips re-computing the snapshot."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_auth_file(tmp_path)

    from hermes_cli import auth as auth_mod

    auth_mod.invalidate_nous_auth_status_cache()

    call_count = {"n": 0}

    def fake_compute():
        call_count["n"] += 1
        return {"logged_in": False, "source": "auth_store", "call": call_count["n"]}

    with patch.object(auth_mod, "_compute_nous_auth_status", side_effect=fake_compute):
        first = auth_mod.get_nous_auth_status()
        second = auth_mod.get_nous_auth_status()
        third = auth_mod.get_nous_auth_status()

    assert call_count["n"] == 1, (
        f"_compute_nous_auth_status was called {call_count['n']}× — "
        "cache is not deduplicating within TTL."
    )
    # Each call returns a copy so callers can't mutate the cached dict.
    assert first == second == third
    first["mutated"] = True
    assert "mutated" not in auth_mod.get_nous_auth_status()

    auth_mod.invalidate_nous_auth_status_cache()


def test_get_nous_auth_status_invalidates_on_auth_file_mtime(tmp_path, monkeypatch):
    """Touching auth.json (login/logout) forces a re-compute."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    auth_path = _seed_auth_file(tmp_path)

    from hermes_cli import auth as auth_mod

    auth_mod.invalidate_nous_auth_status_cache()

    call_count = {"n": 0}

    def fake_compute():
        call_count["n"] += 1
        return {"logged_in": False, "source": "auth_store", "call": call_count["n"]}

    with patch.object(auth_mod, "_compute_nous_auth_status", side_effect=fake_compute):
        auth_mod.get_nous_auth_status()
        # Bump mtime forward so coarse-resolution filesystems still record
        # a change.
        future = auth_path.stat().st_mtime + 5.0
        os.utime(auth_path, (future, future))
        auth_mod.get_nous_auth_status()

    assert call_count["n"] == 2, (
        "auth.json mtime change should invalidate the cache, but only "
        f"{call_count['n']} compute call(s) happened."
    )

    auth_mod.invalidate_nous_auth_status_cache()


def test_invalidate_nous_auth_status_cache_forces_recompute(tmp_path, monkeypatch):
    """Explicit invalidate forces the next call to re-compute."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_auth_file(tmp_path)

    from hermes_cli import auth as auth_mod

    auth_mod.invalidate_nous_auth_status_cache()

    call_count = {"n": 0}

    def fake_compute():
        call_count["n"] += 1
        return {"logged_in": False, "source": "auth_store"}

    with patch.object(auth_mod, "_compute_nous_auth_status", side_effect=fake_compute):
        auth_mod.get_nous_auth_status()
        auth_mod.invalidate_nous_auth_status_cache()
        auth_mod.get_nous_auth_status()

    assert call_count["n"] == 2

    auth_mod.invalidate_nous_auth_status_cache()


def test_get_nous_auth_status_caches_failure_path(tmp_path, monkeypatch):
    """Logged-out snapshots are cached too — that's where the cost was.

    Teknium's case: ~31 cache misses per `hermes tools` "All Platforms"
    menu paint, all returning logged_in=False after a failed refresh POST.
    The whole point of the cache is to memoise that failure path too.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_auth_file(tmp_path)

    from hermes_cli import auth as auth_mod

    auth_mod.invalidate_nous_auth_status_cache()

    call_count = {"n": 0}

    def fake_compute():
        call_count["n"] += 1
        return {"logged_in": False, "source": "auth_store", "error": "refresh failed"}

    with patch.object(auth_mod, "_compute_nous_auth_status", side_effect=fake_compute):
        for _ in range(10):
            auth_mod.get_nous_auth_status()

    assert call_count["n"] == 1, (
        f"Logged-out snapshots must cache; got {call_count['n']} computes for 10 calls."
    )

    auth_mod.invalidate_nous_auth_status_cache()
