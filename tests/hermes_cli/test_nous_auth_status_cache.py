"""Tests for the get_nous_auth_status() process-level cache.

The cache avoids re-validating Nous credentials on every menu paint —
`hermes tools` → "All Platforms" used to fire ~31 OAuth refresh POSTs
against portal.nousresearch.com during one render. The cache is keyed
on auth.json path + content digest (not mtime — this cache stores the
live access_token, so it guards credential identity, same class of bug
already fixed for the Vertex SA credential cache and this file's own
_load_global_auth_store()/_oauth_heal_clean_marks caches) so profile
switches stay isolated while login/logout flows invalidate naturally;
tests and other writers can also call invalidate_nous_auth_status_cache().
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


def test_metadata_preserving_replace_is_not_served_stale(tmp_path, monkeypatch):
    """A stat-only cache key would miss a content change that preserves mtime.

    This is the direct regression test for the credential-cache anti-pattern
    already fixed elsewhere in this codebase (Vertex SA credential cache,
    agent/vertex_adapter.py's _read_sa_file; and this file's own
    _load_global_auth_store()/_oauth_heal_clean_marks caches): a tool that
    replaces a file's bytes while preserving its mtime — dotfile sync,
    backup/restore, `rsync -a` — must not leave get_nous_auth_status()
    serving a stale (here, a *different account's*) access_token for the
    rest of the TTL window.

    Bumping mtime alone would NOT discriminate old vs. new code (content
    changes normally bump mtime too), so the file's exact original
    st_mtime_ns is restored via os.utime after the content swap.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    auth_file = _seed_auth_file(tmp_path)

    from hermes_cli import auth as auth_mod

    auth_mod.invalidate_nous_auth_status_cache()

    call_n = [0]

    def fake_compute():
        call_n[0] += 1
        return {"logged_in": True, "source": "auth_store", "access_token": f"token-{call_n[0]}"}

    with patch.object(auth_mod, "_compute_nous_auth_status", side_effect=fake_compute):
        first = auth_mod.get_nous_auth_status()
        assert first["access_token"] == "token-1"

        original_stat = auth_file.stat()

        # External metadata-preserving replace: the file's bytes change (a
        # different account's credentials, or a stale backup, lands in
        # auth.json) but its mtime is restored to the exact value the cache
        # observed on the first call.
        auth_file.write_text(
            json.dumps({"providers": {}, "marker": "replaced"}), encoding="utf-8"
        )
        os.utime(auth_file, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        assert auth_file.stat().st_mtime_ns == original_stat.st_mtime_ns  # sanity

        second = auth_mod.get_nous_auth_status()

    assert call_n[0] == 2, (
        "Content changed (bytes differ) even though mtime did not — the "
        "digest-keyed cache must detect it and recompute, not keep serving "
        "the stale cached access_token."
    )
    assert second["access_token"] == "token-2"

    auth_mod.invalidate_nous_auth_status_cache()
