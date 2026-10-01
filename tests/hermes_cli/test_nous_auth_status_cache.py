"""Tests for the get_nous_auth_status() process-level cache.

The cache avoids re-validating Nous credentials on every menu paint —
`hermes tools` → "All Platforms" used to fire ~31 OAuth refresh POSTs
against portal.nousresearch.com during one render. The cache is keyed
on auth.json path + mtime so profile switches stay isolated while
login/logout flows invalidate naturally; tests and other writers can
also call invalidate_nous_auth_status_cache().
"""

from __future__ import annotations
import auth.providers.nous_status as _auth_auth_providers_nous_status

from hermes_cli.config_credentials import credential_pool_environment as _phase6_auth_environment

import auth.providers.nous as _auth_auth_providers_nous


import json
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

    _auth_auth_providers_nous_status.invalidate_nous_auth_status_cache()

    call_count = {"n": 0}

    def fake_compute(*, environment):
        environment.require_current_scope()
        call_count["n"] += 1
        return {"logged_in": False, "source": "auth_store", "call": call_count["n"]}

    with patch.object(_auth_auth_providers_nous_status, "_compute_nous_auth_status", side_effect=fake_compute):
        first = _auth_auth_providers_nous_status.get_nous_auth_status(environment=_phase6_auth_environment())
        second = _auth_auth_providers_nous_status.get_nous_auth_status(environment=_phase6_auth_environment())
        third = _auth_auth_providers_nous_status.get_nous_auth_status(environment=_phase6_auth_environment())

    assert call_count["n"] == 1, (
        f"_compute_nous_auth_status was called {call_count['n']}× — "
        "cache is not deduplicating within TTL."
    )
    # Each call returns a copy so callers can't mutate the cached dict.
    assert first == second == third
    first["mutated"] = True
    assert "mutated" not in _auth_auth_providers_nous_status.get_nous_auth_status(environment=_phase6_auth_environment())

    _auth_auth_providers_nous_status.invalidate_nous_auth_status_cache()
