# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""The fallback gate must not read "no wait information" as "exhausted".

``CredentialPool.next_available_at()`` answers ``None`` both when a credential is
available now and when no exhausted entry carries a usable recovery time — its own
docstring tells callers to treat ``None`` as "no wait information". A credential whose
entry the availability scan never counts (metadata-only api_key rows, e.g. a provider
authenticated from a key file rather than a stored runtime key) is neither available
nor exhausted: ``has_available()`` is False and ``next_available_at()`` is ``None``.
Treating that ``None`` as "exhausted" skipped every same-provider fallback of such a
profile, so a transient 429 from the primary went straight to the user (#140568).

A pool that is genuinely out of rotation still skips: every entry DEAD, or a cooldown
longer than the 600 s Retry-After cap the fallback walk already waits out.
"""

from __future__ import annotations

import json
import time

from agent.credential_pool import load_pool
from agent.fallback_cooldown import _candidate_pool_exhausted

MODEL = "google/gemini-4-argon"


def _load_vertex_pool(tmp_path, monkeypatch, entries, *, provider="vertex"):
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {provider: entries}}, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return load_pool(provider)


class _Agent:
    def __init__(self, pool):
        self._credential_pool = pool


def _entry(entry_id, **overrides):
    entry = {
        "id": entry_id,
        "label": entry_id,
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "***",
        "base_url": "https://us-central1-aiplatform.googleapis.com",
    }
    entry.update(overrides)
    return entry


def test_metadata_only_credential_is_not_exhausted(tmp_path, monkeypatch):
    """The reported shape: an entry the availability scan skips, with no cooldown at all.

    ``access_token`` is empty the way a key-file credential persists — the runtime
    token is minted later, so ``_available_entries`` skips the row before it ever
    counts as available, and nothing marks it exhausted either.
    """
    pool = _load_vertex_pool(tmp_path, monkeypatch, [_entry("vertex-sa", access_token="")])

    # Premises of the report, asserted so a wrong premise fails loudly instead of
    # silently passing the fix.
    assert pool.has_credentials() is True
    assert pool.has_available(model=MODEL) is False
    assert pool.next_available_at(model=MODEL) is None
    assert pool.entries()[0].last_status != "dead"

    assert _candidate_pool_exhausted(_Agent(pool), "vertex", MODEL) is False


def test_short_model_cooldown_still_gets_its_chance(tmp_path, monkeypatch):
    """Under the 600 s cap the candidate is tried once rather than skipped."""
    pool = _load_vertex_pool(
        tmp_path, monkeypatch,
        [_entry("vertex-a", model_cooldowns={MODEL: time.time() + 120})],
    )
    assert pool.has_available(model=MODEL) is False
    assert _candidate_pool_exhausted(_Agent(pool), "vertex", MODEL) is False


def test_long_model_cooldown_still_skips(tmp_path, monkeypatch):
    """Past the cap the fallback walk could not wait it out, so it skips (#89401)."""
    pool = _load_vertex_pool(
        tmp_path, monkeypatch,
        [_entry("vertex-a", model_cooldowns={MODEL: time.time() + 3600})],
    )
    assert _candidate_pool_exhausted(_Agent(pool), "vertex", MODEL) is True


def test_all_dead_pool_still_skips(tmp_path, monkeypatch):
    """A revoked credential never re-enters rotation: skipping keeps that intent."""
    pool = _load_vertex_pool(
        tmp_path, monkeypatch,
        [
            _entry("vertex-dead", last_status="dead", last_status_at=time.time()),
            _entry("vertex-dead-2", access_token="", last_status="dead", last_status_at=time.time()),
        ],
    )
    assert pool.has_available(model=MODEL) is False
    assert pool.next_available_at(model=MODEL) is None
    assert _candidate_pool_exhausted(_Agent(pool), "vertex", MODEL) is True
