"""403 cooldown: an auth-classified 403 is a user-fixable key problem, not an hour of downtime.

``_status_403`` in ``agent/error_classifier`` splits 403 two ways: provider billing walls
(OpenRouter ``key limit exceeded``, xAI spending-limit blocks, plan/credit-exhaustion walls)
become ``billing``; everything else — including an entitlement refusal for a subscription the
account does not have — falls through as an auth failure. The pool used to bench both for the
catch-all hour, which is disproportionate for the second class — the key is fine again as soon as
the user rotates it, and a five-minute cooldown lets the pool pick it up without waiting an hour.

The shorter bench must stay keyed on the classification, never on the status alone: a billing 403
retried every five minutes just re-fails forever, and the sole-credential / unverified-billing
degradations must keep applying.
"""

from __future__ import annotations

import json
import time


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _entry(
    error_code: int,
    *,
    age_seconds: float,
    cred_id: str = "cred-1",
    priority: int = 0,
    failure_reason: str | None = None,
) -> dict:
    entry = {
        "id": cred_id,
        "label": cred_id,
        "auth_type": "api_key",
        "priority": priority,
        "source": "manual",
        "access_token": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "last_status": "exhausted",
        "last_status_at": time.time() - age_seconds,
        "last_error_code": error_code,
    }
    if failure_reason is not None:
        entry["failure_reason"] = failure_reason
    return entry


def _load(tmp_path, monkeypatch, entries: list[dict]):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {"openrouter": entries}})
    from agent.credential_pool import load_pool

    return load_pool("openrouter")


# ── TTL contract ────────────────────────────────────────────────────────────


def test_auth_403_ttl_is_minutes_but_sole_degradation_still_applies():
    """A non-billing 403 gets the short bench; a one-entry pool still collapses it to 60s."""
    from agent.credential_pool import EXHAUSTED_TTL_403_SECONDS, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS, _exhausted_ttl

    assert _exhausted_ttl(403, sole_credential=False) == EXHAUSTED_TTL_403_SECONDS
    assert _exhausted_ttl(403, sole_credential=True) == EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS


def test_billing_403_keeps_the_full_bench():
    """Provider billing walls arrive as 403 too — those must not be retried every few minutes."""
    from agent.credential_pool import EXHAUSTED_TTL_DEFAULT_SECONDS, _exhausted_ttl

    assert _exhausted_ttl(403, sole_credential=False, failure_reason="billing") == EXHAUSTED_TTL_DEFAULT_SECONDS
    assert _exhausted_ttl(403, sole_credential=True, failure_reason="billing") == EXHAUSTED_TTL_DEFAULT_SECONDS


def test_unverified_billing_403_still_degrades_to_the_short_cooldown():
    """An ambiguous billing verdict must keep its existing protection (#82154)."""
    from agent.credential_pool import EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS, _exhausted_ttl

    assert (
        _exhausted_ttl(403, sole_credential=False, failure_reason="billing_unverified")
        == EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS
    )


def test_other_statuses_are_untouched():
    """Only 403 changes: 401 keeps its own short bench, throttles and unknowns keep the hour."""
    from agent.credential_pool import (
        EXHAUSTED_TTL_401_SECONDS,
        EXHAUSTED_TTL_429_SECONDS,
        EXHAUSTED_TTL_DEFAULT_SECONDS,
        _exhausted_ttl,
    )

    assert _exhausted_ttl(401, sole_credential=False) == EXHAUSTED_TTL_401_SECONDS
    assert _exhausted_ttl(429, sole_credential=False) == EXHAUSTED_TTL_429_SECONDS
    assert _exhausted_ttl(500, sole_credential=False) == EXHAUSTED_TTL_DEFAULT_SECONDS
    assert _exhausted_ttl(None, sole_credential=False) == EXHAUSTED_TTL_DEFAULT_SECONDS


# ── Pool behaviour (multi-key: sole-credential degradation cannot mask the result) ──


def test_multi_key_auth_403_pool_recovers_in_minutes_not_an_hour(tmp_path, monkeypatch):
    """Two auth-failed keys, both benched 90s ago: the pool is minutes away, not ~58 minutes.

    With the catch-all hour this reads ~3510s; with a wrongly-applied sole-credential collapse it
    would already be available (``next_available_at() is None``).
    """
    pool = _load(
        tmp_path,
        monkeypatch,
        [
            _entry(403, age_seconds=90, cred_id="cred-1", priority=0),
            _entry(403, age_seconds=90, cred_id="cred-2", priority=1),
        ],
    )
    assert pool.has_available() is False
    next_at = pool.next_available_at()
    assert next_at is not None
    remaining = next_at - time.time()
    assert 60 < remaining < 600, f"expected a minutes-scale bench, got {remaining:.0f}s"


def test_multi_key_billing_403_pool_still_waits_the_full_bench(tmp_path, monkeypatch):
    """The classification still wins: billing 403s keep the hour even in a multi-key pool."""
    pool = _load(
        tmp_path,
        monkeypatch,
        [
            _entry(403, age_seconds=90, cred_id="cred-1", priority=0, failure_reason="billing"),
            _entry(403, age_seconds=90, cred_id="cred-2", priority=1, failure_reason="billing"),
        ],
    )
    assert pool.has_available() is False
    next_at = pool.next_available_at()
    assert next_at is not None
    remaining = next_at - time.time()
    assert remaining > 3400, f"billing 403 must keep the hour-long bench, got {remaining:.0f}s"


# ── Persistence: a restart must not re-grade the bench ──────────────────────


def test_billing_403_survives_reload(tmp_path, monkeypatch):
    """The classified reason rides on the entry, so reloading auth.json keeps the hour."""
    pool = _load(
        tmp_path,
        monkeypatch,
        [
            _entry(403, age_seconds=10, cred_id="cred-1", failure_reason="billing"),
            _entry(403, age_seconds=10, cred_id="cred-2", failure_reason="billing"),
        ],
    )
    assert pool.entries()[0].failure_reason == "billing"
    next_at = pool.next_available_at()
    assert next_at is not None
    assert next_at - time.time() > 3400


def test_auth_403_survives_reload(tmp_path, monkeypatch):
    """A bare 403 reloads as a bare 403 — the short bench is derived, not lost."""
    pool = _load(
        tmp_path,
        monkeypatch,
        [
            _entry(403, age_seconds=10, cred_id="cred-1"),
            _entry(403, age_seconds=10, cred_id="cred-2"),
        ],
    )
    next_at = pool.next_available_at()
    assert next_at is not None
    remaining = next_at - time.time()
    assert 60 < remaining < 600, f"expected the short bench after reload, got {remaining:.0f}s"
