"""Sole-credential cooldown: a pool with nothing to rotate to should not bench
its only key for an hour on a transient throttle (429/403/5xx).

Regression for the case where removing fallbacks / running a single API key
turned a transient rate-limit into an hour of hard failures.
"""

from __future__ import annotations

import json
import time

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


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
    _write_auth_store(
        tmp_path,
        {"version": 1, "credential_pool": {"openrouter": entries}},
    )
    from agent.credential_pool import load_pool

    return load_pool("openrouter")


def test_sole_credential_429_recovers_after_short_cooldown(tmp_path, monkeypatch):
    """A single 429-throttled key recovers within ~1 min, not 1 hour.

    Exhausted 90s ago: under the old 1-hour TTL this stays benched (and a
    single-key pool would have nothing to return); the sole-credential short
    cooldown lets it recover.
    """
    pool = _load(tmp_path, monkeypatch, [_entry(429, age_seconds=90)])
    entry = pool.select()
    assert entry is not None
    assert entry.id == "cred-1"
    assert entry.last_status == "ok"


def test_sole_credential_403_recovers_after_short_cooldown(tmp_path, monkeypatch):
    """403 (edge-throttle variant, hits the catch-all default TTL) also recovers."""
    pool = _load(tmp_path, monkeypatch, [_entry(403, age_seconds=90)])
    entry = pool.select()
    assert entry is not None
    assert entry.last_status == "ok"


def test_sole_credential_billing_403_keeps_full_bench(tmp_path, monkeypatch):
    """A 403 classified as BILLING must keep the full bench, not the 60s cooldown.

    Providers overload 403: OpenRouter returns it for `key limit exceeded` and
    xAI for a spending-limit block, both of which `error_classifier` maps to
    FailoverReason.billing. Status alone can't tell those from an edge
    throttle, so retrying a spent account every 60s just re-fails forever.
    The classified reason rides along on the entry and wins over the status.
    """
    pool = _load(
        tmp_path,
        monkeypatch,
        [_entry(403, age_seconds=90, failure_reason="billing")],
    )
    assert pool.has_available() is False
    assert pool.select() is None


def test_sole_credential_billing_403_survives_reload(tmp_path, monkeypatch):
    """The classified reason persists, so a restart can't downgrade the bench.

    `failure_reason` is written to auth.json with the entry; without that, a
    process restart would re-read a bare 403 and hand the spent key back after
    60 seconds.
    """
    from agent.credential_pool import _exhausted_ttl

    pool = _load(
        tmp_path,
        monkeypatch,
        [_entry(403, age_seconds=90, failure_reason="billing")],
    )
    entry = pool.entries()[0]
    assert entry.failure_reason == "billing"
    assert _exhausted_ttl(403, sole_credential=True, failure_reason="billing") == 60 * 60
    assert _exhausted_ttl(403, sole_credential=True) == 60


def test_sole_credential_402_keeps_full_bench(tmp_path, monkeypatch):
    """402 (billing/quota) is genuine exhaustion — a quick retry can't help, so
    the sole-credential short cooldown must NOT apply."""
    pool = _load(tmp_path, monkeypatch, [_entry(402, age_seconds=90)])
    assert pool.has_available() is False
    assert pool.select() is None


def test_sole_credential_next_available_at_uses_short_cooldown(tmp_path, monkeypatch):
    """next_available_at must also honour the sole-credential short cooldown.

    Without this, the fallback restore gate in agent_runtime_helpers waits an
    hour for a 60s cooldown, keeping the agent on a fallback provider far
    longer than necessary.
    """
    from agent.credential_pool import EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS

    pool = _load(tmp_path, monkeypatch, [_entry(429, age_seconds=10)])
    next_at = pool.next_available_at()
    assert next_at is not None
    # Should be ~60s from exhaustion, not ~3600s.  The entry was exhausted 10s
    # ago, so the remaining wait is ~50s.
    remaining = next_at - time.time()
    assert remaining < EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS, (
        f"next_available_at returned {remaining:.0f}s remaining — expected < "
        f"{EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS}s (sole-credential cooldown)"
    )
    assert remaining < 300, (
        f"next_available_at returned {remaining:.0f}s — should be seconds, not hours"
    )


def test_multi_key_429_keeps_full_bench(tmp_path, monkeypatch):
    """With more than one non-DEAD entry there IS something to rotate to, so the
    short cooldown must not kick in — both recently-throttled keys stay benched."""
    pool = _load(
        tmp_path,
        monkeypatch,
        [
            _entry(429, age_seconds=90, cred_id="cred-1", priority=0),
            _entry(429, age_seconds=90, cred_id="cred-2", priority=1),
        ],
    )
    assert pool.has_available() is False
    assert pool.select() is None
