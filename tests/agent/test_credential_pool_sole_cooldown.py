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


def _entry(error_code: int, *, age_seconds: float, cred_id: str = "cred-1", priority: int = 0) -> dict:
    return {
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


def test_sole_credential_402_keeps_full_bench(tmp_path, monkeypatch):
    """402 (billing/quota) is genuine exhaustion — a quick retry can't help, so
    the sole-credential short cooldown must NOT apply."""
    pool = _load(tmp_path, monkeypatch, [_entry(402, age_seconds=90)])
    assert pool.has_available() is False
    assert pool.select() is None


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
