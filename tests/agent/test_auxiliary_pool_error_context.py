# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""Auxiliary pool recovery must bench a credential for the window the provider reported.

The aux ladder is the only caller of ``mark_exhausted_and_rotate`` that used to hand it an
``error_context`` built from ``str(exc)`` alone. The pool then has nothing to size the bench from:
``_normalize_error_context`` finds no ``reset_at``, ``last_error_reset_at`` stays ``None`` and
``_exhausted_until`` falls back to the blind ``EXHAUSTED_TTL_429_SECONDS`` hour. A burst 429 that
the provider said to retry in 30 s therefore parked a healthy credential for an hour, and a couple
of concurrent aux calls drained the whole pool. The main loop has always parsed the response
through ``extract_api_error_context``; the aux path must reach the same window.
"""

import time
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as ac
from agent.credential_pool import EXHAUSTED_TTL_429_SECONDS, CredentialPool, PooledCredential

_BASE = "https://api.anthropic.com"

_RATE_LIMIT_BODY = "Error code: 429 - {'error': {'message': 'Rate limit exceeded', 'code': 429}}"


def _err(status, message, headers=None, body=None):
    """Provider error shaped like the SDK's: a status code plus the HTTP response."""
    exc = Exception(message)
    exc.status_code = status
    exc.response = SimpleNamespace(headers=dict(headers or {}))
    if body is not None:
        exc.body = body
    return exc


class _StubPool:
    """Captures what the ladder hands the pool, so the contract is read off the real call."""

    def __init__(self, next_entry=object()):
        self.rotate_calls = []
        self._next = next_entry

    def has_credentials(self):
        return True

    def try_refresh_current(self):
        return None

    def mark_exhausted_and_rotate(self, **kwargs):
        self.rotate_calls.append(kwargs)
        return self._next


def _recover(monkeypatch, exc, provider="anthropic", failed_api_key="sk-failed"):
    pool = _StubPool()
    monkeypatch.setattr(ac, "load_pool", lambda _provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda _provider: None)
    recovered = ac._recover_provider_pool(provider, exc, failed_api_key=failed_api_key)
    return recovered, pool


def test_retry_after_header_sizes_the_bench_instead_of_the_blind_hour(monkeypatch):
    now = time.time()
    recovered, pool = _recover(monkeypatch, _err(429, _RATE_LIMIT_BODY, headers={"retry-after": "30"}))

    assert recovered is True
    assert len(pool.rotate_calls) == 1
    context = pool.rotate_calls[0]["error_context"]
    assert context["reset_at"] == pytest.approx(now + 30, abs=5)
    # A 30 s window must never be rounded up to the full-hour fallback.
    assert context["reset_at"] < now + EXHAUSTED_TTL_429_SECONDS


def test_vendor_bucket_header_reaches_the_pool_window(monkeypatch):
    now = time.time()
    headers = {"anthropic-ratelimit-requests-reset": "45s"}
    recovered, pool = _recover(monkeypatch, _err(429, _RATE_LIMIT_BODY, headers=headers))

    assert recovered is True
    reset_at = pool.rotate_calls[0]["error_context"]["reset_at"]
    assert reset_at == pytest.approx(now + 45, abs=5)


def test_window_carrying_body_survives_the_handoff(monkeypatch):
    """A reset stated in the payload is as good as one in the headers."""
    now = time.time()
    body = {"error": {"message": "Rate limit exceeded", "code": 429, "resets_in_seconds": 120}}
    recovered, pool = _recover(monkeypatch, _err(429, _RATE_LIMIT_BODY, body=body))

    assert recovered is True
    assert pool.rotate_calls[0]["error_context"]["reset_at"] == pytest.approx(now + 120, abs=5)


def test_message_still_carries_through_when_headers_say_nothing(monkeypatch):
    recovered, pool = _recover(monkeypatch, _err(429, "Error code: 429 - rate limited, retry after 20s"))

    assert recovered is True
    context = pool.rotate_calls[0]["error_context"]
    assert "retry after 20s" in context["message"]
    # The message was always parsed downstream; the header path must not have displaced it.
    assert context["reset_at"] == pytest.approx(time.time() + 20, abs=5)


def test_benches_without_any_window_still_rotate_and_keep_the_status(monkeypatch):
    """No reported window is not a reason to stop rotating — the blind TTL remains the floor."""
    recovered, pool = _recover(monkeypatch, _err(429, "Error code: 429 - too many requests"))

    assert recovered is True
    call = pool.rotate_calls[0]
    assert call["status_code"] == 429
    assert call["api_key_hint"] == "sk-failed"
    assert "reset_at" not in call["error_context"]
    assert call["error_context"]["status_code"] == 429
    assert "too many requests" in call["error_context"]["message"]


def test_payment_and_auth_branches_keep_their_fallback_status(monkeypatch):
    # 402: the pool is handed the fallback code when the exception carries none.
    _, pool = _recover(monkeypatch, _err(402, "Error code: 402 - insufficient credits"))
    assert pool.rotate_calls[0]["status_code"] == 402
    assert pool.rotate_calls[0]["error_context"]["status_code"] == 402

    # 401: the ladder only gets here once a refresh attempt already failed.
    _, pool = _recover(monkeypatch, _err(401, "Error code: 401 - invalid x-api-key"))
    assert pool.rotate_calls[0]["status_code"] == 401
    assert pool.rotate_calls[0]["error_context"]["status_code"] == 401


def test_pool_entry_lands_with_the_reported_window_not_the_hour(monkeypatch):
    """End to end: the bench the pool derives is the provider's, measured on a real pool entry."""
    pool = CredentialPool(provider="anthropic", entries=[
        PooledCredential.from_dict("anthropic", {
            "id": "pref0000", "label": "subscription", "auth_type": "api_key",
            "priority": 0, "access_token": "sk-primary", "base_url": _BASE, "source": "manual",
        }),
        PooledCredential.from_dict("anthropic", {
            "id": "spare00", "label": "spare", "auth_type": "api_key",
            "priority": 1, "access_token": "sk-spare", "base_url": _BASE, "source": "manual",
        }),
    ])
    monkeypatch.setattr(ac, "load_pool", lambda _provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda _provider: None)

    now = time.time()
    exc = _err(429, _RATE_LIMIT_BODY, headers={"retry-after": "30"})
    assert ac._recover_provider_pool("anthropic", exc, failed_api_key="sk-primary") is True

    benched = next(e for e in pool.entries() if e.id == "pref0000")
    assert benched.last_status == "exhausted"
    assert benched.last_error_reset_at == pytest.approx(now + 30, abs=5)
    # The rotation landed on the healthy spare, not back onto the entry it just benched.
    assert pool.current() is not None and pool.current().id == "spare00"


def _pool_entry(provider="anthropic", token="sk-primary"):
    """A two-entry pool so the rotation has somewhere healthy to land."""
    return CredentialPool(provider=provider, entries=[
        PooledCredential.from_dict(provider, {
            "id": "pref0000", "label": "subscription", "auth_type": "api_key",
            "priority": 0, "access_token": token, "base_url": _BASE, "source": "manual",
        }),
        PooledCredential.from_dict(provider, {
            "id": "spare00", "label": "spare", "auth_type": "api_key",
            "priority": 1, "access_token": "sk-spare", "base_url": _BASE, "source": "manual",
        }),
    ])


def test_generic_body_message_does_not_hide_a_window_stated_beside_it(monkeypatch):
    """A body whose ``error.message`` is generic still carries the wait in its full text."""
    now = time.time()
    body = {"error": {"message": "Rate limit exceeded", "code": 429,
                      "detail": "quota window exhausted, please retry after 20s"}}
    pool = _pool_entry()
    monkeypatch.setattr(ac, "load_pool", lambda _provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda _provider: None)

    assert ac._recover_provider_pool("anthropic", _err(429, "Error code: 429 - " + repr(body), body=body),
                                     failed_api_key="sk-primary") is True

    benched = next(e for e in pool.entries() if e.id == "pref0000")
    assert benched.last_status == "exhausted"
    assert benched.last_error_reset_at == pytest.approx(now + 20, abs=5)
    rotated_to = pool.current()
    assert rotated_to is not None
    assert rotated_to.id == "spare00"


def test_window_past_the_five_hundred_char_truncation_still_sizes_the_bench(monkeypatch):
    """Providers pad the body; the window sits past the extractor's truncation."""
    now = time.time()
    pool = _pool_entry()
    monkeypatch.setattr(ac, "load_pool", lambda _provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda _provider: None)

    exc = _err(429, "quota exhausted. " * 40 + "please retry after 90s")
    assert ac._recover_provider_pool("anthropic", exc, failed_api_key="sk-primary") is True

    benched = next(e for e in pool.entries() if e.id == "pref0000")
    assert benched.last_error_reset_at == pytest.approx(now + 90, abs=5)
