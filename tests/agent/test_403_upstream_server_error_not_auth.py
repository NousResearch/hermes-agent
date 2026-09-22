"""t_6ecd37ae: a 403 that re-stamps an *upstream* server failure is not an auth verdict.

Measured live (2026-09-22 17:54:12 EDT, home ``profiles/mem0-bot``): an OpenCode Go
account served HTTP 200 before and after, its ``/usage`` meter read 0% on every
window, and the account was healthy — but one call came back as::

    403 {'error': {'type': 'server_error', 'code': 'server_error',
                   'message': 'Upstream request failed: [server_error] Upstream respons...'}}

``_status_403`` had no branch for that shape, so it fell through to the auth
default: ``PermissionDeniedError`` → ``FailoverReason.auth`` →
``_recover_auth_failure`` → ``rotate_and_swap`` → a flat
``_exhausted_ttl(403, "auth")`` one-hour bench with ``last_error_reset_at=None``.
The sibling account was already benched for its weekly cap, so the pool went
empty and the home ran 17 metered Fireworks calls before the bench was cleared
by hand.

These tests pin both halves: the server-failure family is transient (retried on
the same credential, no bench), while a genuine refusal still benches exactly as
before — and a transient failure on one account leaves the other account's
pre-emptive window bench (written by ``~/.hermes/opencode-go/go_window_watch.py``)
intact instead of emptying the pool.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.error_classifier import FailoverReason, classify_api_error

PROVIDER = "opencode-go"
BASE_URL = "https://opencode.ai/zen/go/v1"

# The exact body from the live incident, verbatim.
LIVE_BODY = {
    "error": {
        "type": "server_error",
        "code": "server_error",
        "message": "Upstream request failed: [server_error] Upstream response error",
    }
}


class MockAPIError(Exception):
    """The status/body surface every SDK exception exposes (openai, litellm)."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _err(status_code=403, body=None, message=None):
    body = LIVE_BODY if body is None else body
    return MockAPIError(message or f"Error code: {status_code} - {json.dumps(body)}",
                        status_code=status_code, body=body)


# ── Classification ───────────────────────────────────────────────────────────


class TestServerFailure403IsNotAuth:
    def test_live_incident_body_is_not_an_auth_verdict(self):
        """The exact measured body: transient, retried on the same key, no bench."""
        result = classify_api_error(_err(403, LIVE_BODY), provider=PROVIDER, model="gpt-5.3-codex")
        assert result.reason != FailoverReason.auth
        assert result.should_rotate_credential is False
        assert result.retryable is True

    @pytest.mark.parametrize(
        "body",
        [
            # code and type both stamped, as the gateway does.
            {"error": {"type": "server_error", "code": "server_error", "message": "Upstream request failed"}},
            # type only — ``_extract_error_code`` reads ``code`` then ``type``.
            {"error": {"type": "server_error", "message": "Upstream request failed"}},
            # sibling server-failure codes from the same family.
            {"error": {"code": "internal_error", "message": "internal"}},
            {"error": {"code": "upstream_error", "message": "upstream"}},
            {"error": {"code": "upstream_service_error", "message": "upstream"}},
            # prose only: a gateway that drops the structured code entirely.
            {"error": {"message": "Upstream request failed: [server_error] (no code field)"}},
            {"error": {"message": "Upstream response error"}},
        ],
    )
    def test_server_failure_family_is_transient(self, body):
        result = classify_api_error(_err(403, body), provider=PROVIDER, model="gpt-5.3-codex")
        assert result.reason == FailoverReason.server_error
        assert result.should_rotate_credential is False
        assert result.retryable is True

    def test_existing_upstream_unavailable_allowlist_unchanged(self):
        """The pre-existing 403 transient allowlist keeps its own verdict (#75388)."""
        body = {"error": {"type": "upstream_unavailable", "code": "upstream_unavailable",
                          "message": "Upstream service temporarily unavailable."}}
        result = classify_api_error(_err(403, body), provider="custom")
        assert result.reason == FailoverReason.overloaded
        assert result.should_rotate_credential is False


class TestGenuineRefusalsStillAuth:
    """Constraint: real auth handling must not be weakened by the fix above."""

    @pytest.mark.parametrize(
        "body",
        [
            {"error": {"message": "Forbidden"}},
            {"error": {"type": "invalid_request_error", "code": "invalid_api_key",
                       "message": "Incorrect API key provided"}},
            {"error": {"type": "permission_denied", "code": "permission_denied",
                       "message": "You do not have access to this resource"}},
        ],
    )
    def test_refusal_shaped_403_stays_auth(self, body):
        result = classify_api_error(_err(403, body), provider=PROVIDER)
        assert result.reason == FailoverReason.auth

    def test_401_invalid_key_is_still_auth_and_rotates(self):
        body = {"error": {"type": "invalid_request_error", "code": "invalid_api_key",
                          "message": "Incorrect API key provided"}}
        result = classify_api_error(_err(401, body), provider=PROVIDER)
        assert result.reason == FailoverReason.auth
        assert result.should_rotate_credential is True

    @pytest.mark.parametrize(
        "message",
        ["Key limit exceeded for this key", "Your spending limit has been reached"],
    )
    def test_billing_403_still_billing(self, message):
        result = classify_api_error(_err(403, {"error": {"message": message}}), provider=PROVIDER)
        assert result.reason == FailoverReason.billing

    def test_server_failure_does_not_shadow_billing_body(self):
        """A billing body whose prose mentions the word "server" is still billing."""
        body = {"error": {"code": "insufficient_quota",
                          "message": "You exceeded your current quota on the server."}}
        result = classify_api_error(_err(403, body), provider=PROVIDER)
        assert result.reason == FailoverReason.billing


# ── End-to-end: the pool must stay serving ───────────────────────────────────


def _entry(cred_id, *, token, priority, base_url=BASE_URL):
    return {
        "id": cred_id,
        "label": cred_id,
        "auth_type": "api_key",
        "priority": priority,
        "source": "manual",
        "access_token": token,
        "base_url": base_url,
    }


def _load_pool(tmp_path, monkeypatch, entries):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {PROVIDER: entries}}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # ``_env_key_var_candidates`` ingests ``OPENCODE_GO_API_KEY`` and its numbered siblings
    # (``_2``, ``_3`` …) from the environment, which would add a third serving credential and
    # make every "the pool is empty" assertion here vacuous on a host that exports them.
    for name in ["OPENCODE_GO_API_KEY"] + [f"OPENCODE_GO_API_KEY_{n}" for n in range(2, 10)]:
        monkeypatch.delenv(name, raising=False)
    from agent.credential_pool import load_pool

    pool = load_pool(PROVIDER)
    assert sorted(e.id for e in pool.entries()) == sorted(e["id"] for e in entries), (
        "fixture leak: the pool was seeded from the environment, not just from the written rows"
    )
    return pool


def _bench_by_watcher(pool, cred_id, *, hours=95.5):
    """The bench ``go_window_watch.py`` writes: 429, rate_limit, ISO reset at the window end.

    Written straight to the row, exactly as the watcher does — it never goes through
    the pool's own exhaustion path.
    """
    reset_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + hours * 3600))
    import os
    path = os.path.join(os.environ["HERMES_HOME"], "auth.json")
    store = json.loads(open(path).read())
    row = next(r for r in store["credential_pool"][PROVIDER] if r["id"] == cred_id)
    row.update({
        "last_status": "exhausted",
        "last_status_at": time.time(),
        "last_error_code": 429,
        "last_error_reason": "usage_limit_reached",
        "last_error_message": "pre-emptive switch: weekly window at 94% (watcher)",
        "last_error_reset_at": reset_iso,
        "failure_reason": "rate_limit",
    })
    open(path, "w").write(json.dumps(store, indent=1))
    from agent.credential_pool import load_pool

    pool = load_pool(PROVIDER)
    assert len(pool.entries()) == len(store["credential_pool"][PROVIDER])
    return pool, reset_iso


def _recover(pool, *, reason, failing_id, failing_key, status_code=403):
    """Run the failing call's recovery path exactly as the turn loop does."""
    from agent.agent_runtime_helpers import recover_with_credential_pool

    agent = SimpleNamespace(
        provider=PROVIDER,
        base_url=BASE_URL,
        model="gpt-5.3-codex",
        api_key=failing_key,
        _credential_pool=pool,
        _credential_pool_entry_id=failing_id,
        _credential_pool_revert_id=None,
        _swap_credential=MagicMock(return_value=True),
        _is_entitlement_failure=lambda error_context, status_code: False,
        _auth_pool_refresh_counts=None,
    )
    return recover_with_credential_pool(
        agent, status_code=status_code, has_retried_429=False, classified_reason=reason,
    )


def test_transient_403_does_not_empty_a_pool_whose_sibling_is_window_benched(tmp_path, monkeypatch):
    """The incident, end to end.

    Account A is window-benched by the watcher; the live 403 hits healthy
    account B. B must stay serving: the pool must not end up with nothing to
    serve and drop the home onto metered inference.
    """
    pool = _load_pool(tmp_path, monkeypatch,
                      [_entry("acct-a", token="key-a", priority=0),
                       _entry("acct-b", token="key-b", priority=1)])
    pool, watcher_reset = _bench_by_watcher(pool, "acct-a")

    verdict = classify_api_error(
        _err(403, LIVE_BODY), provider=PROVIDER, model="gpt-5.3-codex", base_url=BASE_URL
    )
    assert verdict.reason == FailoverReason.server_error, "guard: pre-fix this is auth"

    _recover(pool, reason=verdict.reason, failing_id="acct-b", failing_key="key-b")

    assert pool.has_available() is True, "the pool must still have something to serve"
    entry = pool.select()
    assert entry is not None and entry.id == "acct-b"
    # B carries no bench at all — pre-fix it got failure_reason=auth and a one-hour TTL
    # with last_error_reset_at=None.
    benched = next(e for e in pool.entries() if e.id == "acct-b")
    assert benched.last_status != "exhausted", (
        f"acct-b was benched ({benched.last_status}, reason={benched.last_error_reason!r}, "
        f"failure_reason={benched.failure_reason!r})"
    )
    assert benched.failure_reason != "auth"


def test_transient_403_leaves_the_watchers_window_bench_untouched(tmp_path, monkeypatch):
    """The two bench-writing paths must be provably independent.

    The watcher writes only ``auth.json`` rows (never ``.env``); the pool writes
    only the row that failed. A transient failure on B must not shorten, clear or
    re-stamp A's window bench.
    """
    pool = _load_pool(tmp_path, monkeypatch,
                      [_entry("acct-a", token="key-a", priority=0),
                       _entry("acct-b", token="key-b", priority=1)])
    pool, watcher_reset = _bench_by_watcher(pool, "acct-a")
    before = _read_row("acct-a")

    verdict = classify_api_error(_err(403, LIVE_BODY), provider=PROVIDER, base_url=BASE_URL)
    _recover(pool, reason=verdict.reason, failing_id="acct-b", failing_key="key-b")

    a_after = next(e for e in pool.entries() if e.id == "acct-a")
    after = _read_row("acct-a")
    # Only the fields that carry the bench matter here: when the pool path persists at all it
    # also normalizes unrelated bookkeeping (``request_count``).
    bench_fields = ("last_status", "last_error_code", "last_error_reason",
                    "last_error_reset_at", "failure_reason", "last_error_message")
    assert {k: before.get(k) for k in bench_fields} == {k: after.get(k) for k in bench_fields}, (
        "the watcher's bench was re-stamped by the pool path"
    )
    assert after["last_status"] == "exhausted"
    assert after["failure_reason"] == "rate_limit"
    assert after["last_error_reset_at"] == watcher_reset
    assert a_after.last_error_reset_at == watcher_reset
    # A's bench is still honoured on the selection path (reset_at wins over TTL).
    assert _exhausted_until_seconds(pool, "acct-a") > 3600
    assert pool.has_available() is True  # only because B is still serving


def test_genuine_auth_403_still_benches_beside_a_window_benched_sibling(tmp_path, monkeypatch):
    """Constraint, and the proof that the test above really drives the pool.

    A *genuine* refusal on B must still take B out of rotation, even though that
    leaves the pool empty until A's window reopens — that is what a bad key
    requires. The bug was the classification, not the auth bench.
    """
    pool = _load_pool(tmp_path, monkeypatch,
                      [_entry("acct-a", token="key-a", priority=0),
                       _entry("acct-b", token="key-b", priority=1)])
    pool, _ = _bench_by_watcher(pool, "acct-a")

    _recover(pool, reason=FailoverReason.auth, failing_id="acct-b", failing_key="key-b")

    b = next(e for e in pool.entries() if e.id == "acct-b")
    assert b.last_status == "exhausted"
    assert b.failure_reason == "auth"
    assert b.last_error_reset_at is None
    assert pool.has_available() is False


def _read_row(cred_id):
    import os
    path = os.path.join(os.environ["HERMES_HOME"], "auth.json")
    store = json.loads(open(path).read())
    return next(r for r in store["credential_pool"][PROVIDER] if r["id"] == cred_id)


def _exhausted_until_seconds(pool, cred_id):
    from agent.credential_pool import _exhausted_until

    entry = next(e for e in pool.entries() if e.id == cred_id)
    until = _exhausted_until(entry)
    assert until is not None
    return until - time.time()


def test_sole_credential_403_billing_still_keeps_the_full_bench(tmp_path, monkeypatch):
    """Constraint: the fix must not turn a genuine billing 403 into a 60s retry loop."""
    from agent.credential_pool import _exhausted_ttl

    assert _exhausted_ttl(403, sole_credential=True, failure_reason="billing") == 3600
    assert _exhausted_ttl(403, sole_credential=True) == 60  # unchanged transient rule
