"""A terminally rejected OAuth refresh token must leave a trace at the default log level.

The pool quarantines a dead ``openai-codex`` / ``xai-oauth`` / ``nous`` / ``anthropic`` refresh
token — for the user this is the moment the login is lost, and a debug-only line made it look like
"I signed in once and Hermes keeps failing" (#113023). Two invariants: the WARNING carries the
``hermes auth add <provider>`` hint, and a row the quarantine does not drop (an independent
``manual:*`` login) is marked DEAD so it leaves rotation instead of re-firing the WARNING on every
later refresh attempt.
"""
from __future__ import annotations

import logging
import threading

import pytest

from agent import anthropic_credentials as ac
from agent import credential_pool as cp
from agent.credential_pool import STATUS_DEAD, CredentialPool, PooledCredential


def _pool(provider: str) -> CredentialPool:
    pool = CredentialPool.__new__(CredentialPool)
    pool._lock = threading.RLock()
    pool._entries = []
    pool._active_leases = {}
    pool._current_id = None
    pool._max_concurrent = 2
    pool._unmatched_rotation_streak = 0
    pool.provider = provider
    return pool


def _entry(provider: str, source: str = "device_code") -> PooledCredential:
    return PooledCredential(id="e1", provider=provider, auth_type="oauth", access_token="dead-access",
                            refresh_token="dead-refresh", label="e1", source=source, priority=0)


@pytest.mark.parametrize(
    ("provider", "terminal_predicate", "sync_name", "clear_name", "expected_hint"),
    [
        ("openai-codex", "_is_terminal_codex_oauth_refresh_error", "_sync_entry_from_auth_store",
         "_clear_terminal_tokens_state", "hermes auth add openai-codex"),
        ("nous", "_is_terminal_nous_refresh_error", "_sync_nous_entry_from_auth_store",
         "_clear_terminal_nous_state", "hermes auth add nous"),
    ],
)
def test_terminal_refresh_quarantine_warns_with_reauth_hint(
    monkeypatch, caplog, provider, terminal_predicate, sync_name, clear_name, expected_hint,
):
    pool = _pool(provider)
    entry = _entry(provider)
    pool._entries = [entry]
    cleared: list = []
    monkeypatch.setattr(pool, sync_name, lambda e: e)  # no peer rotated in the meantime
    monkeypatch.setattr(pool, clear_name, lambda e, exc: cleared.append(e.id))
    monkeypatch.setattr(pool, "_quarantine_sources", lambda e, sources: None)
    monkeypatch.setattr(cp.auth_mod, terminal_predicate, lambda exc: True)

    with caplog.at_level(logging.INFO, logger=cp.logger.name):
        result = pool._recover_failed_refresh(entry, RuntimeError("invalid_grant"))

    assert result is None and cleared == ["e1"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "terminally invalid" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert expected_hint in warnings[0].getMessage()
    assert "invalid_grant" in warnings[0].getMessage()


def test_anthropic_dead_grant_warns_and_marks_dead(monkeypatch, caplog):
    """A dead Anthropic grant is not a transient 'exhausted': WARNING with the re-auth hint, row DEAD."""
    pool = _pool("anthropic")
    entry = _entry("anthropic", source="manual:hermes_pkce")
    pool._entries = [entry]
    monkeypatch.setattr(pool, "_sync_entry_from_pool_store", lambda e: e)
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)
    exc = ac.AnthropicOAuthError(400, "invalid_grant", "refresh token revoked", what="refresh")

    with caplog.at_level(logging.INFO, logger=cp.logger.name):
        result = pool._recover_failed_refresh(entry, exc)

    assert result is None
    row = pool._entries[0]
    assert row.last_status == STATUS_DEAD and row.last_error_reason == "invalid_grant"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "terminally invalid" in r.getMessage()]
    assert len(warnings) == 1 and "hermes auth add anthropic" in warnings[0].getMessage()


def test_anthropic_transient_refresh_failure_stays_exhausted(monkeypatch, caplog):
    """Control: a network-shaped failure is still benched as transient, silently."""
    pool = _pool("anthropic")
    entry = _entry("anthropic", source="manual:hermes_pkce")
    pool._entries = [entry]
    monkeypatch.setattr(pool, "_sync_entry_from_pool_store", lambda e: e)
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)

    with caplog.at_level(logging.INFO, logger=cp.logger.name):
        assert pool._recover_failed_refresh(entry, TimeoutError("token endpoint timed out")) is None

    assert pool._entries[0].last_status == cp.STATUS_EXHAUSTED
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_surviving_manual_entry_is_marked_dead_after_terminal_refresh(monkeypatch):
    """An independent ``manual:device_code`` login survives the singleton quarantine; it must leave
    rotation as DEAD rather than sit unmarked and re-fire the WARNING on the next refresh."""
    pool = _pool("openai-codex")
    entry = _entry("openai-codex", source="manual:device_code")
    pool._entries = [entry]
    monkeypatch.setattr(pool, "_sync_entry_from_auth_store", lambda e: e)
    monkeypatch.setattr(pool, "_clear_terminal_tokens_state", lambda e, exc: None)
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)
    monkeypatch.setattr(cp.auth_mod, "_is_terminal_codex_oauth_refresh_error", lambda exc: True)

    assert pool._recover_failed_refresh(entry, RuntimeError("invalid_grant")) is None

    assert [e.id for e in pool._entries] == ["e1"]  # manual rows are never dropped by the quarantine
    assert pool._entries[0].last_status == STATUS_DEAD
