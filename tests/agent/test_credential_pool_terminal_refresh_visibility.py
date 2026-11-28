"""A terminally rejected OAuth refresh token must leave a trace at the default log level.

The pool quarantines a dead ``openai-codex`` / ``xai-oauth`` / ``nous`` / ``anthropic`` refresh
token — for the user this is the moment the login is lost, and a debug-only line made it look like
"I signed in once and Hermes keeps failing" (#113023). Two invariants: the WARNING carries the
``hermes auth add <provider>`` hint, and a row the quarantine does not drop (an independent
``manual:*`` login) is marked DEAD so it leaves rotation instead of re-firing the WARNING on every
later refresh attempt.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from types import SimpleNamespace

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
    pool._persisted_token_pairs = {}
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


def _codex_refresh_error(status: int, body):
    """The real Codex refresh error for a token-endpoint reply (``body=None``: a non-JSON page)."""
    from hermes_cli.auth_codex import _codex_refresh_failure_error

    def _json():
        if body is None:
            raise ValueError("not JSON")
        return body

    return _codex_refresh_failure_error(SimpleNamespace(status_code=status, json=_json))


def test_surviving_manual_entry_is_marked_dead_after_terminal_refresh(monkeypatch):
    """An independent ``manual:device_code`` login survives the singleton quarantine; it must leave
    rotation as DEAD rather than sit unmarked and re-fire the WARNING on the next refresh."""
    pool = _pool("openai-codex")
    entry = _entry("openai-codex", source="manual:device_code")
    pool._entries = [entry]
    monkeypatch.setattr(pool, "_sync_entry_from_auth_store", lambda e: e)
    monkeypatch.setattr(pool, "_clear_terminal_tokens_state", lambda e, exc: None)
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)

    assert pool._recover_failed_refresh(entry, _codex_refresh_error(400, {"error": "invalid_grant"})) is None

    assert [e.id for e in pool._entries] == ["e1"]  # manual rows are never dropped by the quarantine
    assert pool._entries[0].last_status == STATUS_DEAD


@pytest.mark.parametrize(
    ("status", "body", "dead"),
    [
        (401, {"error": {"code": "refresh_token_expired", "message": "Your refresh token has expired."}}, True),
        (401, {"error": {"code": "refresh_token_invalidated", "message": "Your refresh token was revoked."}}, True),
        (401, {"error": {"type": "invalid_request_error", "message": "Unauthorized."}}, True),
        (400, {"code": "refresh_token_expired"}, True),
        (403, {"error": {"type": "invalid_request_error", "message": "Forbidden."}}, False),
        (403, None, False),  # an HTML challenge page from the edge, not a verdict on the token
        (500, {"error": {"code": "server_error", "message": "Try again later."}}, False),
    ],
)
def test_codex_refresh_rejection_is_terminal_exactly_when_the_login_is_dead(monkeypatch, caplog, status, body, dead):
    """Dead by OpenAI's own Codex CLI rule (any 401, or an expired/reused/invalidated code at any
    status) means DEAD with the re-auth WARNING, whatever ``code`` OpenAI spelled it with; anything
    else is benched as transient, silently. The real error builder and the real terminal predicate
    decide; neither is patched."""
    exc = _codex_refresh_error(status, body)
    pool = _pool("openai-codex")
    entry = _entry("openai-codex", source="manual:device_code")
    pool._entries = [entry]
    monkeypatch.setattr(pool, "_sync_entry_from_auth_store", lambda e: e)
    monkeypatch.setattr(pool, "_clear_terminal_tokens_state", lambda e, exc: None)
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)

    with caplog.at_level(logging.INFO, logger=cp.logger.name):
        assert pool._recover_failed_refresh(entry, exc) is None

    assert exc.relogin_required is dead
    assert pool._entries[0].last_status == (STATUS_DEAD if dead else cp.STATUS_EXHAUSTED)
    warned = [r for r in caplog.records if r.levelno == logging.WARNING and "hermes auth add openai-codex" in r.getMessage()]
    assert bool(warned) is dead


def test_expired_codex_login_is_cleared_from_auth_json_with_its_reason(tmp_path, monkeypatch):
    """End to end against a temp HERMES_HOME: only the HTTP reply is stubbed. An expired singleton
    login leaves the pool, its token pair leaves auth.json, and the stored error says why."""
    from agent.credential_pool import load_pool

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    tokens = {"access_token": "dead-access", "refresh_token": "dead-refresh"}
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {"openai-codex": {"tokens": dict(tokens), "auth_mode": "chatgpt"}},
        "credential_pool": {"openai-codex": [
            {"id": "seeded", "label": "device_code", "auth_type": "oauth", "priority": 0,
             "source": "device_code", **tokens}]},
    }), encoding="utf-8")
    expired = {"error": {"code": "refresh_token_expired", "message": "Your refresh token has expired."}}

    def _rejected(access_token, refresh_token):
        raise _codex_refresh_error(401, expired)

    monkeypatch.setattr(cp.auth_mod, "refresh_codex_oauth_pure", _rejected)
    pool = load_pool("openai-codex")

    assert pool.try_refresh_matching(credential_id="seeded") is None

    state = json.loads((home / "auth.json").read_text(encoding="utf-8"))["providers"]["openai-codex"]
    assert "refresh_token" not in state["tokens"] and "access_token" not in state["tokens"]
    assert state["last_auth_error"]["code"] == "codex_refresh_failed"
    assert "refresh_token_expired" in state["last_auth_error"]["message"]
    assert all(e.id != "seeded" or e.last_status == STATUS_DEAD for e in load_pool("openai-codex").entries())


def _expired_invoke_jwt() -> str:
    def _part(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{_part({'alg': 'none'})}.{_part({'sub': 'u', 'scope': 'inference:invoke', 'exp': int(time.time()) - 60})}.sig"


@pytest.mark.parametrize(
    ("nous_state", "expected_code"),
    [
        (None, "nous_auth_missing"),
        ({"client_id": "hermes-cli", "scope": "inference:invoke", "access_token": _expired_invoke_jwt(),
          "refresh_token": "", "expires_at": "2026-02-01T00:00:00+00:00"}, "nous_auth_missing_refresh_token"),
    ],
    ids=["not_logged_in", "expired_jwt_without_refresh_token"],
)
def test_nous_login_missing_refresh_failure_is_terminal(tmp_path, monkeypatch, caplog, nous_state, expected_code):
    """A "needs a login" raise from the real resolver — no Portal login at all, or an unusable
    access token with no refresh token to redeem — is terminal: retrying cannot succeed, so the row
    leaves rotation with a WARNING naming the fix and the reason recorded, instead of an hour-long
    bench with null error fields (#113718). Driven through ``_refresh_entry_impl`` against a temp
    HERMES_HOME so the production raise site's code is what reaches the classifier."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    providers = {"nous": nous_state} if nous_state else {}
    (tmp_path / "auth.json").write_text(json.dumps({"version": 1, "providers": providers}), encoding="utf-8")
    pool = _pool("nous")
    entry = _entry("nous", source="manual:device_code")
    pool._entries = [entry]
    cleared: list = []
    monkeypatch.setattr(pool, "_clear_terminal_nous_state", lambda e, exc: cleared.append(e.id))
    monkeypatch.setattr(pool, "_quarantine_sources", lambda e, sources: None)  # keep the row to inspect it

    with caplog.at_level(logging.INFO, logger=cp.logger.name):
        result = pool._refresh_entry_impl(entry, force=True)

    assert result is None and cleared == ["e1"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "terminally invalid" in r.getMessage()]
    assert len(warnings) == 1
    assert "hermes auth add nous" in warnings[0].getMessage()
    row = pool._entries[0]
    assert row.last_status == STATUS_DEAD
    assert row.last_error_reason == expected_code and row.last_error_message  # no more null error fields


def test_nous_transient_error_still_benched(monkeypatch, caplog):
    """A failure that says nothing about the login stays transient: benched, not cleared."""
    pool = _pool("nous")
    entry = _entry("nous")
    pool._entries = [entry]
    cleared: list = []
    benched: list = []
    monkeypatch.setattr(pool, "_sync_nous_entry_from_auth_store", lambda e: e)
    monkeypatch.setattr(pool, "_clear_terminal_nous_state", lambda e, exc: cleared.append(e.id))
    monkeypatch.setattr(pool, "_mark_exhausted", lambda e, *a, **k: benched.append(e.id))

    with caplog.at_level(logging.INFO, logger=cp.logger.name):
        result = pool._recover_failed_refresh(entry, RuntimeError("boom"))

    assert result is None and cleared == [] and benched == ["e1"]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING and "terminally invalid" in r.getMessage()]
