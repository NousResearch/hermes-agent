"""Upstream error text must not carry a raw API key into ``auth.json`` (#71994).

Providers echo the rejected key back ("Invalid API key: sk-..."). The pool stores that text in
``last_error_message``, which the borrowed-credential sanitizer deliberately keeps, so the
redaction has to happen where the pool records it, and regardless of ``security.redact_secrets``.
"""
from __future__ import annotations

import json
import threading

from agent import credential_pool as cp
from agent.credential_pool import CredentialPool, PooledCredential

_KEY = "sk-ant-api03-" + "Zq7Wm2Xr9Lk4Pv8Ty3Nb6Hc1Jd5Fg0Se" * 2


def test_exhausted_error_message_is_redacted_on_disk_even_with_redaction_disabled(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": [
        {"id": "k1", "label": "a", "auth_type": "api_key", "priority": 0, "source": "manual",
         "access_token": "sk-ant-api03-stored"},
        {"id": "k2", "label": "b", "auth_type": "api_key", "priority": 1, "source": "manual",
         "access_token": "sk-ant-api03-other"},
    ]}}))

    pool = cp.load_pool("anthropic")
    assert pool.select().id == "k1"
    pool.mark_exhausted_and_rotate(status_code=401, error_context={"message": f"Invalid API key: {_KEY}"})

    raw = (home / "auth.json").read_text()
    assert _KEY not in raw
    row = json.loads(raw)["credential_pool"]["anthropic"][0]
    assert row["last_error_message"].startswith("Invalid API key:")


def test_string_error_body_reason_is_redacted_on_disk(tmp_path, monkeypatch):
    """A body whose ``error`` is a bare string makes the whole message the reason too."""
    from agent.agent_runtime_helpers import extract_api_error_context

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": [
        {"id": "k1", "label": "a", "auth_type": "api_key", "priority": 0, "source": "manual",
         "access_token": "sk-ant-stored"},
        {"id": "k2", "label": "b", "auth_type": "api_key", "priority": 1, "source": "manual",
         "access_token": "sk-ant-other"},
    ]}}))
    err = RuntimeError("401")
    err.body = {"error": f"Incorrect API key provided: {_KEY}"}
    context = extract_api_error_context(err)
    assert _KEY in context["reason"]

    pool = cp.load_pool("anthropic")
    assert pool.select().id == "k1"
    pool.mark_exhausted_and_rotate(status_code=401, error_context=context)

    raw = (home / "auth.json").read_text()
    assert _KEY not in raw
    row = json.loads(raw)["credential_pool"]["anthropic"][0]
    assert row["last_error_reason"].startswith("Incorrect API key provided:")


def test_terminal_reason_codes_survive_redaction():
    for code in cp._TERMINAL_AUTH_REASONS:
        normalized = cp._normalize_error_context({"reason": code})
        assert normalized["reason"] == code
        assert CredentialPool._is_terminal_auth_failure(None, 401, normalized)


def test_mark_dead_refresh_grant_redacts_exception_text(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    pool = CredentialPool.__new__(CredentialPool)
    pool._lock = threading.RLock()
    pool._active_leases = {}
    pool._current_id = None
    pool.provider = "anthropic"
    pool._entries = [PooledCredential(id="e1", provider="anthropic", auth_type="oauth", access_token="a",
                                      refresh_token="r", label="e1", source="manual:hermes_pkce", priority=0)]
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)

    pool._mark_dead_refresh_grant(pool._entries[0], RuntimeError(f"invalid_grant for key {_KEY}"))

    row = pool._entries[0]
    assert row.last_status == cp.STATUS_DEAD
    assert _KEY not in row.last_error_message and "invalid_grant" in row.last_error_message
