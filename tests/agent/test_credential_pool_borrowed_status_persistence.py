"""Borrowed credentials keep their exhaustion state across load_pool() calls."""

from __future__ import annotations

import json
import logging


def _fingerprint(provider: str, token: str) -> str:
    from agent.credential_persistence import sanitize_borrowed_credential_payload

    return sanitize_borrowed_credential_payload(
        {"source": "env:OPENROUTER_API_KEY", "access_token": token}, provider
    )["secret_fingerprint"]


def _write_exhausted_env_entry(tmp_path, token: str) -> None:
    """Persist the pool exactly as write_credential_pool would.

    A borrowed source keeps no ``access_token`` on disk — only the
    non-reversible ``secret_fingerprint`` left behind by
    ``sanitize_borrowed_credential_payload``.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": "cred-1",
                            "label": "OPENROUTER_API_KEY",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "env:OPENROUTER_API_KEY",
                            "secret_fingerprint": _fingerprint("openrouter", token),
                            "last_status": "exhausted",
                            "last_status_at": 1000.0,
                            "last_error_code": 429,
                            "last_error_reason": "rate_limit",
                            "last_error_message": "Too many requests",
                            "last_error_reset_at": 2000.0,
                        }
                    ]
                },
            },
            indent=2,
        )
    )


def test_unchanged_env_key_keeps_exhaustion_state(tmp_path, monkeypatch):
    """A reload with the same env key must not clear the 429 cooldown.

    Borrowed sources persist no raw secret, so comparing the incoming token
    against the stored (empty) ``access_token`` reported a rotation on every
    load — wiping the cooldown the pool had just recorded.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "same-key")
    _write_exhausted_env_entry(tmp_path, "same-key")

    from agent.credential_pool import load_pool

    for _ in range(3):
        entry = load_pool("openrouter").entries()[0]
        assert entry.last_status == "exhausted"
        assert entry.last_error_code == 429
        assert entry.last_error_reset_at == 2000.0


def test_rotated_env_key_still_clears_exhaustion_state(tmp_path, monkeypatch):
    """A genuinely different env key is a rotation and must reset the status."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "new-rotated-key")
    _write_exhausted_env_entry(tmp_path, "old-key")

    from agent.credential_pool import load_pool

    entry = load_pool("openrouter").entries()[0]
    assert entry.last_status is None
    assert entry.last_error_code is None
    assert entry.last_error_reset_at is None


def test_missing_fingerprint_is_not_treated_as_a_rotation(tmp_path, monkeypatch):
    """An entry predating fingerprint persistence must not lose its cooldown.

    Nothing on disk identifies the old secret, so no comparison is possible —
    "unknown" is not "changed".
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "some-key")
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": "cred-1",
                            "label": "OPENROUTER_API_KEY",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "env:OPENROUTER_API_KEY",
                            "last_status": "exhausted",
                            "last_status_at": 1000.0,
                            "last_error_code": 429,
                        }
                    ]
                },
            },
            indent=2,
        )
    )

    from agent.credential_pool import load_pool

    entry = load_pool("openrouter").entries()[0]
    assert entry.last_status == "exhausted"
    assert entry.last_error_code == 429


def _exhausted_env_entry(token_fingerprint: str):
    from agent.credential_pool import PooledCredential

    return PooledCredential(
        provider="openrouter",
        id="cred-1",
        label="OPENROUTER_API_KEY",
        auth_type="api_key",
        priority=0,
        source="env:OPENROUTER_API_KEY",
        access_token="",
        last_status="exhausted",
        last_error_code=429,
        extra={"secret_fingerprint": token_fingerprint},
    )


def test_payload_growing_a_secret_field_is_not_a_rotation():
    """Field drift must not silently reintroduce the every-load rotation.

    The stored fingerprint is whichever field the writer preferred. A borrowed
    payload that later carries an ``agent_key`` next to the same unchanged
    ``access_token`` fingerprints a different field — matching against every
    fingerprint the payload can produce keeps that from reading as a rotation.
    """
    from agent.credential_pool import _incoming_token_is_a_rotation

    entry = _exhausted_env_entry(_fingerprint("openrouter", "same-key"))
    payload = {
        "source": "env:OPENROUTER_API_KEY",
        "access_token": "same-key",
        "agent_key": "a-derived-agent-key",
    }

    assert _incoming_token_is_a_rotation(entry, "openrouter", payload) is False


def test_genuinely_different_secret_is_still_a_rotation():
    """The any-field match must not swallow a real key change."""
    from agent.credential_pool import _incoming_token_is_a_rotation

    entry = _exhausted_env_entry(_fingerprint("openrouter", "old-key"))
    payload = {"source": "env:OPENROUTER_API_KEY", "access_token": "new-key"}

    assert _incoming_token_is_a_rotation(entry, "openrouter", payload) is True


def test_unfingerprintable_incoming_payload_logs_why_status_is_kept(caplog):
    """Stored fingerprint, nothing comparable incoming: keep status, say so.

    "Unknown is not changed" is deliberate policy, so it has to be
    distinguishable in a log from a comparison that failed by accident.
    """
    from agent.credential_pool import _incoming_token_is_a_rotation

    entry = _exhausted_env_entry(_fingerprint("openrouter", "same-key"))
    payload = {"source": "env:OPENROUTER_API_KEY", "access_token": ""}

    with caplog.at_level(logging.DEBUG, logger="agent.credential_pool"):
        assert _incoming_token_is_a_rotation(entry, "openrouter", payload) is False

    assert "unknown is not changed" in caplog.text


def test_entry_without_extra_does_not_raise():
    """No construction path may make the comparison blow up on ``extra``."""
    from agent.credential_pool import PooledCredential, _incoming_token_is_a_rotation

    entry = PooledCredential(
        provider="openrouter",
        id="cred-1",
        label="OPENROUTER_API_KEY",
        auth_type="api_key",
        priority=0,
        source="env:OPENROUTER_API_KEY",
        access_token="",
        last_status="exhausted",
    )
    object.__setattr__(entry, "extra", None)
    payload = {"source": "env:OPENROUTER_API_KEY", "access_token": "some-key"}

    assert _incoming_token_is_a_rotation(entry, "openrouter", payload) is False
