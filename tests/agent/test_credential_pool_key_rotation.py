"""Tests for credential pool upsert — key rotation clears exhaustion state."""

from __future__ import annotations

import json


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def test_key_rotation_clears_exhausted_status(tmp_path, monkeypatch):
    """Replacing an exhausted API key via _upsert_entry resets last_status.

    Regression: `hermes setup` saves a new OPENROUTER_API_KEY to .env, which
    triggers _seed_from_env → _upsert_entry.  If the existing pool entry was
    marked exhausted (e.g. from a rate-limit on the old key), the stale status
    was preserved on the new key — making the pool appear unusable even though
    a fresh valid key was present.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
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
                        "access_token": "old-key",
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
    )

    # Simulate the user rotating their key (new value in env)
    monkeypatch.setenv("OPENROUTER_API_KEY", "new-rotated-key")

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    entry = pool.select()

    assert entry is not None, "Pool should have a usable entry after key rotation"
    assert entry.access_token == "new-rotated-key"
    assert entry.last_status is None, "last_status should be cleared after key rotation"
    assert entry.last_status_at is None
    assert entry.last_error_code is None
    assert entry.last_error_reason is None
    assert entry.last_error_message is None
    assert entry.last_error_reset_at is None


def test_same_key_preserves_exhausted_status(tmp_path, monkeypatch):
    """If the key has NOT changed, _upsert_entry does not clear exhaustion state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    from agent.credential_pool import PooledCredential, _upsert_entry

    existing = PooledCredential.from_dict(
        "openrouter",
        {
            "id": "cred-1",
            "label": "OPENROUTER_API_KEY",
            "auth_type": "api_key",
            "priority": 0,
            "source": "env:OPENROUTER_API_KEY",
            "access_token": "same-key",
            "last_status": "exhausted",
            "last_status_at": 1000.0,
            "last_error_code": 429,
            "last_error_reason": "rate_limit",
            "last_error_message": "Too many requests",
            "last_error_reset_at": 2000.0,
        },
    )
    entries = [existing]

    # Upsert with the same token — should NOT clear exhaustion
    _upsert_entry(
        entries,
        "openrouter",
        "env:OPENROUTER_API_KEY",
        {
            "source": "env:OPENROUTER_API_KEY",
            "auth_type": "api_key",
            "access_token": "same-key",
        },
    )

    assert entries[0].last_status == "exhausted", (
        "last_status should not be cleared when the key is unchanged"
    )


def test_env_row_rotation_clears_persisted_exhausted_status(tmp_path, monkeypatch):
    """Env-backed rows keep only a fingerprint on disk; a rotation must still clear the 401.

    ``_upsert_entry`` clears the in-memory status, but ``write_credential_pool`` merged the
    on-disk cooldown back: such a row has no token pair, and its cleared ``last_status_at``
    (None -> epoch 0) looked older than the disk's 401 stamp. The exhausted state of a 401
    taken against the PREVIOUS key therefore survived the key fix on disk, and the next
    process reloaded it as still-exhausted (CommandCode 401 on every non-default profile).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    import time

    from agent.credential_persistence import fingerprint_secret_value

    old_fingerprint = fingerprint_secret_value("old-key")
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "commandcode": [
                    {
                        "id": "cred-1",
                        "label": "COMMANDCODE_API_KEY",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "env:COMMANDCODE_API_KEY",
                        "last_status": "exhausted",
                        # Still inside the exhaustion window: this is the 401 the user takes,
                        # then fixes the key in 1Password a minute later.
                        "last_status_at": time.time() - 5,
                        "last_error_code": 401,
                        "last_error_message": "Invalid 'Authorization' header or token.",
                        "secret_fingerprint": old_fingerprint,
                        "secret_source": "onepassword",
                        "failure_reason": "auth",
                    }
                ]
            },
        },
    )
    monkeypatch.setenv("COMMANDCODE_API_KEY", "new-key")

    from agent.credential_pool import load_pool

    pool = load_pool("commandcode")
    entry = pool.entries()[0]
    assert entry.access_token == "new-key"
    assert entry.last_status is None

    import json

    persisted = json.loads((tmp_path / "hermes" / "auth.json").read_text())[
        "credential_pool"
    ]["commandcode"][0]
    assert persisted["last_status"] is None
    assert persisted["secret_fingerprint"] == fingerprint_secret_value("new-key")
