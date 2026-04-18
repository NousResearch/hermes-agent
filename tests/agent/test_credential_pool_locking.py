"""Tests for credential pool batch-persist and file-locking behaviour."""

from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Batch persist: _available_entries should call _persist at most once
# ---------------------------------------------------------------------------


def test_available_entries_calls_persist_once_for_anthropic_sync(tmp_path, monkeypatch):
    """When _available_entries syncs multiple anthropic entries, _persist is called only once."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)

    # Two claude_code entries that are both exhausted with stale refresh tokens
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-1",
                        "label": "cc-1",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "claude_code",
                        "access_token": "old-access-1",
                        "refresh_token": "old-refresh-1",
                        "last_status": "exhausted",
                        "last_status_at": time.time(),
                        "last_error_code": 401,
                    },
                    {
                        "id": "cred-2",
                        "label": "cc-2",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "claude_code",
                        "access_token": "old-access-2",
                        "refresh_token": "old-refresh-2",
                        "last_status": "exhausted",
                        "last_status_at": time.time(),
                        "last_error_code": 401,
                    },
                ]
            },
        },
    )

    # The credentials file has a newer refresh token
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: {
            "accessToken": "new-access",
            "refreshToken": "new-refresh",
            "expiresAt": int(time.time() * 1000) + 3_600_000,
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_hermes_oauth_credentials",
        lambda: None,
    )

    from agent.credential_pool import CredentialPool, PooledCredential

    raw = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = [
        PooledCredential.from_dict("anthropic", e)
        for e in raw["credential_pool"]["anthropic"]
    ]
    pool = CredentialPool("anthropic", entries)

    with patch.object(pool, "_persist", wraps=pool._persist) as mock_persist:
        available = pool._available_entries(clear_expired=True)

    # Both entries were synced (cleared_any=True) but _persist should be
    # called exactly once — at the end of _available_entries.
    assert mock_persist.call_count == 1


def test_available_entries_calls_persist_once_for_codex_sync(tmp_path, monkeypatch):
    """When _available_entries syncs codex entries, _persist is called only once."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "codex-1",
                        "label": "codex-primary",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "device_code",
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                        "last_status": "exhausted",
                        "last_status_at": time.time(),
                        "last_error_code": 429,
                    },
                ]
            },
        },
    )

    monkeypatch.setattr(
        "agent.credential_pool._import_codex_cli_tokens",
        lambda: {
            "access_token": "cli-new-access",
            "refresh_token": "cli-new-refresh",
        },
    )

    from agent.credential_pool import CredentialPool, PooledCredential

    raw = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = [
        PooledCredential.from_dict("openai-codex", e)
        for e in raw["credential_pool"]["openai-codex"]
    ]
    pool = CredentialPool("openai-codex", entries)

    with patch.object(pool, "_persist", wraps=pool._persist) as mock_persist:
        available = pool._available_entries(clear_expired=True)

    # Sync happened inside the loop with persist=False, then a single
    # _persist at the end because cleared_any was True.
    assert mock_persist.call_count == 1


def test_available_entries_no_persist_when_nothing_changed(tmp_path, monkeypatch):
    """_available_entries should not call _persist when no syncs or clears happen."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-ok",
                        "label": "healthy",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-or-ok",
                    }
                ]
            },
        },
    )

    from agent.credential_pool import CredentialPool, PooledCredential

    raw = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = [
        PooledCredential.from_dict("openrouter", e)
        for e in raw["credential_pool"]["openrouter"]
    ]
    pool = CredentialPool("openrouter", entries)

    with patch.object(pool, "_persist", wraps=pool._persist) as mock_persist:
        available = pool._available_entries()

    assert len(available) == 1
    assert mock_persist.call_count == 0


# ---------------------------------------------------------------------------
# File locking: _persist and load_pool use _auth_store_lock
# ---------------------------------------------------------------------------


def test_persist_acquires_auth_store_lock(tmp_path, monkeypatch):
    """_persist should acquire _auth_store_lock around the write."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {}})

    from agent.credential_pool import CredentialPool, PooledCredential

    entries = [
        PooledCredential(
            provider="openrouter",
            id="cred-1",
            label="test",
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token="sk-or-test",
        )
    ]
    pool = CredentialPool("openrouter", entries)

    lock_entered = False

    original_lock = pool._persist.__wrapped__ if hasattr(pool._persist, '__wrapped__') else None

    with patch("agent.credential_pool._auth_store_lock") as mock_lock:
        # Make the lock context manager work correctly
        mock_ctx = MagicMock()
        mock_lock.return_value = mock_ctx
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        pool._persist()

        mock_lock.assert_called_once()
        mock_ctx.__enter__.assert_called_once()
        mock_ctx.__exit__.assert_called_once()


def test_load_pool_acquires_auth_store_lock_for_read(tmp_path, monkeypatch):
    """load_pool should acquire _auth_store_lock when reading the pool file."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "test",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                    }
                ]
            },
        },
    )

    lock_calls = []

    from agent.credential_pool import _auth_store_lock as real_lock
    from contextlib import contextmanager

    @contextmanager
    def tracking_lock(*args, **kwargs):
        lock_calls.append("acquired")
        with real_lock(*args, **kwargs):
            yield

    with patch("agent.credential_pool._auth_store_lock", side_effect=tracking_lock):
        from agent.credential_pool import load_pool
        pool = load_pool("openrouter")

    # At least one lock acquisition should happen for the read path
    assert len(lock_calls) >= 1
    assert pool.has_credentials()
