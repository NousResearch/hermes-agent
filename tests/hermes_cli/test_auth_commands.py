"""Tests for auth subcommands backed by the credential pool."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _jwt_with_email(email: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"email": email}).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.signature"


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def test_auth_add_api_key_persists_manual_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "openrouter"
        auth_type = "api-key"
        api_key = "sk-or-manual"
        label = "personal"

    auth_add_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = payload["credential_pool"]["openrouter"]
    entry = next(item for item in entries if item["source"] == "manual")
    assert entry["label"] == "personal"
    assert entry["auth_type"] == "api_key"
    assert entry["source"] == "manual"
    assert entry["access_token"] == "sk-or-manual"


def test_auth_add_anthropic_oauth_persists_pool_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    token = _jwt_with_email("claude@example.com")
    monkeypatch.setattr(
        "agent.anthropic_adapter.run_hermes_oauth_login_pure",
        lambda: {
            "access_token": token,
            "refresh_token": "refresh-token",
            "expires_at_ms": 1711234567000,
        },
    )

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "anthropic"
        auth_type = "oauth"
        api_key = None
        label = None

    auth_add_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = payload["credential_pool"]["anthropic"]
    entry = next(item for item in entries if item["source"] == "manual:hermes_pkce")
    assert entry["label"] == "claude@example.com"
    assert entry["source"] == "manual:hermes_pkce"
    assert entry["refresh_token"] == "refresh-token"
    assert entry["expires_at_ms"] == 1711234567000


def test_auth_add_nous_oauth_persists_pool_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    token = _jwt_with_email("nous@example.com")
    monkeypatch.setattr(
        "hermes_cli.auth._nous_device_code_login",
        lambda **kwargs: {
            "portal_base_url": "https://portal.example.com",
            "inference_base_url": "https://inference.example.com/v1",
            "client_id": "hermes-cli",
            "scope": "inference:mint_agent_key",
            "token_type": "Bearer",
            "access_token": token,
            "refresh_token": "refresh-token",
            "obtained_at": "2026-03-23T10:00:00+00:00",
            "expires_at": "2026-03-23T11:00:00+00:00",
            "expires_in": 3600,
            "agent_key": "ak-test",
            "agent_key_id": "ak-id",
            "agent_key_expires_at": "2026-03-23T10:30:00+00:00",
            "agent_key_expires_in": 1800,
            "agent_key_reused": False,
            "agent_key_obtained_at": "2026-03-23T10:00:10+00:00",
            "tls": {"insecure": False, "ca_bundle": None},
        },
    )

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "nous"
        auth_type = "oauth"
        api_key = None
        label = None
        portal_url = None
        inference_url = None
        client_id = None
        scope = None
        no_browser = False
        timeout = None
        insecure = False
        ca_bundle = None

    auth_add_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = payload["credential_pool"]["nous"]
    entry = next(item for item in entries if item["source"] == "manual:device_code")
    assert entry["label"] == "nous@example.com"
    assert entry["source"] == "manual:device_code"
    assert entry["agent_key"] == "ak-test"
    assert entry["portal_base_url"] == "https://portal.example.com"


def test_auth_add_codex_oauth_persists_pool_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    token = _jwt_with_email("codex@example.com")
    monkeypatch.setattr(
        "hermes_cli.auth._codex_device_code_login",
        lambda: {
            "tokens": {
                "access_token": token,
                "refresh_token": "refresh-token",
            },
            "base_url": "https://chatgpt.com/backend-api/codex",
            "last_refresh": "2026-03-23T10:00:00Z",
        },
    )

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "openai-codex"
        auth_type = "oauth"
        api_key = None
        label = None

    auth_add_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = payload["credential_pool"]["openai-codex"]
    entry = next(item for item in entries if item["source"] == "manual:device_code")
    assert entry["label"] == "codex@example.com"
    assert entry["source"] == "manual:device_code"
    assert entry["refresh_token"] == "refresh-token"
    assert entry["base_url"] == "https://chatgpt.com/backend-api/codex"


def test_auth_remove_reindexes_priorities(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # Prevent pool auto-seeding from host env vars and file-backed sources
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-ant-api-primary",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "sk-ant-api-secondary",
                    },
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_remove_command

    class _Args:
        provider = "anthropic"
        target = "1"

    auth_remove_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = payload["credential_pool"]["anthropic"]
    assert len(entries) == 1
    assert entries[0]["label"] == "secondary"
    assert entries[0]["priority"] == 0


def test_auth_remove_accepts_label_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "cred-1",
                        "label": "work-account",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:device_code",
                        "access_token": "tok-1",
                    },
                    {
                        "id": "cred-2",
                        "label": "personal-account",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual:device_code",
                        "access_token": "tok-2",
                    },
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_remove_command

    class _Args:
        provider = "openai-codex"
        target = "personal-account"

    auth_remove_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entries = payload["credential_pool"]["openai-codex"]
    assert len(entries) == 1
    assert entries[0]["label"] == "work-account"


def test_auth_remove_prefers_exact_numeric_label_over_index(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "cred-a",
                        "label": "first",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:device_code",
                        "access_token": "tok-a",
                    },
                    {
                        "id": "cred-b",
                        "label": "2",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual:device_code",
                        "access_token": "tok-b",
                    },
                    {
                        "id": "cred-c",
                        "label": "third",
                        "auth_type": "oauth",
                        "priority": 2,
                        "source": "manual:device_code",
                        "access_token": "tok-c",
                    },
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_remove_command

    class _Args:
        provider = "openai-codex"
        target = "2"

    auth_remove_command(_Args())

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    labels = [entry["label"] for entry in payload["credential_pool"]["openai-codex"]]
    assert labels == ["first", "third"]


def test_auth_reset_clears_provider_statuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-ant-api-primary",
                        "last_status": "exhausted",
                        "last_status_at": 1711230000.0,
                        "last_error_code": 402,
                    }
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_reset_command

    class _Args:
        provider = "anthropic"

    auth_reset_command(_Args())

    out = capsys.readouterr().out
    assert "Reset status" in out

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entry = payload["credential_pool"]["anthropic"][0]
    assert entry["last_status"] is None
    assert entry["last_status_at"] is None
    assert entry["last_error_code"] is None


def test_clear_provider_auth_removes_provider_pool_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "active_provider": "anthropic",
            "providers": {
                "anthropic": {"access_token": "legacy-token"},
            },
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:hermes_pkce",
                        "access_token": "pool-token",
                    }
                ],
                "openrouter": [
                    {
                        "id": "cred-2",
                        "label": "other-provider",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-or-test",
                    }
                ],
            },
        },
    )

    from hermes_cli.auth import clear_provider_auth

    assert clear_provider_auth("anthropic") is True

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert payload["active_provider"] is None
    assert "anthropic" not in payload.get("providers", {})
    assert "anthropic" not in payload.get("credential_pool", {})
    assert "openrouter" in payload.get("credential_pool", {})


def test_auth_list_does_not_call_mutating_select(monkeypatch, capsys):
    from hermes_cli.auth_commands import auth_list_command

    class _Entry:
        id = "cred-1"
        label = "primary"
        auth_type="***"
        source = "manual"
        last_status = None
        last_error_code = None
        last_status_at = None

    class _Pool:
        def entries(self):
            return [_Entry()]

        def peek(self):
            return _Entry()

        def select(self):
            raise AssertionError("auth_list_command should not call select()")

    monkeypatch.setattr(
        "hermes_cli.auth_commands.load_pool",
        lambda provider: _Pool() if provider == "openrouter" else type("_EmptyPool", (), {"entries": lambda self: []})(),
    )

    class _Args:
        provider = "openrouter"

    auth_list_command(_Args())

    out = capsys.readouterr().out
    assert "openrouter (1 credentials):" in out
    assert "primary" in out


def test_auth_list_shows_exhausted_cooldown(monkeypatch, capsys):
    from hermes_cli.auth_commands import auth_list_command

    class _Entry:
        id = "cred-1"
        label = "primary"
        auth_type = "api_key"
        source = "manual"
        last_status = "exhausted"
        last_error_code = 429
        last_status_at = 1000.0

    class _Pool:
        def entries(self):
            return [_Entry()]

        def peek(self):
            return None

    monkeypatch.setattr("hermes_cli.auth_commands.load_pool", lambda provider: _Pool())
    monkeypatch.setattr("hermes_cli.auth_commands.time.time", lambda: 1030.0)

    class _Args:
        provider = "openrouter"

    auth_list_command(_Args())

    out = capsys.readouterr().out
    assert "exhausted (429)" in out
    assert "59m 30s left" in out


def test_auth_list_prefers_explicit_reset_time(monkeypatch, capsys):
    from hermes_cli.auth_commands import auth_list_command

    class _Entry:
        id = "cred-1"
        label = "weekly"
        auth_type = "oauth"
        source = "manual:device_code"
        last_status = "exhausted"
        last_error_code = 429
        last_error_reason = "device_code_exhausted"
        last_error_message = "Weekly credits exhausted."
        last_error_reset_at = "2026-04-12T10:30:00Z"
        last_status_at = 1000.0

    class _Pool:
        def entries(self):
            return [_Entry()]

        def peek(self):
            return None

    monkeypatch.setattr("hermes_cli.auth_commands.load_pool", lambda provider: _Pool())
    monkeypatch.setattr(
        "hermes_cli.auth_commands.time.time",
        lambda: datetime(2026, 4, 5, 10, 30, tzinfo=timezone.utc).timestamp(),
    )

    class _Args:
        provider = "openai-codex"

    auth_list_command(_Args())

    out = capsys.readouterr().out
    assert "device_code_exhausted" in out
    assert "7d 0h left" in out


def test_auth_remove_env_seeded_clears_env_var(tmp_path, monkeypatch):
    """Removing an env-seeded credential should also clear the env var from .env
    so the entry doesn't get re-seeded on the next load_pool() call."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Write a .env with an OpenRouter key
    env_path = hermes_home / ".env"
    env_path.write_text("OPENROUTER_API_KEY=sk-or-test-key-12345\nOTHER_KEY=keep-me\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345")

    # Seed the pool with the env entry
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "env-1",
                        "label": "OPENROUTER_API_KEY",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "env:OPENROUTER_API_KEY",
                        "access_token": "sk-or-test-key-12345",
                    }
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_remove_command

    class _Args:
        provider = "openrouter"
        target = "1"

    auth_remove_command(_Args())

    # Env var should be cleared from os.environ
    import os
    assert os.environ.get("OPENROUTER_API_KEY") is None

    # Env var should be removed from .env file
    env_content = env_path.read_text()
    assert "OPENROUTER_API_KEY" not in env_content
    # Other keys should still be there
    assert "OTHER_KEY=keep-me" in env_content


def test_auth_remove_env_seeded_does_not_resurrect(tmp_path, monkeypatch):
    """After removing an env-seeded credential, load_pool should NOT re-create it."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Write .env with an OpenRouter key
    env_path = hermes_home / ".env"
    env_path.write_text("OPENROUTER_API_KEY=sk-or-test-key-12345\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345")

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "env-1",
                        "label": "OPENROUTER_API_KEY",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "env:OPENROUTER_API_KEY",
                        "access_token": "sk-or-test-key-12345",
                    }
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_remove_command

    class _Args:
        provider = "openrouter"
        target = "1"

    auth_remove_command(_Args())

    # Now reload the pool — the entry should NOT come back
    from agent.credential_pool import load_pool
    pool = load_pool("openrouter")
    assert not pool.has_credentials()


def test_auth_remove_manual_entry_does_not_touch_env(tmp_path, monkeypatch):
    """Removing a manually-added credential should NOT touch .env."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    env_path = hermes_home / ".env"
    env_path.write_text("SOME_KEY=some-value\n")

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "manual-1",
                        "label": "my-key",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-or-manual-key",
                    }
                ]
            },
        },
    )

    from hermes_cli.auth_commands import auth_remove_command

    class _Args:
        provider = "openrouter"
        target = "1"

    auth_remove_command(_Args())

    # .env should be untouched
    assert env_path.read_text() == "SOME_KEY=some-value\n"


def test_auth_remove_claude_code_suppresses_reseed(tmp_path, monkeypatch):
    """Removing a claude_code credential must prevent it from being re-seeded."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, {"claude_code"}),
    )
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)

    auth_store = {
        "version": 1,
        "credential_pool": {
            "anthropic": [{
                "id": "cc1",
                "label": "claude_code",
                "auth_type": "oauth",
                "priority": 0,
                "source": "claude_code",
                "access_token": "sk-ant-oat01-token",
            }]
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))

    from types import SimpleNamespace
    from hermes_cli.auth_commands import auth_remove_command
    auth_remove_command(SimpleNamespace(provider="anthropic", target="1"))

    updated = json.loads((hermes_home / "auth.json").read_text())
    suppressed = updated.get("suppressed_sources", {})
    assert "anthropic" in suppressed
    assert "claude_code" in suppressed["anthropic"]
