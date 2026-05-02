"""Tests for auth subcommands backed by the credential pool."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import yaml


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

    # Pool has exactly one canonical `device_code` entry — not a duplicate
    # pair of `manual:device_code` + `device_code` (the latter would be
    # materialised by _seed_from_singletons on every load_pool).
    entries = payload["credential_pool"]["nous"]
    device_code_entries = [
        item for item in entries if item["source"] == "device_code"
    ]
    assert len(device_code_entries) == 1, entries
    assert not any(item["source"] == "manual:device_code" for item in entries)
    entry = device_code_entries[0]
    assert entry["source"] == "device_code"
    assert entry["agent_key"] == "ak-test"
    assert entry["portal_base_url"] == "https://portal.example.com"

    # `hermes auth add nous` must also populate providers.nous so the
    # 401-recovery path (resolve_nous_runtime_credentials) can mint a fresh
    # agent_key when the 24h TTL expires. If this mirror is missing, recovery
    # raises "Hermes is not logged into Nous Portal" and the agent dies.
    singleton = payload["providers"]["nous"]
    assert singleton["access_token"] == token
    assert singleton["refresh_token"] == "refresh-token"
    assert singleton["agent_key"] == "ak-test"
    assert singleton["portal_base_url"] == "https://portal.example.com"
    assert singleton["inference_base_url"] == "https://inference.example.com/v1"


def test_auth_add_nous_oauth_honors_custom_label(tmp_path, monkeypatch):
    """`hermes auth add nous --type oauth --label <name>` must preserve the
    custom label end-to-end — it was silently dropped in the first cut of the
    persist_nous_credentials helper because `--label` wasn't threaded through.
    """
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
        label = "my-nous"
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

    # Custom label reaches the pool entry …
    pool_entry = payload["credential_pool"]["nous"][0]
    assert pool_entry["source"] == "device_code"
    assert pool_entry["label"] == "my-nous"

    # … and survives in providers.nous so a subsequent load_pool() re-seeds
    # it without reverting to the auto-derived fingerprint.
    assert payload["providers"]["nous"]["label"] == "my-nous"


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
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
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
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
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


def test_logout_resets_codex_config_when_auth_state_already_cleared(tmp_path, monkeypatch, capsys):
    """`hermes logout --provider openai-codex` must still clear model.provider.

    Users can end up with auth.json already cleared but config.yaml still set to
    openai-codex.  Previously logout reported no auth state and left the agent
    pinned to the Codex provider.
    """
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "credential_pool": {}})
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-5.3-codex\n"
        "  provider: openai-codex\n"
        "  base_url: https://chatgpt.com/backend-api/codex\n"
    )

    from types import SimpleNamespace
    from hermes_cli.auth import logout_command

    logout_command(SimpleNamespace(provider="openai-codex"))

    out = capsys.readouterr().out
    assert "Logged out of OpenAI Codex." in out
    config_text = (hermes_home / "config.yaml").read_text()
    assert "provider: auto" in config_text
    assert "base_url: https://openrouter.ai/api/v1" in config_text


def test_logout_defaults_to_configured_codex_when_no_active_provider(tmp_path, monkeypatch, capsys):
    """Bare `hermes logout` should target configured Codex if auth has no active provider."""
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_auth_store(tmp_path, {"version": 1, "providers": {}, "credential_pool": {}})
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-5.3-codex\n"
        "  provider: openai-codex\n"
        "  base_url: https://chatgpt.com/backend-api/codex\n"
    )

    from types import SimpleNamespace
    from hermes_cli.auth import logout_command

    logout_command(SimpleNamespace(provider=None))

    out = capsys.readouterr().out
    assert "Logged out of OpenAI Codex." in out
    config_text = (hermes_home / "config.yaml").read_text()
    assert "provider: auto" in config_text


def test_logout_clears_stale_active_codex_without_provider_credentials(tmp_path, monkeypatch, capsys):
    """Logout must clear active_provider even when provider credential payloads are gone."""
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {},
            "credential_pool": {},
        },
    )
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-5.3-codex\n"
        "  provider: openai-codex\n"
        "  base_url: https://chatgpt.com/backend-api/codex\n"
    )

    from types import SimpleNamespace
    from hermes_cli.auth import logout_command

    logout_command(SimpleNamespace(provider=None))

    out = capsys.readouterr().out
    assert "Logged out of OpenAI Codex." in out
    auth_payload = json.loads((hermes_home / "auth.json").read_text())
    assert auth_payload.get("active_provider") is None
    config_text = (hermes_home / "config.yaml").read_text()
    assert "provider: auto" in config_text


def test_reset_config_provider_uses_atomic_yaml_write(tmp_path, monkeypatch):
    """Logout config reset should delegate the YAML write atomically."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config_path = hermes_home / "config.yaml"
    original = {
        "model": {
            "default": "gpt-5.3-codex",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }
    }
    config_path.write_text(yaml.safe_dump(original, sort_keys=False), encoding="utf-8")
    original_text = config_path.read_text(encoding="utf-8")

    from hermes_cli.auth import _reset_config_provider

    def _boom(path, data, **kwargs):
        assert path == config_path
        assert data["model"]["provider"] == "auto"
        assert data["model"]["base_url"] == "https://openrouter.ai/api/v1"
        assert kwargs["sort_keys"] is False
        raise OSError("simulated atomic write failure")

    with patch("hermes_cli.auth.atomic_yaml_write", side_effect=_boom) as mock_write:
        with pytest.raises(OSError, match="simulated atomic write failure"):
            _reset_config_provider()

    assert mock_write.call_count == 1
    assert config_path.read_text(encoding="utf-8") == original_text


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
    assert "rate-limited (429)" in out
    assert "59m 30s left" in out


def test_auth_list_shows_auth_failure_when_exhausted_entry_is_unauthorized(monkeypatch, capsys):
    from hermes_cli.auth_commands import auth_list_command

    class _Entry:
        id = "cred-1"
        label = "primary"
        auth_type = "oauth"
        source = "manual:device_code"
        last_status = "exhausted"
        last_error_code = 401
        last_error_reason = "invalid_token"
        last_error_message = "Access token expired or revoked."
        last_status_at = 1000.0

    class _Pool:
        def entries(self):
            return [_Entry()]

        def peek(self):
            return None

    monkeypatch.setattr("hermes_cli.auth_commands.load_pool", lambda provider: _Pool())
    monkeypatch.setattr("hermes_cli.auth_commands.time.time", lambda: 1030.0)

    class _Args:
        provider = "openai-codex"

    auth_list_command(_Args())

    out = capsys.readouterr().out
    assert "auth failed invalid_token (401)" in out
    assert "re-auth may be required" in out
    assert "left" not in out


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


def test_unsuppress_credential_source_clears_marker(tmp_path, monkeypatch):
    """unsuppress_credential_source() removes a previously-set marker."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1})

    from hermes_cli.auth import suppress_credential_source, unsuppress_credential_source, is_source_suppressed

    suppress_credential_source("openai-codex", "device_code")
    assert is_source_suppressed("openai-codex", "device_code") is True

    cleared = unsuppress_credential_source("openai-codex", "device_code")
    assert cleared is True
    assert is_source_suppressed("openai-codex", "device_code") is False

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    # Empty suppressed_sources dict should be cleaned up entirely
    assert "suppressed_sources" not in payload


def test_unsuppress_credential_source_returns_false_when_absent(tmp_path, monkeypatch):
    """unsuppress_credential_source() returns False if no marker exists."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1})

    from hermes_cli.auth import unsuppress_credential_source

    assert unsuppress_credential_source("openai-codex", "device_code") is False
    assert unsuppress_credential_source("nonexistent", "whatever") is False


def test_unsuppress_credential_source_preserves_other_markers(tmp_path, monkeypatch):
    """Clearing one marker must not affect unrelated markers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1})

    from hermes_cli.auth import (
        suppress_credential_source,
        unsuppress_credential_source,
        is_source_suppressed,
    )

    suppress_credential_source("openai-codex", "device_code")
    suppress_credential_source("anthropic", "claude_code")

    assert unsuppress_credential_source("openai-codex", "device_code") is True
    assert is_source_suppressed("anthropic", "claude_code") is True


def test_auth_remove_codex_device_code_suppresses_reseed(tmp_path, monkeypatch):
    """Removing an auto-seeded openai-codex credential must mark the source as suppressed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, {"device_code"}),
    )
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)

    auth_store = {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "acc-1",
                    "refresh_token": "ref-1",
                },
            },
        },
        "credential_pool": {
            "openai-codex": [{
                "id": "cx1",
                "label": "codex-auto",
                "auth_type": "oauth",
                "priority": 0,
                "source": "device_code",
                "access_token": "acc-1",
                "refresh_token": "ref-1",
            }]
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))

    from types import SimpleNamespace
    from hermes_cli.auth_commands import auth_remove_command

    auth_remove_command(SimpleNamespace(provider="openai-codex", target="1"))

    updated = json.loads((hermes_home / "auth.json").read_text())
    suppressed = updated.get("suppressed_sources", {})
    assert "openai-codex" in suppressed
    assert "device_code" in suppressed["openai-codex"]
    # Tokens in providers state should also be cleared
    assert "openai-codex" not in updated.get("providers", {})


def test_auth_remove_codex_manual_source_suppresses_reseed(tmp_path, monkeypatch):
    """Removing a manually-added (`manual:device_code`) openai-codex credential must also suppress."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)

    auth_store = {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "acc-2",
                    "refresh_token": "ref-2",
                },
            },
        },
        "credential_pool": {
            "openai-codex": [{
                "id": "cx2",
                "label": "manual-codex",
                "auth_type": "oauth",
                "priority": 0,
                "source": "manual:device_code",
                "access_token": "acc-2",
                "refresh_token": "ref-2",
            }]
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))

    from types import SimpleNamespace
    from hermes_cli.auth_commands import auth_remove_command

    auth_remove_command(SimpleNamespace(provider="openai-codex", target="1"))

    updated = json.loads((hermes_home / "auth.json").read_text())
    suppressed = updated.get("suppressed_sources", {})
    # Critical: manual:device_code source must also trigger the suppression path
    assert "openai-codex" in suppressed
    assert "device_code" in suppressed["openai-codex"]
    assert "openai-codex" not in updated.get("providers", {})


def test_auth_add_codex_clears_suppression_marker(tmp_path, monkeypatch):
    """Re-linking codex via `hermes auth add openai-codex` must clear any suppression marker."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)

    # Pre-existing suppression (simulating a prior `hermes auth remove`)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {"openai-codex": ["device_code"]},
    }))

    token = _jwt_with_email("codex@example.com")
    monkeypatch.setattr(
        "hermes_cli.auth._codex_device_code_login",
        lambda: {
            "tokens": {
                "access_token": token,
                "refresh_token": "refreshed",
            },
            "base_url": "https://chatgpt.com/backend-api/codex",
            "last_refresh": "2026-01-01T00:00:00Z",
        },
    )

    from hermes_cli.auth_commands import auth_add_command

    class _Args:
        provider = "openai-codex"
        auth_type = "oauth"
        api_key = None
        label = None

    auth_add_command(_Args())

    payload = json.loads((hermes_home / "auth.json").read_text())
    # Suppression marker must be cleared
    assert "openai-codex" not in payload.get("suppressed_sources", {})
    # New pool entry must be present
    entries = payload["credential_pool"]["openai-codex"]
    assert any(e["source"] == "manual:device_code" for e in entries)


def test_seed_from_singletons_respects_codex_suppression(tmp_path, monkeypatch):
    """_seed_from_singletons() for openai-codex must skip auto-import when suppressed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)

    # Suppression marker in place
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {"openai-codex": ["device_code"]},
    }))

    # Make _import_codex_cli_tokens return tokens — these would normally trigger
    # a re-seed, but suppression must skip it.
    def _fake_import():
        return {
            "access_token": "would-be-reimported",
            "refresh_token": "would-be-reimported",
        }

    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", _fake_import)

    from agent.credential_pool import _seed_from_singletons

    entries = []
    changed, active_sources = _seed_from_singletons("openai-codex", entries)

    # With suppression in place: nothing changes, no entries added, no sources
    assert changed is False
    assert entries == []
    assert active_sources == set()

    # Verify the auth store was NOT modified (no auto-import happened)
    after = json.loads((hermes_home / "auth.json").read_text())
    assert "openai-codex" not in after.get("providers", {})


def test_auth_remove_env_seeded_suppresses_shell_exported_var(tmp_path, monkeypatch, capsys):
    """`hermes auth remove xai 1` must stick even when the env var is exported
    by the shell (not written into ~/.hermes/.env).  Before PR for #13371 the
    removal silently restored on next load_pool() because _seed_from_env()
    re-read os.environ.  Now env:<VAR> is suppressed in auth.json.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Simulate shell export (NOT written to .env)
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-shell-export")
    (hermes_home / ".env").write_text("")

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "xai": [{
                    "id": "env-1",
                    "label": "XAI_API_KEY",
                    "auth_type": "api_key",
                    "priority": 0,
                    "source": "env:XAI_API_KEY",
                    "access_token": "sk-xai-shell-export",
                    "base_url": "https://api.x.ai/v1",
                }]
            },
        },
    )

    from types import SimpleNamespace
    from hermes_cli.auth_commands import auth_remove_command
    auth_remove_command(SimpleNamespace(provider="xai", target="1"))

    # Suppression marker written
    after = json.loads((hermes_home / "auth.json").read_text())
    assert "env:XAI_API_KEY" in after.get("suppressed_sources", {}).get("xai", [])

    # Diagnostic printed pointing at the shell
    out = capsys.readouterr().out
    assert "still set in your shell environment" in out
    assert "Cleared XAI_API_KEY from .env" not in out  # wasn't in .env

    # Fresh simulation: shell re-exports, reload pool
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-shell-export")
    from agent.credential_pool import load_pool
    pool = load_pool("xai")
    assert not pool.has_credentials(), "pool must stay empty — env:XAI_API_KEY suppressed"


def test_auth_remove_env_seeded_dotenv_only_no_shell_hint(tmp_path, monkeypatch, capsys):
    """When the env var lives only in ~/.hermes/.env (not the shell), the
    shell-hint should NOT be printed — avoid scaring the user about a
    non-existent shell export.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Key ONLY in .env, shell must not have it
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (hermes_home / ".env").write_text("DEEPSEEK_API_KEY=sk-ds-only\n")
    # Mimic load_env() populating os.environ
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-only")

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "deepseek": [{
                    "id": "env-1",
                    "label": "DEEPSEEK_API_KEY",
                    "auth_type": "api_key",
                    "priority": 0,
                    "source": "env:DEEPSEEK_API_KEY",
                    "access_token": "sk-ds-only",
                }]
            },
        },
    )

    from types import SimpleNamespace
    from hermes_cli.auth_commands import auth_remove_command
    auth_remove_command(SimpleNamespace(provider="deepseek", target="1"))

    out = capsys.readouterr().out
    assert "Cleared DEEPSEEK_API_KEY from .env" in out
    assert "still set in your shell environment" not in out
    assert (hermes_home / ".env").read_text().strip() == ""


def test_auth_add_clears_env_suppression_for_provider(tmp_path, monkeypatch):
    """Re-adding a credential via `hermes auth add <provider>` clears any
    env:<VAR> suppression marker — strong signal the user wants auth back.
    Matches the Codex device_code re-link behaviour.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {},
            "suppressed_sources": {"xai": ["env:XAI_API_KEY"]},
        },
    )

    from types import SimpleNamespace
    from hermes_cli.auth import is_source_suppressed
    from hermes_cli.auth_commands import auth_add_command

    assert is_source_suppressed("xai", "env:XAI_API_KEY") is True
    auth_add_command(SimpleNamespace(
        provider="xai", auth_type="api_key",
        api_key="sk-xai-manual", label="manual",
    ))
    assert is_source_suppressed("xai", "env:XAI_API_KEY") is False


def test_seed_from_env_respects_env_suppression(tmp_path, monkeypatch):
    """_seed_from_env() must skip env:<VAR> sources that the user suppressed
    via `hermes auth remove`.  This is the gate that prevents shell-exported
    keys from resurrecting removed credentials.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-shell-export")

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {"xai": ["env:XAI_API_KEY"]},
    }))

    from agent.credential_pool import _seed_from_env

    entries = []
    changed, active = _seed_from_env("xai", entries)
    assert changed is False
    assert entries == []
    assert active == set()


def test_seed_from_env_respects_openrouter_suppression(tmp_path, monkeypatch):
    """OpenRouter is the special-case branch in _seed_from_env; verify it
    honours suppression too.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-shell-export")

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {"openrouter": ["env:OPENROUTER_API_KEY"]},
    }))

    from agent.credential_pool import _seed_from_env

    entries = []
    changed, active = _seed_from_env("openrouter", entries)
    assert changed is False
    assert entries == []
    assert active == set()


# =============================================================================
# Unified credential-source stickiness — every source Hermes reads from has a
# registered RemovalStep in agent.credential_sources, and every seeding path
# gates on is_source_suppressed.  Below: one test per source proving remove
# sticks across a fresh load_pool() call.
# =============================================================================


def test_seed_from_singletons_respects_nous_suppression(tmp_path, monkeypatch):
    """nous device_code must not re-seed from auth.json when suppressed."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {"nous": {"access_token": "tok", "refresh_token": "r", "expires_at": 9999999999}},
        "suppressed_sources": {"nous": ["device_code"]},
    }))

    from agent.credential_pool import _seed_from_singletons
    entries = []
    changed, active = _seed_from_singletons("nous", entries)
    assert changed is False
    assert entries == []
    assert active == set()


def test_seed_from_singletons_respects_copilot_suppression(tmp_path, monkeypatch):
    """copilot gh_cli must not re-seed when suppressed."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {"copilot": ["gh_cli"]},
    }))

    # Stub resolve_copilot_token to return a live token
    import hermes_cli.copilot_auth as ca
    monkeypatch.setattr(ca, "resolve_copilot_token", lambda: ("ghp_fake", "gh auth token"))

    from agent.credential_pool import _seed_from_singletons
    entries = []
    changed, active = _seed_from_singletons("copilot", entries)
    assert changed is False
    assert entries == []
    assert active == set()


def test_seed_from_singletons_respects_qwen_suppression(tmp_path, monkeypatch):
    """qwen-oauth qwen-cli must not re-seed from ~/.qwen/oauth_creds.json when suppressed."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {"qwen-oauth": ["qwen-cli"]},
    }))

    import hermes_cli.auth as ha
    monkeypatch.setattr(ha, "resolve_qwen_runtime_credentials", lambda **kw: {
        "api_key": "tok", "source": "qwen-cli", "base_url": "https://q",
    })

    from agent.credential_pool import _seed_from_singletons
    entries = []
    changed, active = _seed_from_singletons("qwen-oauth", entries)
    assert changed is False
    assert entries == []
    assert active == set()


def test_seed_from_singletons_respects_hermes_pkce_suppression(tmp_path, monkeypatch):
    """anthropic hermes_pkce must not re-seed from ~/.hermes/.anthropic_oauth.json when suppressed."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import yaml
    (hermes_home / "config.yaml").write_text(yaml.dump({"model": {"provider": "anthropic", "model": "claude"}}))
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {"anthropic": ["hermes_pkce"]},
    }))

    # Stub the readers so only hermes_pkce is "available"; claude_code returns None
    import agent.anthropic_adapter as aa
    monkeypatch.setattr(aa, "read_hermes_oauth_credentials", lambda: {
        "accessToken": "tok", "refreshToken": "r", "expiresAt": 9999999999000,
    })
    monkeypatch.setattr(aa, "read_claude_code_credentials", lambda: None)

    from agent.credential_pool import _seed_from_singletons
    entries = []
    changed, active = _seed_from_singletons("anthropic", entries)
    # hermes_pkce suppressed, claude_code returns None → nothing should be seeded
    assert entries == []
    assert "hermes_pkce" not in active


def test_seed_custom_pool_respects_config_suppression(tmp_path, monkeypatch):
    """Custom provider config:<name> source must not re-seed when suppressed."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import yaml
    (hermes_home / "config.yaml").write_text(yaml.dump({
        "model": {},
        "custom_providers": [
            {"name": "my", "base_url": "https://c.example.com", "api_key": "sk-custom"},
        ],
    }))

    from agent.credential_pool import _seed_custom_pool, get_custom_provider_pool_key
    pool_key = get_custom_provider_pool_key("https://c.example.com")

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "suppressed_sources": {pool_key: ["config:my"]},
    }))

    entries = []
    changed, active = _seed_custom_pool(pool_key, entries)
    assert changed is False
    assert entries == []
    assert "config:my" not in active


def test_credential_sources_registry_has_expected_steps():
    """Sanity check — the registry contains the expected RemovalSteps.

    Adding a new credential source is routine, so this is a structural
    invariant check (every step has a description, every step is unique,
    core steps are present) rather than a frozen snapshot. Frozen
    snapshots of catalog-like data violate the AGENTS.md "don't write
    change-detector tests" rule — they break every time someone adds a
    provider.
    """
    from agent.credential_sources import _REGISTRY

    descriptions = [step.description for step in _REGISTRY]
    # No empty descriptions, no duplicates.
    assert all(d for d in descriptions), "Every removal step must have a description"
    assert len(descriptions) == len(set(descriptions)), (
        f"Registry has duplicate step descriptions: {descriptions}"
    )
    # Core steps must be present — these are the ones the rest of the code
    # assumes exist. When deliberately dropping one, update this list.
    required = {
        "gh auth token / COPILOT_GITHUB_TOKEN / GH_TOKEN",
        "Any env-seeded credential (XAI_API_KEY, DEEPSEEK_API_KEY, etc.)",
        "~/.claude/.credentials.json",
        "~/.hermes/.anthropic_oauth.json",
        "auth.json providers.nous",
        "auth.json providers.openai-codex + ~/.codex/auth.json",
        "auth.json providers.minimax-oauth",
        "~/.qwen/oauth_creds.json",
        "Custom provider config.yaml api_key field",
    }
    missing = required - set(descriptions)
    assert not missing, f"Registry missing required steps: {missing}"


def test_credential_sources_find_step_returns_none_for_manual():
    """Manual entries have nothing external to clean up — no step registered."""
    from agent.credential_sources import find_removal_step
    assert find_removal_step("openrouter", "manual") is None
    assert find_removal_step("xai", "manual") is None


def test_credential_sources_find_step_copilot_before_generic_env(tmp_path, monkeypatch):
    """copilot env:GH_TOKEN must dispatch to the copilot step, not the
    generic env-var step.  The copilot step handles the duplicate-source
    problem (same token seeded as both gh_cli and env:<VAR>); the generic
    env step would only suppress one of the variants.
    """
    from agent.credential_sources import find_removal_step

    step = find_removal_step("copilot", "env:GH_TOKEN")
    assert step is not None
    assert "copilot" in step.description.lower() or "gh" in step.description.lower()

    # Generic step still matches any other provider's env var
    step = find_removal_step("xai", "env:XAI_API_KEY")
    assert step is not None
    assert "env-seeded" in step.description.lower()


def test_auth_remove_copilot_suppresses_all_variants(tmp_path, monkeypatch):
    """Removing any copilot source must suppress gh_cli + all env:* variants
    so the duplicate-seed paths don't resurrect the credential.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "copilot": [{
                    "id": "c1",
                    "label": "gh auth token",
                    "auth_type": "api_key",
                    "priority": 0,
                    "source": "gh_cli",
                    "access_token": "ghp_fake",
                }]
            },
        },
    )

    from types import SimpleNamespace
    from hermes_cli.auth import is_source_suppressed
    from hermes_cli.auth_commands import auth_remove_command

    auth_remove_command(SimpleNamespace(provider="copilot", target="1"))

    assert is_source_suppressed("copilot", "gh_cli")
    assert is_source_suppressed("copilot", "env:COPILOT_GITHUB_TOKEN")
    assert is_source_suppressed("copilot", "env:GH_TOKEN")
    assert is_source_suppressed("copilot", "env:GITHUB_TOKEN")


def test_auth_add_clears_all_suppressions_including_non_env(tmp_path, monkeypatch):
    """Re-adding a credential via `hermes auth add <provider>` clears ALL
    suppression markers for the provider, not just env:*.  This matches
    the single "re-engage" semantic — the user wants auth back, period.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {},
            "suppressed_sources": {
                "copilot": ["gh_cli", "env:GH_TOKEN", "env:COPILOT_GITHUB_TOKEN"],
            },
        },
    )

    from types import SimpleNamespace
    from hermes_cli.auth import is_source_suppressed
    from hermes_cli.auth_commands import auth_add_command

    auth_add_command(SimpleNamespace(
        provider="copilot", auth_type="api_key",
        api_key="ghp-manual", label="m",
    ))

    assert not is_source_suppressed("copilot", "gh_cli")
    assert not is_source_suppressed("copilot", "env:GH_TOKEN")
    assert not is_source_suppressed("copilot", "env:COPILOT_GITHUB_TOKEN")


def test_auth_remove_codex_manual_device_code_suppresses_canonical(tmp_path, monkeypatch):
    """Removing a manual:device_code entry (from `hermes auth add openai-codex`)
    must suppress the canonical ``device_code`` key, not ``manual:device_code``.
    The re-seed gate in _seed_from_singletons checks ``device_code``.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {"openai-codex": {"tokens": {"access_token": "t", "refresh_token": "r"}}},
            "credential_pool": {
                "openai-codex": [{
                    "id": "cdx",
                    "label": "manual-codex",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "manual:device_code",
                    "access_token": "t",
                }]
            },
        },
    )

    from types import SimpleNamespace
    from hermes_cli.auth import is_source_suppressed
    from hermes_cli.auth_commands import auth_remove_command

    auth_remove_command(SimpleNamespace(provider="openai-codex", target="1"))
    assert is_source_suppressed("openai-codex", "device_code")
