"""Tests for multi-credential runtime pooling and rotation."""

from __future__ import annotations

import json
import time

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def test_fill_first_selection_skips_recently_exhausted_entry(tmp_path, monkeypatch):
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
                        "access_token": "***",
                        "last_status": "exhausted",
                        "last_status_at": time.time(),
                        "last_error_code": 402,
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "***",
                        "last_status": "ok",
                        "last_status_at": None,
                        "last_error_code": None,
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    entry = pool.select()

    assert entry is not None
    assert entry.id == "cred-2"
    assert pool.current().id == "cred-2"


def test_select_clears_expired_exhaustion(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-1",
                        "label": "old",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                        "last_status": "exhausted",
                        "last_status_at": time.time() - 90000,
                        "last_error_code": 402,
                    }
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    entry = pool.select()

    assert entry is not None
    assert entry.last_status == "ok"


def test_round_robin_strategy_rotates_priorities(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "***",
                    },
                ]
            },
        },
    )
    config_path = tmp_path / "hermes" / "config.yaml"
    config_path.write_text("credential_pool_strategies:\n  openrouter: round_robin\n")

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    first = pool.select()
    assert first is not None
    assert first.id == "cred-1"

    reloaded = load_pool("openrouter")
    second = reloaded.select()
    assert second is not None
    assert second.id == "cred-2"


def test_random_strategy_uses_random_choice(tmp_path, monkeypatch):
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
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "***",
                    },
                ]
            },
        },
    )
    config_path = tmp_path / "hermes" / "config.yaml"
    config_path.write_text("credential_pool_strategies:\n  openrouter: random\n")

    monkeypatch.setattr("agent.credential_pool.random.choice", lambda entries: entries[-1])

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    selected = pool.select()
    assert selected is not None
    assert selected.id == "cred-2"



def test_exhausted_entry_resets_after_ttl(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-or-primary",
                        "base_url": "https://openrouter.ai/api/v1",
                        "last_status": "exhausted",
                        "last_status_at": time.time() - 90000,
                        "last_error_code": 429,
                    }
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    entry = pool.select()

    assert entry is not None
    assert entry.id == "cred-1"
    assert entry.last_status == "ok"


def test_exhausted_402_entry_resets_after_one_hour(tmp_path, monkeypatch):
    """402-exhausted credentials recover after 1 hour, not 24."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                        "base_url": "https://openrouter.ai/api/v1",
                        "last_status": "exhausted",
                        "last_status_at": time.time() - 3700,  # ~1h2m ago
                        "last_error_code": 402,
                    }
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    entry = pool.select()

    assert entry is not None
    assert entry.id == "cred-1"
    assert entry.last_status == "ok"


def test_explicit_reset_timestamp_overrides_default_429_ttl(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # Prevent auto-seeding from Codex CLI tokens on the host
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: None,
    )
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "cred-1",
                        "label": "weekly-reset",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:device_code",
                        "access_token": "tok-1",
                        "last_status": "exhausted",
                        "last_status_at": time.time() - 7200,
                        "last_error_code": 429,
                        "last_error_reason": "device_code_exhausted",
                        "last_error_reset_at": time.time() + 7 * 24 * 60 * 60,
                    }
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    assert pool.has_available() is False
    assert pool.select() is None


def test_mark_exhausted_and_rotate_persists_status(tmp_path, monkeypatch):
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

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    assert pool.select().id == "cred-1"

    next_entry = pool.mark_exhausted_and_rotate(status_code=402)

    assert next_entry is not None
    assert next_entry.id == "cred-2"

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    persisted = auth_payload["credential_pool"]["anthropic"][0]
    assert persisted["last_status"] == "exhausted"
    assert persisted["last_error_code"] == 402


def test_try_refresh_current_updates_only_current_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "device_code",
                        "access_token": "access-old",
                        "refresh_token": "refresh-old",
                        "base_url": "https://chatgpt.com/backend-api/codex",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "device_code",
                        "access_token": "access-other",
                        "refresh_token": "refresh-other",
                        "base_url": "https://chatgpt.com/backend-api/codex",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    monkeypatch.setattr(
        "hermes_cli.auth.refresh_codex_oauth_pure",
        lambda access_token, refresh_token, timeout_seconds=20.0: {
            "access_token": "access-new",
            "refresh_token": "refresh-new",
        },
    )

    pool = load_pool("openai-codex")
    current = pool.select()
    assert current.id == "cred-1"

    refreshed = pool.try_refresh_current()

    assert refreshed is not None
    assert refreshed.access_token == "access-new"

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    primary, secondary = auth_payload["credential_pool"]["openai-codex"]
    assert primary["access_token"] == "access-new"
    assert primary["refresh_token"] == "refresh-new"
    assert secondary["access_token"] == "access-other"
    assert secondary["refresh_token"] == "refresh-other"


def test_load_pool_seeds_env_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-seeded")
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    entry = pool.select()

    assert entry is not None
    assert entry.source == "env:OPENROUTER_API_KEY"
    assert entry.access_token == "sk-or-seeded"


def test_load_pool_removes_stale_seeded_env_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "seeded-env",
                        "label": "OPENROUTER_API_KEY",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "env:OPENROUTER_API_KEY",
                        "access_token": "stale-token",
                        "base_url": "https://openrouter.ai/api/v1",
                    }
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")

    assert pool.entries() == []

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert auth_payload["credential_pool"]["openrouter"] == []


def test_load_pool_migrates_nous_provider_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "active_provider": "nous",
            "providers": {
                "nous": {
                    "portal_base_url": "https://portal.example.com",
                    "inference_base_url": "https://inference.example.com/v1",
                    "client_id": "hermes-cli",
                    "token_type": "Bearer",
                    "scope": "inference:mint_agent_key",
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": "2026-03-24T12:00:00+00:00",
                    "agent_key": "agent-key",
                    "agent_key_expires_at": "2026-03-24T13:30:00+00:00",
                }
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("nous")
    entry = pool.select()

    assert entry is not None
    assert entry.source == "device_code"
    assert entry.portal_base_url == "https://portal.example.com"
    assert entry.agent_key == "agent-key"


def test_load_pool_removes_stale_file_backed_singleton_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "seeded-file",
                        "label": "claude-code",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "claude_code",
                        "access_token": "stale-access-token",
                        "refresh_token": "stale-refresh-token",
                        "expires_at_ms": int(time.time() * 1000) + 60_000,
                    }
                ]
            },
        },
    )

    monkeypatch.setattr(
        "agent.anthropic_adapter.read_hermes_oauth_credentials",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: None,
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")

    assert pool.entries() == []

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert auth_payload["credential_pool"]["anthropic"] == []


def test_load_pool_migrates_nous_provider_state_preserves_tls(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "active_provider": "nous",
            "providers": {
                "nous": {
                    "portal_base_url": "https://portal.example.com",
                    "inference_base_url": "https://inference.example.com/v1",
                    "client_id": "hermes-cli",
                    "token_type": "Bearer",
                    "scope": "inference:mint_agent_key",
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": "2026-03-24T12:00:00+00:00",
                    "agent_key": "agent-key",
                    "agent_key_expires_at": "2026-03-24T13:30:00+00:00",
                    "tls": {
                        "insecure": True,
                        "ca_bundle": "/tmp/nous-ca.pem",
                    },
                }
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("nous")
    entry = pool.select()

    assert entry is not None
    assert entry.tls == {
        "insecure": True,
        "ca_bundle": "/tmp/nous-ca.pem",
    }

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert auth_payload["credential_pool"]["nous"][0]["tls"] == {
        "insecure": True,
        "ca_bundle": "/tmp/nous-ca.pem",
    }


def test_singleton_seed_does_not_clobber_manual_oauth_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "manual-1",
                        "label": "manual-pkce",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:hermes_pkce",
                        "access_token": "manual-token",
                        "refresh_token": "manual-refresh",
                        "expires_at_ms": 1711234567000,
                    }
                ]
            },
        },
    )

    monkeypatch.setattr(
        "agent.anthropic_adapter.read_hermes_oauth_credentials",
        lambda: {
            "accessToken": "seeded-token",
            "refreshToken": "seeded-refresh",
            "expiresAt": 1711234999000,
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: None,
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    entries = pool.entries()

    assert len(entries) == 2
    assert {entry.source for entry in entries} == {"manual:hermes_pkce", "hermes_pkce"}


def test_load_pool_prefers_anthropic_env_token_over_file_backed_oauth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_TOKEN", "env-override-token")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    monkeypatch.setattr(
        "agent.anthropic_adapter.read_hermes_oauth_credentials",
        lambda: {
            "accessToken": "file-backed-token",
            "refreshToken": "refresh-token",
            "expiresAt": int(time.time() * 1000) + 3_600_000,
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: None,
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    entry = pool.select()

    assert entry is not None
    assert entry.source == "env:ANTHROPIC_TOKEN"
    assert entry.access_token == "env-override-token"


def test_least_used_strategy_selects_lowest_count(tmp_path, monkeypatch):
    """least_used strategy should select the credential with the lowest request_count."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        "agent.credential_pool.get_pool_strategy",
        lambda _provider: "least_used",
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_env",
        lambda provider, entries: (False, set()),
    )
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "key-a",
                        "label": "heavy",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-or-heavy",
                        "request_count": 100,
                    },
                    {
                        "id": "key-b",
                        "label": "light",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "sk-or-light",
                        "request_count": 10,
                    },
                    {
                        "id": "key-c",
                        "label": "medium",
                        "auth_type": "api_key",
                        "priority": 2,
                        "source": "manual",
                        "access_token": "sk-or-medium",
                        "request_count": 50,
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    entry = pool.select()
    assert entry is not None
    assert entry.id == "key-b"
    assert entry.access_token == "sk-or-light"


def test_thread_safety_concurrent_select(tmp_path, monkeypatch):
    """Concurrent select() calls should not corrupt pool state."""
    import threading as _threading

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        "agent.credential_pool.get_pool_strategy",
        lambda _provider: "round_robin",
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_env",
        lambda provider, entries: (False, set()),
    )
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": f"key-{i}",
                        "label": f"key-{i}",
                        "auth_type": "api_key",
                        "priority": i,
                        "source": "manual",
                        "access_token": f"sk-or-{i}",
                    }
                    for i in range(5)
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    results = []
    errors = []

    def worker():
        try:
            for _ in range(20):
                entry = pool.select()
                if entry:
                    results.append(entry.id)
        except Exception as exc:
            errors.append(exc)

    threads = [_threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(results) == 80  # 4 threads * 20 selects


def test_custom_endpoint_pool_keyed_by_name(tmp_path, monkeypatch):
    """Verify load_pool('custom:together.ai') works and returns entries from auth.json."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # Disable seeding so we only test stored entries
    monkeypatch.setattr(
        "agent.credential_pool._seed_custom_pool",
        lambda pool_key, entries: (False, set()),
    )
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "custom:together.ai": [
                    {
                        "id": "cred-1",
                        "label": "together-key",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-together-xxx",
                        "base_url": "https://api.together.ai/v1",
                    },
                    {
                        "id": "cred-2",
                        "label": "together-key-2",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "sk-together-yyy",
                        "base_url": "https://api.together.ai/v1",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("custom:together.ai")
    assert pool.has_credentials()
    entries = pool.entries()
    assert len(entries) == 2
    assert entries[0].access_token == "sk-together-xxx"
    assert entries[1].access_token == "sk-together-yyy"

    # Select should return the first entry (fill_first default)
    entry = pool.select()
    assert entry is not None
    assert entry.id == "cred-1"


def test_custom_endpoint_pool_seeds_from_config(tmp_path, monkeypatch):
    """Verify seeding from custom_providers api_key in config.yaml."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1})

    # Write config.yaml with a custom_providers entry
    config_path = tmp_path / "hermes" / "config.yaml"
    import yaml
    config_path.write_text(yaml.dump({
        "custom_providers": [
            {
                "name": "Together.ai",
                "base_url": "https://api.together.ai/v1",
                "api_key": "sk-config-seeded",
            }
        ]
    }))

    from agent.credential_pool import load_pool

    pool = load_pool("custom:together.ai")
    assert pool.has_credentials()
    entries = pool.entries()
    assert len(entries) == 1
    assert entries[0].access_token == "sk-config-seeded"
    assert entries[0].source == "config:Together.ai"


def test_custom_endpoint_pool_seeds_from_model_config(tmp_path, monkeypatch):
    """Verify seeding from model.api_key when model.provider=='custom' and base_url matches."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1})

    import yaml
    config_path = tmp_path / "hermes" / "config.yaml"
    config_path.write_text(yaml.dump({
        "custom_providers": [
            {
                "name": "Together.ai",
                "base_url": "https://api.together.ai/v1",
            }
        ],
        "model": {
            "provider": "custom",
            "base_url": "https://api.together.ai/v1",
            "api_key": "sk-model-key",
        },
    }))

    from agent.credential_pool import load_pool

    pool = load_pool("custom:together.ai")
    assert pool.has_credentials()
    entries = pool.entries()
    # Should have the model_config entry
    model_entries = [e for e in entries if e.source == "model_config"]
    assert len(model_entries) == 1
    assert model_entries[0].access_token == "sk-model-key"


def test_custom_pool_does_not_break_existing_providers(tmp_path, monkeypatch):
    """Existing registry providers work exactly as before with custom pool support."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    entry = pool.select()
    assert entry is not None
    assert entry.source == "env:OPENROUTER_API_KEY"
    assert entry.access_token == "sk-or-test"


def test_get_custom_provider_pool_key(tmp_path, monkeypatch):
    """get_custom_provider_pool_key maps base_url to custom:<name> pool key."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    import yaml
    config_path = tmp_path / "hermes" / "config.yaml"
    config_path.write_text(yaml.dump({
        "custom_providers": [
            {
                "name": "Together.ai",
                "base_url": "https://api.together.ai/v1",
                "api_key": "sk-xxx",
            },
            {
                "name": "My Local Server",
                "base_url": "http://localhost:8080/v1",
            },
        ]
    }))

    from agent.credential_pool import get_custom_provider_pool_key

    assert get_custom_provider_pool_key("https://api.together.ai/v1") == "custom:together.ai"
    assert get_custom_provider_pool_key("https://api.together.ai/v1/") == "custom:together.ai"
    assert get_custom_provider_pool_key("http://localhost:8080/v1") == "custom:my-local-server"
    assert get_custom_provider_pool_key("https://unknown.example.com/v1") is None
    assert get_custom_provider_pool_key("") is None


def test_list_custom_pool_providers(tmp_path, monkeypatch):
    """list_custom_pool_providers returns custom: pool keys from auth.json."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "a1",
                        "label": "test",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                    }
                ],
                "custom:together.ai": [
                    {
                        "id": "c1",
                        "label": "together",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                    }
                ],
                "custom:fireworks": [
                    {
                        "id": "c2",
                        "label": "fireworks",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                    }
                ],
                "custom:empty": [],
            },
        },
    )

    from agent.credential_pool import list_custom_pool_providers

    result = list_custom_pool_providers()
    assert result == ["custom:fireworks", "custom:together.ai"]
    # "custom:empty" not included because it's empty



def test_acquire_lease_prefers_unleased_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "***",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    first = pool.acquire_lease()
    second = pool.acquire_lease()

    assert first == "cred-1"
    assert second == "cred-2"
    assert pool._active_leases.get("cred-1", 0) == 1
    assert pool._active_leases.get("cred-2", 0) == 1



def test_release_lease_decrements_counter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "***",
                    }
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    leased = pool.acquire_lease()
    assert leased == "cred-1"
    assert pool._active_leases.get("cred-1", 0) == 1

    pool.release_lease("cred-1")
    assert pool._active_leases.get("cred-1", 0) == 0


def test_load_pool_does_not_seed_claude_code_when_anthropic_not_configured(tmp_path, monkeypatch):
    """Claude Code credentials must not be auto-seeded when the user never selected anthropic."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {}})

    # Claude Code credentials exist on disk
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: {"accessToken": "sk-ant...oken", "refreshToken": "rt", "expiresAt": 9999999999999},
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_hermes_oauth_credentials",
        lambda: None,
    )
    # User configured kimi-coding, NOT anthropic
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured",
        lambda pid: pid == "kimi-coding",
    )

    from agent.credential_pool import load_pool
    pool = load_pool("anthropic")

    # Should NOT have seeded the claude_code entry
    assert pool.entries() == []


def test_load_pool_seeds_copilot_via_gh_auth_token(tmp_path, monkeypatch):
    """Copilot credentials from `gh auth token` should be seeded into the pool."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {}})

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_fake_token_abc123", "gh auth token"),
    )

    from agent.credential_pool import load_pool
    pool = load_pool("copilot")

    assert pool.has_credentials()
    entries = pool.entries()
    assert len(entries) == 1
    assert entries[0].source == "gh_cli"
    assert entries[0].access_token == "gho_fake_token_abc123"
    assert entries[0].base_url == "https://api.githubcopilot.com"


def test_load_pool_does_not_seed_copilot_when_no_token(tmp_path, monkeypatch):
    """Copilot pool should be empty when resolve_copilot_token() returns nothing."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {}})

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("", ""),
    )

    from agent.credential_pool import load_pool
    pool = load_pool("copilot")

    assert not pool.has_credentials()
    assert pool.entries() == []


def test_load_pool_seeds_qwen_oauth_via_cli_tokens(tmp_path, monkeypatch):
    """Qwen OAuth credentials from ~/.qwen/oauth_creds.json should be seeded into the pool."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {}})

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_qwen_runtime_credentials",
        lambda **kw: {
            "provider": "qwen-oauth",
            "base_url": "https://portal.qwen.ai/v1",
            "api_key": "qwen_fake_token_xyz",
            "source": "qwen-cli",
            "expires_at_ms": 1900000000000,
            "auth_file": str(tmp_path / ".qwen" / "oauth_creds.json"),
        },
    )

    from agent.credential_pool import load_pool
    pool = load_pool("qwen-oauth")

    assert pool.has_credentials()
    entries = pool.entries()
    assert len(entries) == 1
    assert entries[0].source == "qwen-cli"
    assert entries[0].access_token == "qwen_fake_token_xyz"


def test_load_pool_does_not_seed_qwen_oauth_when_no_token(tmp_path, monkeypatch):
    """Qwen OAuth pool should be empty when no CLI credentials exist."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {}})

    from hermes_cli.auth import AuthError

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_qwen_runtime_credentials",
        lambda **kw: (_ for _ in ()).throw(
            AuthError("Qwen CLI credentials not found.", provider="qwen-oauth", code="qwen_auth_missing")
        ),
    )

    from agent.credential_pool import load_pool
    pool = load_pool("qwen-oauth")

    assert not pool.has_credentials()
    assert pool.entries() == []
