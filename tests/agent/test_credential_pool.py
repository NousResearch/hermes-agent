"""Tests for multi-credential runtime pooling and rotation."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _jwt_with_claims(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


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


def test_exhausted_401_entry_resets_after_five_minutes(tmp_path, monkeypatch):
    """Transient auth failures should not strand single-key setups for an hour."""
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
                        "last_status_at": time.time() - 310,
                        "last_error_code": 401,
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


def test_token_invalidated_marks_credential_dead(tmp_path, monkeypatch):
    """OpenAI Codex token_invalidated must mark the credential DEAD, not exhausted.

    Regression for #32849: when an OAuth credential is revoked upstream, the
    1-hour exhausted TTL means it re-enters rotation every hour and fails
    again with the same 401 — surfacing as "Failed to generate context
    summary" on context compression.  Terminal OAuth failures should never
    auto-recover.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "cred-dead",
                        "label": "revoked",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:device_code",
                        "access_token": "revoked-at",
                        "refresh_token": "revoked-rt",
                    },
                    {
                        "id": "cred-ok",
                        "label": "healthy",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual:device_code",
                        "access_token": "healthy-at",
                        "refresh_token": "healthy-rt",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_DEAD

    pool = load_pool("openai-codex")
    assert pool.select().id == "cred-dead"

    # Simulate the exact OpenAI Codex 401 token_invalidated response shape.
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=401,
        error_context={
            "reason": "token_invalidated",
            "message": "Your authentication token has been invalidated. Please try signing in again.",
        },
    )

    # Rotation still works — we hand off to the healthy credential.
    assert next_entry is not None
    assert next_entry.id == "cred-ok"

    # The revoked credential is now permanently marked DEAD.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    persisted = auth_payload["credential_pool"]["openai-codex"][0]
    assert persisted["last_status"] == STATUS_DEAD
    assert persisted["last_error_code"] == 401
    assert persisted["last_error_reason"] == "token_invalidated"


def test_dead_credential_never_re_enters_rotation_after_ttl(tmp_path, monkeypatch):
    """A DEAD credential must stay excluded regardless of how much time passes.

    The exhausted TTL clears entries after 5 min (401) / 1 hour (429).
    A DEAD credential has no recovery TTL — it stays dead until either
    (a) an explicit re-auth write-side sync rewrites the tokens, or
    (b) the manual-prune TTL elapses (covered by separate tests below).
    This test verifies the core invariant in the recent-entry window.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # DEAD entry from 2 hours ago — well past the exhausted TTLs (5min/1h)
    # but well within the 24h manual-prune window.
    two_hours_ago = time.time() - (2 * 3600)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "cred-dead",
                        "label": "revoked",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:device_code",
                        "access_token": "revoked-at",
                        "refresh_token": "revoked-rt",
                        "last_status": "dead",
                        "last_status_at": two_hours_ago,
                        "last_error_code": 401,
                        "last_error_reason": "token_invalidated",
                    },
                    {
                        "id": "cred-ok",
                        "label": "healthy",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual:device_code",
                        "access_token": "healthy-at",
                        "refresh_token": "healthy-rt",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_DEAD

    pool = load_pool("openai-codex")
    selected = pool.select()
    # Should skip the dead entry and pick the healthy one — even though
    # the dead entry has priority 0 (would normally be picked first) and
    # plenty of time has passed since it was marked dead.
    assert selected is not None
    assert selected.id == "cred-ok"

    # The DEAD entry is still marked dead on disk — not cleared by TTL.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    dead_entry = next(e for e in auth_payload["credential_pool"]["openai-codex"]
                       if e["id"] == "cred-dead")
    assert dead_entry["last_status"] == STATUS_DEAD


def test_429_rate_limit_still_uses_exhausted_not_dead(tmp_path, monkeypatch):
    """429 rate limits must NOT be treated as terminal.

    They should keep the existing 1-hour TTL cooldown semantics so the
    credential re-enters rotation once the rate window resets.
    """
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
                        "source": "manual:device_code",
                        "access_token": "at-1",
                        "refresh_token": "rt-1",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual:device_code",
                        "access_token": "at-2",
                        "refresh_token": "rt-2",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_EXHAUSTED

    pool = load_pool("openai-codex")
    assert pool.select().id == "cred-1"

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "rate_limit_exceeded", "message": "Rate limit exceeded"},
    )
    assert next_entry is not None
    assert next_entry.id == "cred-2"

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    persisted = auth_payload["credential_pool"]["openai-codex"][0]
    # 429 stays exhausted (transient) — NOT dead.
    assert persisted["last_status"] == STATUS_EXHAUSTED
    assert persisted["last_error_code"] == 429


def test_generic_401_without_terminal_reason_still_uses_exhausted(tmp_path, monkeypatch):
    """A 401 with no specific code/reason should keep TTL semantics.

    Only specific terminal reasons (token_invalidated, token_revoked, etc.)
    transition to DEAD.  A generic 401 might be a transient server-side
    issue worth retrying after the 5-min TTL.
    """
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
                        "source": "manual:device_code",
                        "access_token": "at-1",
                        "refresh_token": "rt-1",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual:device_code",
                        "access_token": "at-2",
                        "refresh_token": "rt-2",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_EXHAUSTED

    pool = load_pool("openai-codex")
    pool.select()

    # 401 with no specific reason — stays exhausted, NOT dead.
    pool.mark_exhausted_and_rotate(
        status_code=401,
        error_context={"message": "Unauthorized"},
    )

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    persisted = auth_payload["credential_pool"]["openai-codex"][0]
    assert persisted["last_status"] == STATUS_EXHAUSTED
    assert persisted["last_error_code"] == 401


def test_dead_manual_entry_pruned_after_24h(tmp_path, monkeypatch):
    """A DEAD manual entry is removed from the pool after the prune TTL.

    Manual entries (``manual:*``) are independent credentials with no
    singleton to re-seed from, so we can clean them up after a quiet
    window without losing recoverability — the user can always re-add
    via ``hermes auth add``.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # DEAD entry from > 24h ago
    long_ago = time.time() - (25 * 3600)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "cred-old-dead",
                        "label": "ancient-dead",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:device_code",
                        "access_token": "stale",
                        "refresh_token": "stale",
                        "last_status": "dead",
                        "last_status_at": long_ago,
                        "last_error_code": 401,
                        "last_error_reason": "token_invalidated",
                    },
                    {
                        "id": "cred-ok",
                        "label": "healthy",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual:device_code",
                        "access_token": "healthy-at",
                        "refresh_token": "healthy-rt",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    # Trigger _available_entries via select; that runs the prune.
    selected = pool.select()
    assert selected is not None
    assert selected.id == "cred-ok"

    # On-disk pool should have the dead entry removed.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    persisted = auth_payload["credential_pool"]["openai-codex"]
    assert len(persisted) == 1
    assert persisted[0]["id"] == "cred-ok"


def test_dead_manual_entry_kept_within_24h(tmp_path, monkeypatch):
    """A DEAD manual entry stays in the pool until the prune TTL elapses.

    Recent DEAD entries are kept so the audit trail (last_error_reason,
    timestamps) remains visible while the user investigates.  They simply
    don't participate in rotation (covered by the DEAD-skip test above).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # DEAD entry from only an hour ago — well within the 24h window
    recent = time.time() - 3600
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "cred-recent-dead",
                        "label": "recent-dead",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual:device_code",
                        "access_token": "stale",
                        "refresh_token": "stale",
                        "last_status": "dead",
                        "last_status_at": recent,
                        "last_error_code": 401,
                        "last_error_reason": "token_invalidated",
                    },
                    {
                        "id": "cred-ok",
                        "label": "healthy",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual:device_code",
                        "access_token": "healthy-at",
                        "refresh_token": "healthy-rt",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_DEAD

    pool = load_pool("openai-codex")
    selected = pool.select()
    assert selected is not None
    assert selected.id == "cred-ok"

    # On-disk pool should still have BOTH entries — recent dead is preserved.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    persisted = auth_payload["credential_pool"]["openai-codex"]
    assert len(persisted) == 2
    dead_entry = next(e for e in persisted if e["id"] == "cred-recent-dead")
    assert dead_entry["last_status"] == STATUS_DEAD


def test_dead_singleton_seeded_entry_not_pruned(tmp_path, monkeypatch):
    """A DEAD ``device_code`` entry must NOT be pruned even after 24h.

    Singleton-seeded entries get re-created by ``_seed_from_singletons`` on
    every ``load_pool()``, so pruning them is pointless — they reappear
    immediately with the same stale singleton tokens.  Keep them visible
    with the DEAD marker so the user knows what's broken.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    long_ago = time.time() - (48 * 3600)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "revoked-at", "refresh_token": "revoked-rt"},
                    "last_refresh": "2026-01-01T00:00:00Z",
                    "auth_mode": "chatgpt",
                },
            },
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "cred-seeded-dead",
                        "label": "seeded-dead",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "device_code",   # singleton-seeded, NOT manual
                        "access_token": "revoked-at",
                        "refresh_token": "revoked-rt",
                        "last_status": "dead",
                        "last_status_at": long_ago,
                        "last_error_code": 401,
                        "last_error_reason": "token_invalidated",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_DEAD

    pool = load_pool("openai-codex")
    # No healthy entry available; select returns None (pool empty for rotation).
    assert pool.select() is None

    # On-disk: the singleton-seeded DEAD entry is preserved.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    persisted = auth_payload["credential_pool"]["openai-codex"]
    assert len(persisted) == 1
    assert persisted[0]["id"] == "cred-seeded-dead"
    assert persisted[0]["last_status"] == STATUS_DEAD


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



def test_load_pool_does_not_persist_env_seeded_secret_value(tmp_path, monkeypatch):
    """Runtime env keys may be used in memory but must not land in auth.json."""
    sentinel = "S3NTINEL_DO_NOT_PERSIST_OPENROUTER"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    entry = pool.select()

    assert entry is not None
    assert entry.source == "env:OPENROUTER_API_KEY"
    assert entry.access_token == sentinel

    auth_text = (tmp_path / "hermes" / "auth.json").read_text()
    assert sentinel not in auth_text
    persisted = json.loads(auth_text)["credential_pool"]["openrouter"][0]
    assert persisted["source"] == "env:OPENROUTER_API_KEY"
    assert persisted["label"] == "OPENROUTER_API_KEY"
    assert persisted["auth_type"] == "api_key"
    assert persisted["priority"] == 0
    assert "access_token" not in persisted
    assert persisted["secret_fingerprint"].startswith("sha256:")



def test_load_pool_persists_bitwarden_origin_metadata_without_secret(tmp_path, monkeypatch):
    """Bitwarden-injected env vars retain source metadata but not raw values."""
    sentinel = "S3NTINEL_DO_NOT_PERSIST_BITWARDEN"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)
    monkeypatch.setattr(
        "hermes_cli.env_loader.get_secret_source",
        lambda env_var: "bitwarden" if env_var == "OPENROUTER_API_KEY" else None,
    )
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    entry = pool.select()

    assert entry is not None
    assert entry.access_token == sentinel
    assert entry.source == "env:OPENROUTER_API_KEY"

    auth_text = (tmp_path / "hermes" / "auth.json").read_text()
    assert sentinel not in auth_text
    persisted = json.loads(auth_text)["credential_pool"]["openrouter"][0]
    assert persisted["source"] == "env:OPENROUTER_API_KEY"
    assert persisted["secret_source"] == "bitwarden"
    assert "access_token" not in persisted



def test_load_pool_sanitizes_legacy_raw_borrowed_entry_when_value_unchanged(tmp_path, monkeypatch):
    """Existing raw env-seeded pool entries are rewritten even if the env value matches."""
    sentinel = "S3NTINEL_DO_NOT_PERSIST_LEGACY_RAW"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "legacy-env",
                        "label": "OPENROUTER_API_KEY",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "env:OPENROUTER_API_KEY",
                        "access_token": sentinel,
                        "base_url": "https://openrouter.ai/api/v1",
                    }
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")
    entry = pool.select()

    assert entry is not None
    assert entry.access_token == sentinel
    auth_text = (tmp_path / "hermes" / "auth.json").read_text()
    assert sentinel not in auth_text
    persisted = json.loads(auth_text)["credential_pool"]["openrouter"][0]
    assert persisted["id"] == "legacy-env"
    assert "access_token" not in persisted
    assert persisted["secret_fingerprint"].startswith("sha256:")



def test_pooled_credential_to_dict_strips_borrowed_secret_fields():
    from agent.credential_pool import PooledCredential

    sentinel = "S3NTINEL_DO_NOT_PERSIST_TO_DICT"
    credential = PooledCredential(
        provider="openrouter",
        id="borrowed-1",
        label="vault-ref",
        auth_type="api_key",
        priority=3,
        source="vault:openrouter/api-key",
        access_token=sentinel,
        refresh_token=f"refresh-{sentinel}",
        agent_key=f"agent-{sentinel}",
        request_count=7,
        last_status="ok",
        extra={
            "api_key": f"extra-{sentinel}",
            "client_secret": f"client-{sentinel}",
            "secret_key": f"secret-key-{sentinel}",
            "authToken": f"auth-token-{sentinel}",
            "refreshToken": f"camel-refresh-{sentinel}",
            "authorization": f"Bearer {sentinel}",
            "tokens": {"access_token": f"nested-{sentinel}"},
            "token_type": "Bearer",
            "scope": "inference",
        },
    )

    payload = credential.to_dict()
    serialized = json.dumps(payload)

    assert sentinel not in serialized
    assert "access_token" not in payload
    assert "refresh_token" not in payload
    assert "agent_key" not in payload
    assert "api_key" not in payload
    assert "client_secret" not in payload
    assert "secret_key" not in payload
    assert "authToken" not in payload
    assert "refreshToken" not in payload
    assert "authorization" not in payload
    assert "tokens" not in payload
    assert payload["source"] == "vault:openrouter/api-key"
    assert payload["label"] == "vault-ref"
    assert payload["request_count"] == 7
    assert payload["token_type"] == "Bearer"
    assert payload["scope"] == "inference"
    assert payload["secret_fingerprint"].startswith("sha256:")



@pytest.mark.parametrize("source", [
    "age://openrouter/api-key",
    "systemd",
    "keyring",
    "1password",
    "pass",
    "sops",
    "future_secret_store:openrouter",
])
def test_borrowed_source_variants_strip_secret_fields(source):
    from agent.credential_pool import PooledCredential

    sentinel = f"S3NTINEL_DO_NOT_PERSIST_{source.replace(':', '_').replace('/', '_')}"
    credential = PooledCredential(
        provider="openrouter",
        id="borrowed-variant",
        label="borrowed",
        auth_type="api_key",
        priority=0,
        source=source,
        access_token=sentinel,
        refresh_token=f"refresh-{sentinel}",
    )

    payload = credential.to_dict()
    serialized = json.dumps(payload)

    assert sentinel not in serialized
    assert "access_token" not in payload
    assert "refresh_token" not in payload
    assert payload["source"] == source
    assert payload["secret_fingerprint"].startswith("sha256:")



def test_load_pool_prunes_stale_borrowed_custom_config_entry(tmp_path, monkeypatch):
    sentinel = "S3NTINEL_DO_NOT_PERSIST_STALE_CUSTOM"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "custom:foo": [
                    {
                        "id": "stale-custom",
                        "label": "Foo",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "config:Foo",
                        "access_token": sentinel,
                        "base_url": "https://foo.example/v1",
                    }
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("custom:foo")

    assert pool.entries() == []
    auth_text = (tmp_path / "hermes" / "auth.json").read_text()
    assert sentinel not in auth_text
    assert json.loads(auth_text)["credential_pool"]["custom:foo"] == []



def test_write_credential_pool_sanitizes_borrowed_payload_at_disk_boundary(tmp_path, monkeypatch):
    """Direct dictionary callers cannot bypass the borrowed-secret guard."""
    sentinel = "S3NTINEL_DO_NOT_PERSIST_DIRECT_WRITE"
    manual_secret = "MANUAL_SECRET_STAYS_PERSISTABLE"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    from hermes_cli.auth import write_credential_pool

    write_credential_pool("openrouter", [
        {
            "id": "borrowed-1",
            "label": "systemd-ref",
            "auth_type": "api_key",
            "priority": 0,
            "source": "systemd://hermes/openrouter",
            "access_token": sentinel,
            "refresh_token": f"refresh-{sentinel}",
            "agent_key": f"agent-{sentinel}",
            "api_key": f"extra-{sentinel}",
        },
        {
            "id": "manual-1",
            "label": "manual",
            "auth_type": "api_key",
            "priority": 1,
            "source": "manual",
            "access_token": manual_secret,
        },
    ])

    auth_text = (tmp_path / "hermes" / "auth.json").read_text()
    assert sentinel not in auth_text
    assert manual_secret in auth_text
    entries = json.loads(auth_text)["credential_pool"]["openrouter"]
    borrowed, manual = entries
    assert borrowed["source"] == "systemd://hermes/openrouter"
    assert "access_token" not in borrowed
    assert "refresh_token" not in borrowed
    assert "agent_key" not in borrowed
    assert "api_key" not in borrowed
    assert borrowed["secret_fingerprint"].startswith("sha256:")
    assert manual["access_token"] == manual_secret



def test_write_credential_pool_treats_unowned_oauth_source_as_borrowed(tmp_path, monkeypatch):
    sentinel = "S3NTINEL_DO_NOT_PERSIST_UNOWNED_OAUTH"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    from hermes_cli.auth import write_credential_pool

    write_credential_pool("openrouter", [
        {
            "id": "unowned-oauth",
            "label": "unowned-oauth",
            "auth_type": "oauth",
            "priority": 0,
            "source": "oauth",
            "access_token": sentinel,
            "refresh_token": f"refresh-{sentinel}",
        }
    ])

    auth_text = (tmp_path / "hermes" / "auth.json").read_text()
    assert sentinel not in auth_text
    persisted = json.loads(auth_text)["credential_pool"]["openrouter"][0]
    assert persisted["source"] == "oauth"
    assert "access_token" not in persisted
    assert "refresh_token" not in persisted
    assert persisted["secret_fingerprint"].startswith("sha256:")



def test_write_credential_pool_preserves_known_provider_owned_oauth_state(tmp_path, monkeypatch):
    sentinel = "PROVIDER_OWNED_DEVICE_CODE_STAYS_PERSISTABLE"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    from hermes_cli.auth import write_credential_pool

    write_credential_pool("nous", [
        {
            "id": "nous-device",
            "label": "device-code",
            "auth_type": "oauth",
            "priority": 0,
            "source": "device_code",
            "access_token": sentinel,
            "refresh_token": f"refresh-{sentinel}",
            "agent_key": f"agent-{sentinel}",
        }
    ])

    persisted = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"]["nous"][0]
    assert persisted["access_token"] == sentinel
    assert persisted["refresh_token"] == f"refresh-{sentinel}"
    assert persisted["agent_key"] == f"agent-{sentinel}"



def test_load_pool_prefers_dotenv_over_stale_os_environ(tmp_path, monkeypatch):
    """Regression for #18254: stale OPENROUTER_API_KEY in os.environ (inherited
    from a parent shell) must NOT shadow the fresh key in ~/.hermes/.env when
    seeding the credential pool. Before the fix, `get_env_value()` preferred
    os.environ and silently wrote the stale value into auth.json, causing
    persistent 401 errors after key rotation.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Simulate the bug: parent shell exported a stale test key
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-STALE-from-shell")

    # User edited ~/.hermes/.env with the fresh key
    (hermes_home / ".env").write_text(
        "OPENROUTER_API_KEY=sk-or-FRESH-from-dotenv\n"
    )

    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    from agent.credential_pool import load_pool
    pool = load_pool("openrouter")
    entry = pool.select()

    assert entry is not None
    assert entry.source == "env:OPENROUTER_API_KEY"
    # The fresh key from .env must win over the stale shell export
    assert entry.access_token == "sk-or-FRESH-from-dotenv", (
        f"Expected .env to win, got {entry.access_token!r}"
    )


def test_load_pool_falls_back_to_os_environ_when_dotenv_empty(tmp_path, monkeypatch):
    """When ~/.hermes/.env does not define OPENROUTER_API_KEY (typical Docker /
    K8s / systemd deployment), seeding must still pick up the key from
    os.environ. Guards against regressions that would break production
    deployments relying on runtime-injected env vars.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-runtime-env")

    # .env exists but does not define OPENROUTER_API_KEY
    (hermes_home / ".env").write_text("SOME_OTHER_VAR=unrelated\n")

    _write_auth_store(tmp_path, {"version": 1, "providers": {}})

    from agent.credential_pool import load_pool
    pool = load_pool("openrouter")
    entry = pool.select()

    assert entry is not None
    assert entry.access_token == "sk-or-from-runtime-env"


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
                    "scope": "inference:invoke",
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


def test_load_pool_mirrors_nous_invoke_jwt_agent_key_runtime_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    expires_at = datetime.fromtimestamp(time.time() + 3600, tz=timezone.utc).isoformat()
    token = _jwt_with_claims({
        "sub": "test-user",
        "scope": ["inference:invoke"],
        "exp": int(time.time() + 3600),
    })
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
                    "scope": "inference:invoke",
                    "access_token": token,
                    "refresh_token": "refresh-token",
                    "expires_at": expires_at,
                    "agent_key": token,
                    "agent_key_expires_at": expires_at,
                }
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("nous")
    entry = pool.select()

    assert entry is not None
    assert entry.source == "device_code"
    assert entry.agent_key == token
    assert entry.runtime_api_key == token

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    pool_entry = auth_payload["credential_pool"]["nous"][0]
    assert pool_entry["agent_key"] == token
    assert pool_entry["agent_key_expires_at"] == expires_at


def test_nous_runtime_api_key_rejects_opaque_agent_key():
    from agent.credential_pool import PooledCredential

    entry = PooledCredential(
        provider="nous",
        id="nous-opaque",
        label="opaque",
        auth_type="oauth",
        priority=0,
        source="device_code",
        access_token="opaque-access-token",
        refresh_token="refresh-token",
        agent_key="opaque-agent-key",
        agent_key_expires_at=datetime.fromtimestamp(
            time.time() + 3600,
            tz=timezone.utc,
        ).isoformat(),
        extra={"scope": "inference:invoke"},
    )

    assert entry.runtime_api_key == ""


def test_nous_runtime_api_key_accepts_sk_prefixed_agent_key():
    """Regression for #47950: Portal ``sk-`` API keys are valid bearer tokens."""
    from agent.credential_pool import PooledCredential

    entry = PooledCredential(
        provider="nous",
        id="nous-sk-key",
        label="api-key",
        auth_type="api_key",
        priority=0,
        source="device_code",
        access_token="sk-nous-test-1234567890abcdef",
        agent_key="sk-nous-test-1234567890abcdef",
        agent_key_expires_at=datetime.fromtimestamp(
            time.time() + 3600,
            tz=timezone.utc,
        ).isoformat(),
        extra={"scope": "inference:invoke"},
    )

    assert entry.runtime_api_key == "sk-nous-test-1234567890abcdef"


def test_nous_runtime_api_key_accepts_sk_access_token_when_agent_key_absent():
    """``sk-`` key stored only in ``access_token`` is also accepted (#47950)."""
    from agent.credential_pool import PooledCredential

    entry = PooledCredential(
        provider="nous",
        id="nous-sk-access",
        label="api-key",
        auth_type="api_key",
        priority=0,
        source="device_code",
        access_token="sk-nous-access-tok-abcdef",
        agent_key=None,
        extra={"scope": "inference:invoke"},
    )

    assert entry.runtime_api_key == "sk-nous-access-tok-abcdef"


def test_nous_pool_terminal_refresh_removes_device_code_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
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
                    "scope": "inference:invoke",
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": "2026-03-24T12:00:00+00:00",
                    "agent_key": "agent-key",
                    "agent_key_expires_at": "2026-03-24T13:30:00+00:00",
                }
            },
        },
    )

    from agent.credential_pool import PooledCredential, load_pool
    from hermes_cli import auth as auth_mod
    from hermes_cli.auth import AuthError

    refresh_calls = {"count": 0}

    def _terminal_refresh_failure(*_args, **_kwargs):
        refresh_calls["count"] += 1
        raise AuthError(
            "Refresh session has been revoked",
            provider="nous",
            code="invalid_grant",
            relogin_required=True,
        )

    pool = load_pool("nous")
    selected = pool.select()
    assert selected is not None
    assert selected.source == "device_code"
    pool.add_entry(PooledCredential.from_dict("nous", {
        "id": "legacy-seeded",
        "source": "manual:device_code",
        "auth_type": "oauth",
        "access_token": "old-access-token",
        "refresh_token": "old-refresh-token",
        "agent_key": "old-agent-key",
    }))
    pool.add_entry(PooledCredential.from_dict("nous", {
        "id": "manual-key",
        "source": "manual",
        "auth_type": "api_key",
        "access_token": "manual-nous-key",
    }))

    monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials", _terminal_refresh_failure)

    assert pool.try_refresh_current() is None

    assert [entry.id for entry in pool.entries()] == ["manual-key"]

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    nous_state = auth_payload["providers"]["nous"]
    assert not nous_state.get("refresh_token")
    assert not nous_state.get("access_token")
    assert not nous_state.get("agent_key")
    assert nous_state["last_auth_error"]["code"] == "invalid_grant"
    assert [entry["id"] for entry in auth_payload["credential_pool"]["nous"]] == ["manual-key"]

    assert pool.try_refresh_current() is None
    assert refresh_calls["count"] == 1


def test_load_pool_removes_nous_device_code_when_singleton_quarantined(tmp_path, monkeypatch):
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
                    "last_auth_error": {"code": "invalid_grant"},
                }
            },
            "credential_pool": {
                "nous": [
                    {
                        "id": "seeded-current",
                        "source": "device_code",
                        "auth_type": "oauth",
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                        "agent_key": "stale-agent",
                    },
                    {
                        "id": "seeded-legacy",
                        "source": "manual:device_code",
                        "auth_type": "oauth",
                        "access_token": "older-stale-access",
                    },
                    {
                        "id": "manual-key",
                        "source": "manual",
                        "auth_type": "api_key",
                        "access_token": "manual-nous-key",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("nous")

    assert [entry.id for entry in pool.entries()] == ["manual-key"]
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert [entry["id"] for entry in auth_payload["credential_pool"]["nous"]] == ["manual-key"]


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
                    "scope": "inference:invoke",
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


def test_load_pool_api_key_path_skips_oauth_autodiscovery(tmp_path, monkeypatch):
    """API-key auth path: autodiscovered OAuth creds must NOT be seeded.

    When the user picks "Anthropic API key" at `hermes setup`,
    `save_anthropic_api_key()` writes ANTHROPIC_API_KEY and zeros
    ANTHROPIC_TOKEN.  That env-var pattern is the explicit signal that the
    user opted into the API-key path and explicitly OUT of the OAuth
    masquerade (Claude Code identity injection + `mcp_` tool-name rewrite
    + claude-cli user-agent).  Autodiscovered Claude Code / Hermes PKCE
    tokens from other tools' credential files must NOT be silently mixed
    into the anthropic pool — otherwise rotation on a 401/429 could flip
    the session onto OAuth credentials mid-conversation.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-explicit-user-key")
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)

    pkce_called = {"n": 0}
    cc_called = {"n": 0}

    def _fake_pkce():
        pkce_called["n"] += 1
        return {
            "accessToken": "sk-ant-oat01-pkce-token",
            "refreshToken": "pkce-refresh",
            "expiresAt": int(time.time() * 1000) + 3_600_000,
        }

    def _fake_cc():
        cc_called["n"] += 1
        return {
            "accessToken": "sk-ant-oat01-claude-code-token",
            "refreshToken": "cc-refresh",
            "expiresAt": int(time.time() * 1000) + 3_600_000,
        }

    monkeypatch.setattr("agent.anthropic_adapter.read_hermes_oauth_credentials", _fake_pkce)
    monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", _fake_cc)

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    sources = {entry.source for entry in pool.entries()}

    # Only the explicit API-key entry should be in the pool.
    assert sources == {"env:ANTHROPIC_API_KEY"}, f"got {sources}"
    # And we should not have even called the autodiscovery readers.
    assert pkce_called["n"] == 0
    assert cc_called["n"] == 0


def test_load_pool_api_key_path_prunes_stale_oauth_entries(tmp_path, monkeypatch):
    """Switching OAuth -> API key must prune stale OAuth entries from auth.json.

    Without this, a user who logs into OAuth (seeding `claude_code` or
    `hermes_pkce` into auth.json) and later switches to the API key at
    `hermes setup` would still have those OAuth entries dormant on disk.
    Pool rotation on a transient 401 could revive them and flip the
    session onto the OAuth masquerade.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-explicit-user-key")
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    # Plant a stale claude_code entry in the on-disk pool (as if a previous
    # OAuth session seeded it).
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "stale1",
                        "source": "claude_code",
                        "auth_type": "oauth",
                        "access_token": "sk-ant-oat01-stale-claude-code",
                        "refresh_token": "stale-refresh",
                        "expires_at_ms": int(time.time() * 1000) + 3_600_000,
                        "priority": 0,
                        "label": "stale-claude-code",
                        "request_count": 0,
                    },
                ],
            },
        },
    )
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)
    monkeypatch.setattr("agent.anthropic_adapter.read_hermes_oauth_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    sources = {entry.source for entry in pool.entries()}

    # Stale claude_code entry must be gone, API key must be present.
    assert "claude_code" not in sources
    assert "env:ANTHROPIC_API_KEY" in sources


def test_load_pool_oauth_path_still_autodiscovers(tmp_path, monkeypatch):
    """OAuth path: ANTHROPIC_TOKEN set, autodiscovery still fires.

    Regression guard: the API-key gate must not affect users who chose the
    OAuth path at `hermes setup`.  When ANTHROPIC_TOKEN is set (and
    ANTHROPIC_API_KEY is empty), autodiscovered Claude Code creds should
    still be seeded into the pool as before.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-explicit-oauth-token")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "providers": {}})
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)

    monkeypatch.setattr(
        "agent.anthropic_adapter.read_hermes_oauth_credentials",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: {
            "accessToken": "sk-ant-oat01-autodiscovered-cc",
            "refreshToken": "cc-refresh",
            "expiresAt": int(time.time() * 1000) + 3_600_000,
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    sources = {entry.source for entry in pool.entries()}

    # Both env OAuth token and autodiscovered Claude Code creds should be there.
    assert "env:ANTHROPIC_TOKEN" in sources
    assert "claude_code" in sources


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


def test_get_custom_provider_pool_key_prefers_name_over_base_url(tmp_path, monkeypatch):
    """When two custom providers share the same base_url, provider_name resolves to the correct one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    import yaml
    config_path = tmp_path / "hermes" / "config.yaml"
    config_path.write_text(yaml.dump({
        "custom_providers": [
            {
                "name": "provider-a",
                "base_url": "http://gateway:8080/v1",
                "api_key": "sk-aaa",
            },
            {
                "name": "provider-b",
                "base_url": "http://gateway:8080/v1",
                "api_key": "sk-bbb",
            },
        ]
    }))

    from agent.credential_pool import get_custom_provider_pool_key

    # Without provider_name, first match wins (backward compatible)
    assert get_custom_provider_pool_key("http://gateway:8080/v1") == "custom:provider-a"

    # With provider_name, exact name match wins regardless of order
    assert get_custom_provider_pool_key("http://gateway:8080/v1", provider_name="provider-b") == "custom:provider-b"
    assert get_custom_provider_pool_key("http://gateway:8080/v1", provider_name="provider-a") == "custom:provider-a"

    # Name match with non-matching base_url still works via fallback
    assert get_custom_provider_pool_key("http://gateway:8080/v1", provider_name="nonexistent") == "custom:provider-a"

    # Empty provider_name is same as None (backward compatible)
    assert get_custom_provider_pool_key("http://gateway:8080/v1", provider_name="") == "custom:provider-a"


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


def test_nous_seed_from_singletons_preserves_obtained_at_timestamps(tmp_path, monkeypatch):
    """Regression test for #15099 secondary issue.

    When ``_seed_from_singletons`` materialises a device_code pool entry from
    the ``providers.nous`` singleton, it must carry the mint/refresh
    timestamps (``obtained_at``, ``agent_key_obtained_at``, ``expires_in``,
    etc.) into the pool entry.  Without them, freshness-sensitive consumers
    (self-heal hooks, pool pruning by age) treat just-minted credentials as
    older than they actually are and evict them.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "providers": {
                "nous": {
                    "access_token": "at_XXXXXXXX",
                    "refresh_token": "rt_YYYYYYYY",
                    "client_id": "hermes-cli",
                    "portal_base_url": "https://portal.nousresearch.com",
                    "inference_base_url": "https://inference.nousresearch.com/v1",
                    "token_type": "Bearer",
                    "scope": "openid profile",
                    "obtained_at": "2026-04-24T10:00:00+00:00",
                    "expires_at": "2026-04-24T11:00:00+00:00",
                    "expires_in": 3600,
                    "agent_key": "sk-nous-AAAA",
                    "agent_key_id": "ak_123",
                    "agent_key_expires_at": "2026-04-25T10:00:00+00:00",
                    "agent_key_expires_in": 86400,
                    "agent_key_reused": False,
                    "agent_key_obtained_at": "2026-04-24T10:00:05+00:00",
                    "tls": {"insecure": False, "ca_bundle": None},
                },
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("nous")
    entries = pool.entries()

    device_entries = [e for e in entries if e.source == "device_code"]
    assert len(device_entries) == 1, f"expected single device_code entry; got {len(device_entries)}"
    e = device_entries[0]

    # Direct dataclass fields — must survive the singleton → pool copy.
    assert e.access_token == "at_XXXXXXXX"
    assert e.refresh_token == "rt_YYYYYYYY"
    assert e.expires_at == "2026-04-24T11:00:00+00:00"
    assert e.agent_key == "sk-nous-AAAA"
    assert e.agent_key_expires_at == "2026-04-25T10:00:00+00:00"

    # Extra fields — this is what regressed.  These must be carried through
    # via ``extra`` dict or __getattr__, NOT silently dropped.
    assert e.obtained_at == "2026-04-24T10:00:00+00:00", (
        f"obtained_at was dropped during seed; got {e.obtained_at!r}. This breaks "
        f"downstream pool-freshness consumers (#15099)."
    )
    assert e.agent_key_obtained_at == "2026-04-24T10:00:05+00:00"
    assert e.expires_in == 3600
    assert e.agent_key_id == "ak_123"
    assert e.agent_key_expires_in == 86400
    assert e.agent_key_reused is False


class TestLeastUsedStrategy:
    """Regression: least_used strategy must increment request_count on select."""

    def test_request_count_increments(self):
        """Each select() call should increment the chosen entry's request_count."""
        from unittest.mock import patch as _patch
        from agent.credential_pool import CredentialPool, PooledCredential, STRATEGY_LEAST_USED

        entries = [
            PooledCredential(provider="test", id="a", label="a", auth_type="api_key",
                             source="a", access_token="tok-a", priority=0, request_count=0),
            PooledCredential(provider="test", id="b", label="b", auth_type="api_key",
                             source="b", access_token="tok-b", priority=1, request_count=0),
        ]
        with _patch("agent.credential_pool.get_pool_strategy", return_value=STRATEGY_LEAST_USED):
            pool = CredentialPool("test", entries)

        # First select should pick entry with lowest count (both 0 → first)
        e1 = pool.select()
        assert e1 is not None
        count_after_first = e1.request_count
        assert count_after_first == 1, f"Expected 1 after first select, got {count_after_first}"

        # Second select should pick the OTHER entry (now has lower count)
        e2 = pool.select()
        assert e2 is not None
        assert e2.id != e1.id or e2.request_count == 2, (
            "least_used should alternate or increment"
        )


# ── PR #10160 salvage: Nous OAuth cross-process sync tests ─────────────────

def test_sync_nous_entry_from_auth_store_adopts_newer_tokens(tmp_path, monkeypatch):
    """When auth.json has a newer refresh token, the pool entry should adopt it."""
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
                    "scope": "inference:invoke",
                    "access_token": "access-OLD",
                    "refresh_token": "refresh-OLD",
                    "expires_at": "2026-03-24T12:00:00+00:00",
                    "agent_key": "agent-key-OLD",
                    "agent_key_expires_at": "2026-03-24T13:30:00+00:00",
                }
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("nous")
    entry = pool.select()
    assert entry is not None
    assert entry.refresh_token == "refresh-OLD"

    # Simulate another process refreshing the token in auth.json
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
                    "scope": "inference:invoke",
                    "access_token": "access-NEW",
                    "refresh_token": "refresh-NEW",
                    "expires_at": "2026-03-24T12:30:00+00:00",
                    "agent_key": "agent-key-NEW",
                    "agent_key_expires_at": "2026-03-24T14:00:00+00:00",
                }
            },
        },
    )

    synced = pool._sync_nous_entry_from_auth_store(entry)
    assert synced is not entry
    assert synced.access_token == "access-NEW"
    assert synced.refresh_token == "refresh-NEW"
    assert synced.agent_key == "agent-key-NEW"
    assert synced.agent_key_expires_at == "2026-03-24T14:00:00+00:00"

def test_sync_nous_entry_noop_when_tokens_match(tmp_path, monkeypatch):
    """When auth.json has the same refresh token, sync should be a no-op."""
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
                    "scope": "inference:invoke",
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

    synced = pool._sync_nous_entry_from_auth_store(entry)
    assert synced is entry

def test_nous_exhausted_entry_recovers_via_auth_store_sync(tmp_path, monkeypatch):
    """An exhausted Nous entry should recover when auth.json has newer tokens."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from agent.credential_pool import load_pool, STATUS_EXHAUSTED
    from dataclasses import replace as dc_replace

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
                    "scope": "inference:invoke",
                    "access_token": "access-OLD",
                    "refresh_token": "refresh-OLD",
                    "expires_at": "2026-03-24T12:00:00+00:00",
                    "agent_key": "agent-key",
                    "agent_key_expires_at": "2026-03-24T13:30:00+00:00",
                }
            },
        },
    )

    pool = load_pool("nous")
    entry = pool.select()
    assert entry is not None

    # Mark entry as exhausted (simulating a failed refresh)
    exhausted = dc_replace(
        entry,
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_code=401,
    )
    pool._replace_entry(entry, exhausted)
    pool._persist()

    # Simulate another process having successfully refreshed
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
                    "scope": "inference:invoke",
                    "access_token": "access-FRESH",
                    "refresh_token": "refresh-FRESH",
                    "expires_at": "2026-03-24T12:30:00+00:00",
                    "agent_key": "agent-key-FRESH",
                    "agent_key_expires_at": "2026-03-24T14:00:00+00:00",
                }
            },
        },
    )

    available = pool._available_entries(clear_expired=True)
    assert len(available) == 1
    assert available[0].refresh_token == "refresh-FRESH"
    assert available[0].last_status is None


# ── OpenAI Codex OAuth cross-process sync tests ────────────────────────────

def _codex_auth_store(access: str, refresh: str) -> dict:
    return {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access,
                    "refresh_token": refresh,
                    "id_token": "id-" + access,
                },
                "last_refresh": "2026-04-28T00:00:00Z",
            }
        },
    }


def test_sync_codex_entry_from_auth_store_adopts_newer_tokens(tmp_path, monkeypatch):
    """When auth.json has newer Codex tokens, the pool entry should adopt them."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, _codex_auth_store("access-OLD", "refresh-OLD"))

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entry = pool.select()
    assert entry is not None
    assert entry.access_token == "access-OLD"
    assert entry.refresh_token == "refresh-OLD"

    # Simulate `hermes auth openai-codex` replacing the token pair on disk.
    _write_auth_store(tmp_path, _codex_auth_store("access-NEW", "refresh-NEW"))

    synced = pool._sync_codex_entry_from_auth_store(entry)
    assert synced is not entry
    assert synced.access_token == "access-NEW"
    assert synced.refresh_token == "refresh-NEW"
    assert synced.last_status is None
    assert synced.last_error_code is None
    assert synced.last_error_reset_at is None


def test_sync_codex_entry_noop_when_tokens_match(tmp_path, monkeypatch):
    """When auth.json has the same tokens, sync should be a no-op."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, _codex_auth_store("access-same", "refresh-same"))

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entry = pool.select()
    assert entry is not None

    synced = pool._sync_codex_entry_from_auth_store(entry)
    assert synced is entry


def test_codex_exhausted_entry_recovers_via_auth_store_sync(tmp_path, monkeypatch):
    """An exhausted Codex entry should recover when auth.json has newer tokens.

    Reproduces the Discord report (p1aceho1der, Apr 2026): after a Codex
    rate-limit reset the user ran `hermes model` to reauth, but the pool
    entry stayed marked EXHAUSTED with last_error_reset_at many hours in
    the future — so `_available_entries` kept returning empty and every
    request failed with "no available entries (all exhausted or empty)".
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from agent.credential_pool import load_pool, STATUS_EXHAUSTED
    from dataclasses import replace as dc_replace

    _write_auth_store(tmp_path, _codex_auth_store("access-OLD", "refresh-OLD"))

    pool = load_pool("openai-codex")
    entry = pool.select()
    assert entry is not None

    # Mark entry as exhausted with last_error_reset_at one hour in the
    # future (Codex 429 weekly-window pattern).
    now = time.time()
    exhausted = dc_replace(
        entry,
        last_status=STATUS_EXHAUSTED,
        last_status_at=now,
        last_error_code=429,
        last_error_reset_at=now + 3600,
    )
    pool._replace_entry(entry, exhausted)
    pool._persist()

    # Sanity: before the reauth, _available_entries refuses to return
    # this entry because last_error_reset_at is in the future.
    # (clear_expired would only clear it AFTER exhausted_until elapsed.)
    available_before = pool._available_entries(clear_expired=True, refresh=False)
    assert available_before == []

    # Simulate `hermes model` / `hermes auth` refreshing the tokens.
    _write_auth_store(tmp_path, _codex_auth_store("access-FRESH", "refresh-FRESH"))

    available = pool._available_entries(clear_expired=True, refresh=False)
    assert len(available) == 1
    assert available[0].access_token == "access-FRESH"
    assert available[0].refresh_token == "refresh-FRESH"
    assert available[0].last_status is None
    assert available[0].last_error_reset_at is None


def test_codex_exhausted_entry_stays_stuck_without_auth_store_update(tmp_path, monkeypatch):
    """Regression guard: if auth.json tokens haven't changed, the exhausted
    entry must stay stuck behind its reset window — sync must not spuriously
    clear status just because the entry is STATUS_EXHAUSTED."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from agent.credential_pool import load_pool, STATUS_EXHAUSTED
    from dataclasses import replace as dc_replace

    _write_auth_store(tmp_path, _codex_auth_store("access-same", "refresh-same"))

    pool = load_pool("openai-codex")
    entry = pool.select()
    assert entry is not None

    now = time.time()
    exhausted = dc_replace(
        entry,
        last_status=STATUS_EXHAUSTED,
        last_status_at=now,
        last_error_code=429,
        last_error_reset_at=now + 3600,
    )
    pool._replace_entry(entry, exhausted)
    pool._persist()

    # auth.json unchanged → sync returns same entry → exhausted_until check
    # still skips it.
    available = pool._available_entries(clear_expired=True, refresh=False)
    assert available == []


# ---------------------------------------------------------------------------
# xAI OAuth terminal error quarantine
# ---------------------------------------------------------------------------


def _xai_auth_store(access_token: str, refresh_token: str) -> dict:
    return {
        "version": 1,
        "active_provider": "xai-oauth",
        "providers": {
            "xai-oauth": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "discovery": {"token_endpoint": "https://accounts.x.ai/oauth2/token"},
                "redirect_uri": "http://localhost:12345/callback",
            }
        },
    }


def test_is_terminal_xai_oauth_refresh_error():
    from hermes_cli.auth import AuthError, _is_terminal_xai_oauth_refresh_error

    assert _is_terminal_xai_oauth_refresh_error(
        AuthError("Refresh failed", provider="xai-oauth", code="xai_refresh_failed", relogin_required=True)
    )
    assert _is_terminal_xai_oauth_refresh_error(
        AuthError("No token", provider="xai-oauth", code="xai_auth_missing_refresh_token", relogin_required=True)
    )
    # transient 429/5xx: relogin_required=False → not terminal
    assert not _is_terminal_xai_oauth_refresh_error(
        AuthError("Rate limit", provider="xai-oauth", code="xai_refresh_failed", relogin_required=False)
    )
    # Nous error does not trigger xAI check
    assert not _is_terminal_xai_oauth_refresh_error(
        AuthError("Revoked", provider="nous", code="invalid_grant", relogin_required=True)
    )
    # Generic exception
    assert not _is_terminal_xai_oauth_refresh_error(ValueError("oops"))


def test_xai_oauth_terminal_refresh_clears_auth_json_and_removes_pool_entries(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_OAUTH_ACCESS_TOKEN", raising=False)

    _write_auth_store(tmp_path, _xai_auth_store("old-access-token", "old-refresh-token"))

    from agent.credential_pool import PooledCredential, load_pool
    import hermes_cli.auth as auth_mod
    from hermes_cli.auth import AuthError

    pool = load_pool("xai-oauth")
    selected = pool.select()
    assert selected is not None
    assert selected.source == "loopback_pkce"

    # Add a manual API-key entry that must survive the quarantine.
    pool.add_entry(PooledCredential.from_dict("xai-oauth", {
        "id": "manual-key",
        "source": "manual",
        "auth_type": "api_key",
        "access_token": "manual-xai-key",
    }))

    refresh_calls = {"count": 0}

    def _terminal_refresh_failure(*_args, **_kwargs):
        refresh_calls["count"] += 1
        raise AuthError(
            "Refresh session has been revoked",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr(auth_mod, "refresh_xai_oauth_pure", _terminal_refresh_failure)

    assert pool.try_refresh_current() is None

    # Only the manual entry survives.
    assert [entry.id for entry in pool.entries()] == ["manual-key"]

    # Auth.json tokens must be cleared.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    xai_state = auth_payload["providers"]["xai-oauth"]
    tokens = xai_state.get("tokens", {})
    assert not tokens.get("access_token")
    assert not tokens.get("refresh_token")
    assert xai_state["last_auth_error"]["code"] == "xai_refresh_failed"
    assert xai_state["last_auth_error"]["relogin_required"] is True

    # Persisted pool must also have only the manual entry.
    assert [entry["id"] for entry in auth_payload["credential_pool"]["xai-oauth"]] == ["manual-key"]

    # A second try_refresh_current must not call refresh_xai_oauth_pure again
    # (pool is now empty of loopback entries and current is None).
    assert pool.try_refresh_current() is None
    assert refresh_calls["count"] == 1


def test_xai_oauth_nonterminal_refresh_does_not_quarantine(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_OAUTH_ACCESS_TOKEN", raising=False)

    _write_auth_store(tmp_path, _xai_auth_store("old-access-token", "old-refresh-token"))

    from agent.credential_pool import load_pool
    import hermes_cli.auth as auth_mod
    from hermes_cli.auth import AuthError

    pool = load_pool("xai-oauth")
    assert pool.select() is not None

    def _transient_failure(*_args, **_kwargs):
        raise AuthError(
            "Rate limited",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=False,
        )

    monkeypatch.setattr(auth_mod, "refresh_xai_oauth_pure", _transient_failure)

    pool.try_refresh_current()

    # Tokens must NOT be cleared from auth.json.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    tokens = auth_payload["providers"]["xai-oauth"].get("tokens", {})
    assert tokens.get("access_token") == "old-access-token"
    assert tokens.get("refresh_token") == "old-refresh-token"


# ---------------------------------------------------------------------------
# Codex OAuth terminal error quarantine
# ---------------------------------------------------------------------------


def _codex_auth_store(access_token: str, refresh_token: str) -> dict:
    return {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
            }
        },
    }


def test_is_terminal_codex_oauth_refresh_error():
    from hermes_cli.auth import AuthError, _is_terminal_codex_oauth_refresh_error

    assert _is_terminal_codex_oauth_refresh_error(
        AuthError("Refresh failed", provider="openai-codex", code="codex_refresh_failed", relogin_required=True)
    )
    assert _is_terminal_codex_oauth_refresh_error(
        AuthError("No token", provider="openai-codex", code="codex_auth_missing_refresh_token", relogin_required=True)
    )
    assert _is_terminal_codex_oauth_refresh_error(
        AuthError("Revoked", provider="openai-codex", code="invalid_grant", relogin_required=True)
    )
    assert _is_terminal_codex_oauth_refresh_error(
        AuthError("Reused", provider="openai-codex", code="refresh_token_reused", relogin_required=True)
    )
    # transient 429/5xx: relogin_required=False -> not terminal
    assert not _is_terminal_codex_oauth_refresh_error(
        AuthError("Rate limit", provider="openai-codex", code="codex_refresh_failed", relogin_required=False)
    )
    # xAI error does not trigger Codex check
    assert not _is_terminal_codex_oauth_refresh_error(
        AuthError("Revoked", provider="xai-oauth", code="xai_refresh_failed", relogin_required=True)
    )
    # Generic exception
    assert not _is_terminal_codex_oauth_refresh_error(ValueError("oops"))


def test_codex_oauth_terminal_refresh_clears_auth_json_and_removes_pool_entries(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_OAUTH_ACCESS_TOKEN", raising=False)

    _write_auth_store(tmp_path, _codex_auth_store("old-access-token", "old-refresh-token"))

    from agent.credential_pool import PooledCredential, load_pool
    import hermes_cli.auth as auth_mod
    from hermes_cli.auth import AuthError

    pool = load_pool("openai-codex")
    selected = pool.select()
    assert selected is not None
    assert selected.source == "device_code"

    # Add a manual API-key entry that must survive the quarantine.
    pool.add_entry(PooledCredential.from_dict("openai-codex", {
        "id": "manual-key",
        "source": "manual",
        "auth_type": "api_key",
        "access_token": "manual-codex-key",
    }))

    refresh_calls = {"count": 0}

    def _terminal_refresh_failure(*_args, **_kwargs):
        refresh_calls["count"] += 1
        raise AuthError(
            "Refresh session has been revoked",
            provider="openai-codex",
            code="codex_refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _terminal_refresh_failure)

    assert pool.try_refresh_current() is None

    # Only the manual entry survives.
    assert [entry.id for entry in pool.entries()] == ["manual-key"]

    # Auth.json tokens must be cleared.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    codex_state = auth_payload["providers"]["openai-codex"]
    tokens = codex_state.get("tokens", {})
    assert not tokens.get("access_token")
    assert not tokens.get("refresh_token")
    assert codex_state["last_auth_error"]["code"] == "codex_refresh_failed"
    assert codex_state["last_auth_error"]["relogin_required"] is True

    # Persisted pool must also have only the manual entry.
    assert [entry["id"] for entry in auth_payload["credential_pool"]["openai-codex"]] == ["manual-key"]

    # A second try_refresh_current must not call refresh_codex_oauth_pure again.
    assert pool.try_refresh_current() is None
    assert refresh_calls["count"] == 1


def test_codex_oauth_nonterminal_refresh_does_not_quarantine(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_OAUTH_ACCESS_TOKEN", raising=False)

    _write_auth_store(tmp_path, _codex_auth_store("old-access-token", "old-refresh-token"))

    from agent.credential_pool import load_pool
    import hermes_cli.auth as auth_mod
    from hermes_cli.auth import AuthError

    pool = load_pool("openai-codex")
    assert pool.select() is not None

    def _transient_failure(*_args, **_kwargs):
        raise AuthError(
            "Rate limited",
            provider="openai-codex",
            code="codex_refresh_failed",
            relogin_required=False,
        )

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _transient_failure)

    pool.try_refresh_current()

    # Tokens must NOT be cleared from auth.json.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    tokens = auth_payload["providers"]["openai-codex"].get("tokens", {})
    assert tokens.get("access_token") == "old-access-token"
    assert tokens.get("refresh_token") == "old-refresh-token"
