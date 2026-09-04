"""Tests for multi-credential runtime pooling and rotation."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

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




def test_billing_rotation_marks_all_entries_sharing_failed_key(tmp_path, monkeypatch):
    """A 402 must exhaust every pool entry backed by the same API key.

    Regression: the same key can back more than one pool entry — e.g. an
    explicit pool entry plus a ``model_config`` entry auto-seeded from
    ``model.api_key`` (both carry the identical ``runtime_api_key``).  When
    ``mark_exhausted_and_rotate`` is called with ``api_key_hint`` it matched
    only the *first* such entry, leaving the sibling OK.  ``_select_unlocked()``
    then kept handing back the same depleted key, so the billing-recovery
    ``continue`` loop in the conversation retry path never converged — the
    request hung ~2.5min until the client disconnected, with no 402 ever
    surfaced to the user.  All entries sharing the failed key must be
    exhausted so the pool reaches "no available entries" and the error
    propagates immediately.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    shared_key = "sk-deepseek-shared"
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "custom": [
                    {
                        "id": "cred-explicit",
                        "label": "520555",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": shared_key,
                        "base_url": "https://api.deepseek.com",
                    },
                    {
                        "id": "cred-model-config",
                        "label": "model_config",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": shared_key,
                        "base_url": "https://api.deepseek.com",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_EXHAUSTED

    pool = load_pool("custom")

    # First 402 on the shared key: rotation must NOT hand back a sibling
    # entry that wraps the same depleted key — it must converge to None.
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=402,
        api_key_hint=shared_key,
    )
    assert next_entry is None

    # Both entries are now exhausted (not just the first match).
    statuses = {entry.id: entry.last_status for entry in pool.entries()}
    assert statuses["cred-explicit"] == STATUS_EXHAUSTED
    assert statuses["cred-model-config"] == STATUS_EXHAUSTED


def test_stale_credential_id_prefers_api_key_hint(tmp_path, monkeypatch):
    """#79156: disagreeing credential_id + api_key_hint must mark the key.

    After per-turn env refresh rewrites ``api_key`` without rebinding the
    pool entry id, recovery still passes the stale id of the healthy
    fallback together with the primary key that actually failed. The
    healthy key must not inherit the primary's 429.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-primary",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-ant-api-primary",
                    },
                    {
                        "id": "cred-backup",
                        "label": "backup",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "sk-ant-api-backup",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_EXHAUSTED

    pool = load_pool("anthropic")
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint="sk-ant-api-primary",
        credential_id="cred-backup",  # stale id after env refresh (#79156)
    )

    statuses = {entry.id: entry.last_status for entry in pool.entries()}
    assert statuses["cred-primary"] == STATUS_EXHAUSTED
    assert statuses["cred-backup"] != STATUS_EXHAUSTED
    # Rotation hands the healthy backup (or None if selection prefers next).
    if next_entry is not None:
        assert next_entry.id == "cred-backup"
        assert next_entry.runtime_api_key == "sk-ant-api-backup"


def test_unmatched_api_key_hint_rotates_without_benching_innocent_key(tmp_path, monkeypatch):
    """An api_key_hint matching no entry must not quarantine a healthy key.

    Regression: when the hint was unmatched (key rotated away, or a wrapper
    whose runtime key differs), mark_exhausted_and_rotate fell through to
    current()/_select_unlocked() — on a freshly loaded pool that selects the
    NEXT healthy key and benched it for the full cooldown TTL, punishing an
    innocent credential.  Now it rotates without marking anything.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # Keep the dev machine's live ~/.claude credentials from seeding a
    # claude_code singleton entry into this pool (same isolation as the
    # other anthropic pool tests in this file).
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
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

    from agent.credential_pool import load_pool, STATUS_DEAD, STATUS_EXHAUSTED

    # Freshly loaded pool: current() is None, exactly the shape of the bug.
    pool = load_pool("anthropic")

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint="sk-ant-api-rotated-away",
    )

    # A fresh selection is still handed back so the caller can retry...
    assert next_entry is not None

    # ...but no credential was benched, in memory or on disk.
    assert all(
        entry.last_status not in (STATUS_EXHAUSTED, STATUS_DEAD)
        for entry in pool.entries()
    )
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    for persisted in auth_payload["credential_pool"]["anthropic"]:
        assert persisted.get("last_status") not in (STATUS_EXHAUSTED, STATUS_DEAD)
        assert persisted.get("last_error_code") is None


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


def test_dead_manual_prune_policy_is_profile_scoped(tmp_path, monkeypatch):
    """Each profile applies its own configured DEAD-manual retention policy."""
    from agent.credential_pool import load_pool

    def write_profile(name: str, *, prune: bool, ttl_hours: int) -> Path:
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            "credential_pool:\n"
            f"  prune_dead_manual_entries: {str(prune).lower()}\n"
            f"  dead_manual_prune_ttl_hours: {ttl_hours}\n",
            encoding="utf-8",
        )
        (home / "auth.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "credential_pool": {
                        "openai-codex": [
                            {
                                "id": "dead-manual",
                                "label": "dead-entry",
                                "auth_type": "oauth",
                                "source": "manual:device_code",
                                "access_token": "test-access-token",
                                "refresh_token": "test-refresh-token",
                                "last_status": "dead",
                                "last_status_at": time.time() - (25 * 3600),
                            },
                            {
                                "id": "healthy-manual",
                                "label": "healthy-entry",
                                "auth_type": "oauth",
                                "source": "manual:device_code",
                                "access_token": "test-access-token-2",
                                "refresh_token": "test-refresh-token-2",
                            },
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        return home

    retain_home = write_profile("retain", prune=False, ttl_hours=1)
    prune_home = write_profile("prune", prune=True, ttl_hours=1)

    monkeypatch.setenv("HERMES_HOME", str(retain_home))
    assert load_pool("openai-codex").select().id == "healthy-manual"
    retained = json.loads((retain_home / "auth.json").read_text())
    assert {entry["id"] for entry in retained["credential_pool"]["openai-codex"]} == {
        "dead-manual",
        "healthy-manual",
    }

    monkeypatch.setenv("HERMES_HOME", str(prune_home))
    assert load_pool("openai-codex").select().id == "healthy-manual"
    pruned = json.loads((prune_home / "auth.json").read_text())
    assert [entry["id"] for entry in pruned["credential_pool"]["openai-codex"]] == [
        "healthy-manual"
    ]


def test_dead_manual_prune_settings_reject_nonfinite_ttl(monkeypatch):
    """Invalid retention values fall back safely instead of disabling cleanup."""
    from agent import credential_pool as pool_mod

    monkeypatch.setattr(
        pool_mod,
        "_load_config_safe",
        lambda: {
            "credential_pool": {
                "prune_dead_manual_entries": True,
                "dead_manual_prune_ttl_hours": "nan",
            }
        },
    )

    assert pool_mod.get_dead_manual_prune_settings() == (
        True,
        float(pool_mod.DEAD_MANUAL_PRUNE_TTL_SECONDS),
    )


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


def test_load_pool_collapses_duplicate_env_rows_to_active_key(tmp_path, monkeypatch):
    """One env source is one credential, even if auth.json contains stale duplicates."""
    key = "sk-or-active-main-key"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "current-row",
                        "label": "OPENROUTER_API_KEY",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "env:OPENROUTER_API_KEY",
                    },
                    {
                        "id": "stale-duplicate",
                        "label": "OPENROUTER_API_KEY",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "env:OPENROUTER_API_KEY",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openrouter")

    assert [(entry.id, entry.runtime_api_key) for entry in pool.entries()] == [
        ("current-row", key)
    ]
    persisted = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert [entry["id"] for entry in persisted["credential_pool"]["openrouter"]] == [
        "current-row"
    ]


def test_credential_pool_never_selects_empty_borrowed_entry():
    from agent.credential_pool import CredentialPool, PooledCredential

    pool = CredentialPool(
        "openrouter",
        [
            PooledCredential(
                provider="openrouter",
                id="metadata-only",
                label="OPENROUTER_API_KEY",
                auth_type="api_key",
                priority=0,
                source="env:OPENROUTER_API_KEY",
                access_token="",
            )
        ],
    )

    assert pool.select() is None
    assert pool.acquire_lease() is None


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

    monkeypatch.setattr("agent.anthropic_credentials.read_hermes_oauth_credentials", _fake_pkce)
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", _fake_cc)

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
    monkeypatch.setattr("agent.anthropic_credentials.read_hermes_oauth_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)

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
        "agent.anthropic_credentials.read_hermes_oauth_credentials",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials",
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








    # "custom:empty" not included because it's empty








def test_load_pool_does_not_seed_claude_code_when_anthropic_not_configured(tmp_path, monkeypatch):
    """Claude Code credentials must not be auto-seeded when the user never selected anthropic."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {}})

    # Claude Code credentials exist on disk
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials",
        lambda: {"accessToken": "sk-ant...oken", "refreshToken": "rt", "expiresAt": 9999999999999},
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_hermes_oauth_credentials",
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


def test_load_pool_skips_exchange_for_suppressed_copilot(tmp_path, monkeypatch):
    """A suppressed copilot source must NOT run the token exchange.

    Regression test: the suppression gate used to sit AFTER
    ``get_copilot_api_token`` (which retries 3x with backoff, ~13s worst
    case), so every pool load — model picker open, /model, agent startup —
    burned the full exchange dead time for a source the user had already
    removed with ``hermes auth remove copilot gh_cli``.  The gate must run
    BEFORE the network call.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {},
            "suppressed_sources": {"copilot": ["gh_cli"]},
        },
    )

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_fake_token_abc123", "gh auth token"),
    )

    exchange_called = False

    def _boom(token):
        nonlocal exchange_called
        exchange_called = True
        raise AssertionError("exchange must not run for a suppressed source")

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        _boom,
    )

    from agent.credential_pool import load_pool
    pool = load_pool("copilot")

    assert not exchange_called
    assert not pool.has_credentials()
    assert pool.entries() == []


def test_load_pool_respects_env_var_copilot_suppression(tmp_path, monkeypatch):
    """Suppressing env:GH_TOKEN must gate a GH_TOKEN-sourced token.

    Regression test for the source_name classification: a substring match
    (``"gh" in source.lower()``) classified GH_TOKEN/GITHUB_TOKEN as gh_cli,
    so a user's env-var-specific suppression was silently bypassed and the
    exchange ran anyway.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {},
            "suppressed_sources": {"copilot": ["env:GH_TOKEN"]},
        },
    )

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_fake_token_env", "GH_TOKEN"),
    )

    exchange_called = False

    def _boom(token):
        nonlocal exchange_called
        exchange_called = True
        raise AssertionError("exchange must not run for a suppressed env source")

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        _boom,
    )

    from agent.credential_pool import load_pool
    pool = load_pool("copilot")

    assert not exchange_called
    assert pool.entries() == []


def test_load_pool_gh_cli_suppression_does_not_block_env_tokens(tmp_path, monkeypatch):
    """Suppressing gh_cli must NOT swallow an env-var-sourced token.

    The inverse of the substring bug: GH_TOKEN misclassified as gh_cli meant
    suppressing the CLI path also silently dropped env tokens.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {},
            "suppressed_sources": {"copilot": ["gh_cli"]},
        },
    )

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_fake_token_env", "GH_TOKEN"),
    )
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        lambda token: ("capi_exchanged_token", None),
    )

    from agent.credential_pool import load_pool
    pool = load_pool("copilot")

    assert [e.source for e in pool.entries()] == ["env:GH_TOKEN"]


def test_load_pool_skips_resolve_when_all_copilot_sources_suppressed(tmp_path, monkeypatch):
    """With every copilot source suppressed, resolve_copilot_token (which
    shells out to ``gh auth token``) must not run at all."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from hermes_cli.copilot_auth import COPILOT_ENV_VARS
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {},
            "suppressed_sources": {
                "copilot": ["gh_cli"] + [f"env:{v}" for v in COPILOT_ENV_VARS],
            },
        },
    )

    def _boom():
        raise AssertionError("resolve_copilot_token must not run when all sources are suppressed")

    monkeypatch.setattr("hermes_cli.copilot_auth.resolve_copilot_token", _boom)

    from agent.credential_pool import load_pool
    pool = load_pool("copilot")

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





# ── OpenAI Codex OAuth cross-process sync tests ────────────────────────────










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








def test_persist_preserves_concurrent_disk_only_entry(tmp_path, monkeypatch):
    """Regression for #19566: stale rotation writes keep concurrent entries."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # Block external-credential autodiscovery: a real ~/.claude/.credentials.json
    # on a dev machine would seed an extra claude_code entry and break the
    # exact-id assertions below (passes on CI where no such file exists).
    monkeypatch.setattr("agent.anthropic_credentials.read_hermes_oauth_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-A",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-A",
                    },
                    {
                        "id": "cred-B",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "sk-B",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool
    from hermes_cli.auth import read_credential_pool, write_credential_pool

    pool = load_pool("anthropic")
    assert {entry.id for entry in pool.entries()} == {"cred-A", "cred-B"}

    disk_snapshot = read_credential_pool("anthropic")
    disk_snapshot.append(
        {
            "id": "cred-C",
            "label": "added-concurrently",
            "auth_type": "api_key",
            "priority": 2,
            "source": "manual",
            "access_token": "sk-C",
        }
    )
    write_credential_pool("anthropic", disk_snapshot)

    pool.mark_exhausted_and_rotate(status_code=429)

    final = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    final_ids = [entry["id"] for entry in final["credential_pool"]["anthropic"]]
    assert set(final_ids) == {"cred-A", "cred-B", "cred-C"}
    persisted_a = next(
        entry
        for entry in final["credential_pool"]["anthropic"]
        if entry["id"] == "cred-A"
    )
    assert persisted_a["last_status"] == "exhausted"




# ---------------------------------------------------------------------------
# _sync_anthropic_entry_from_credentials_file — parity fix tests
# ---------------------------------------------------------------------------

def _make_anthropic_claude_code_pool(tmp_path, monkeypatch, *, access_token, refresh_token, expires_at_ms=9_999_999_999_000):
    """Helper: load an Anthropic pool seeded with a single claude_code entry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {}})
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: pid == "anthropic")
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_hermes_oauth_credentials",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials",
        lambda: {"accessToken": access_token, "refreshToken": refresh_token, "expiresAt": expires_at_ms},
    )
    from agent.credential_pool import load_pool
    pool = load_pool("anthropic")
    entry = pool.select()
    assert entry is not None
    assert entry.source == "claude_code"
    return pool, entry






def test_sync_anthropic_entry_tokens_unchanged_no_op(tmp_path, monkeypatch):
    """Sync must be a no-op when credentials file matches the pool entry."""
    pool, entry = _make_anthropic_claude_code_pool(
        tmp_path, monkeypatch,
        access_token="same-access",
        refresh_token="same-refresh",
    )

    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials",
        lambda: {"accessToken": "same-access", "refreshToken": "same-refresh", "expiresAt": 9_999_999_999_000},
    )

    synced = pool._sync_anthropic_entry_from_credentials_file(entry)

    assert synced is entry, "no-op sync must return the original entry object"


def test_sync_anthropic_entry_clears_all_error_fields(tmp_path, monkeypatch):
    """Syncing fresh tokens must clear all six error/status fields on the entry.

    Before the fix, last_error_reason / last_error_message / last_error_reset_at
    were left set, so a previously-exhausted entry could stay stuck even after
    fresh tokens arrived from the credentials file.
    """
    from dataclasses import replace as dc_replace
    from agent.credential_pool import STATUS_EXHAUSTED

    pool, entry = _make_anthropic_claude_code_pool(
        tmp_path, monkeypatch,
        access_token="stale-access",
        refresh_token="stale-refresh",
    )

    now = time.time()
    exhausted = dc_replace(
        entry,
        last_status=STATUS_EXHAUSTED,
        last_status_at=now,
        last_error_code=401,
        last_error_reason="token_expired",
        last_error_message="Access token has expired",
        last_error_reset_at=now + 300,
    )
    pool._replace_entry(entry, exhausted)

    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials",
        lambda: {"accessToken": "fresh-access", "refreshToken": "fresh-refresh", "expiresAt": 9_999_999_999_000},
    )

    synced = pool._sync_anthropic_entry_from_credentials_file(exhausted)

    assert synced is not exhausted
    assert synced.access_token == "fresh-access"
    assert synced.last_status is None
    assert synced.last_status_at is None
    assert synced.last_error_code is None
    assert synced.last_error_reason is None
    assert synced.last_error_message is None
    assert synced.last_error_reset_at is None


def _load_two_ok_pool(tmp_path, monkeypatch):
    """A pool with two OK anthropic entries, current = cred-1."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-1", "label": "primary", "auth_type": "api_key",
                        "priority": 0, "source": "manual", "access_token": "***",
                        "last_status": "ok", "last_status_at": None, "last_error_code": None,
                    },
                    {
                        "id": "cred-2", "label": "secondary", "auth_type": "api_key",
                        "priority": 1, "source": "manual", "access_token": "***",
                        "last_status": "ok", "last_status_at": None, "last_error_code": None,
                    },
                ]
            },
        },
    )
    from agent.credential_pool import load_pool

    return load_pool("anthropic")


def _fresh_entry(pool):
    """A copy of the pool's first entry under a new id, for add_entry()."""
    from dataclasses import replace as dc_replace

    return dc_replace(pool.entries()[0], id="cred-new")


class TestCredentialPoolQueryLocking:
    """Public pool-state methods must run under ``self._lock``.

    ``has_available``/``peek``/``current``/``entries`` all touch
    ``self._entries`` (and ``_available_entries`` even prunes + persists),
    and the management surface (``has_credentials``/``reset_statuses``/
    ``remove_index``/``resolve_target``/``add_entry``) reads or rebinds
    ``self._entries`` and persists auth.json, so they must all hold the
    same lock every mutating entry point uses.  A naive fix would deadlock
    because the lock is non-reentrant and ``peek`` calls ``current`` +
    ``_available_entries``; these tests guard both the no-deadlock and the
    actually-locked properties.
    """

    def test_query_methods_do_not_deadlock(self, tmp_path, monkeypatch):
        pool = _load_two_ok_pool(tmp_path, monkeypatch)
        pool.select()  # set a current entry

        # peek() internally calls current() + _available_entries(); if any of
        # these re-acquired the non-reentrant lock we'd hang here forever.
        assert pool.current() is not None
        assert pool.peek() is not None
        assert pool.has_available() is True
        assert pool.has_credentials() is True
        assert pool.resolve_target("cred-1")[1] is not None
        # (env may seed extra singleton entries; just assert ours are present)
        assert {"cred-1", "cred-2"} <= {e.id for e in pool.entries()}
        # try_refresh_matching's no-hint branch resolves the current entry
        # while already holding the lock — must use _current_unlocked(), not
        # current(), or it deadlocks on the non-reentrant lock (found when
        # rebasing this fix over the #69843 salvage which added the method).
        pool.try_refresh_matching()

    @pytest.mark.parametrize(
        "method,get_args",
        [
            ("has_available", lambda pool: ()),
            ("peek", lambda pool: ()),
            ("current", lambda pool: ()),
            ("entries", lambda pool: ()),
            ("has_credentials", lambda pool: ()),
            ("reset_statuses", lambda pool: ()),
            ("resolve_target", lambda pool: ("cred-1",)),
            ("remove_index", lambda pool: (1,)),
            ("add_entry", lambda pool: (_fresh_entry(pool),)),
        ],
    )
    def test_query_method_acquires_lock(self, tmp_path, monkeypatch, method, get_args):
        import threading

        pool = _load_two_ok_pool(tmp_path, monkeypatch)
        pool.select()
        args = get_args(pool)

        inner = pool._lock

        class _InstrumentedLock:
            """Probe that records acquire attempts, so the test can prove the
            worker actually reached ``self._lock`` before asserting that it
            blocks (a plain timed wait passes spuriously if the worker is
            simply never scheduled)."""

            def __init__(self):
                self.attempted = threading.Event()

            def acquire(self, *args, **kwargs):
                self.attempted.set()
                return inner.acquire(*args, **kwargs)

            def release(self):
                inner.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *exc):
                self.release()

        probe = _InstrumentedLock()
        pool._lock = probe

        done = threading.Event()

        def _call():
            getattr(pool, method)(*args)
            done.set()

        # Hold the real lock (without tripping the probe), then fire the query
        # on another thread. If the method acquires self._lock (as it must),
        # it blocks until we release.
        inner.acquire()
        try:
            worker = threading.Thread(target=_call, daemon=True)
            worker.start()
            assert probe.attempted.wait(timeout=2.0), (
                f"{method}() never attempted to acquire self._lock"
            )
            assert not done.wait(timeout=0.5), (
                f"{method}() returned while the pool lock was held — it is not "
                f"blocking on self._lock"
            )
        finally:
            inner.release()

        assert done.wait(timeout=2.0), f"{method}() did not complete after lock release"
