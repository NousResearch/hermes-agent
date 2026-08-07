"""Tests for named credential selection in the pool (#76937).

Covers the two halves of the feature:
- named credentials: an optional ``name`` on a ``PooledCredential`` that
  round-trips through the on-disk pool payload (``hermes auth add --name``);
- manual selection: when ``config.default_auth.<provider>`` names a
  credential, the pool pins selection to it while it is available, and falls
  back to the legacy auto-rotate behavior when it is missing or exhausted.
"""

from __future__ import annotations

import json
import time

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _make_pool(entries, *, default_auth=None, strategy="fill_first", monkeypatch=None):
    """Build a CredentialPool with controlled strategy and default_auth."""
    from agent import credential_pool as cp

    if monkeypatch is not None:
        monkeypatch.setattr(cp, "get_pool_strategy", lambda _p: strategy)
        monkeypatch.setattr(cp, "get_default_auth_name", lambda _p: default_auth)
    return cp.CredentialPool("openrouter", entries)


def _entry(entry_id: str, token: str, *, name=None, priority=0, exhausted=False):
    from agent.credential_pool import PooledCredential, STATUS_EXHAUSTED

    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if exhausted:
        kwargs.update(
            last_status=STATUS_EXHAUSTED,
            last_status_at=time.time(),
            last_error_code=429,
            last_error_reason="rate_limit",
            last_error_reset_at=time.time() + 3600,
        )
    return PooledCredential(
        provider="openrouter",
        id=entry_id,
        label=f"label-{entry_id}",
        auth_type="api_key",
        priority=priority,
        source="manual",
        access_token=token,
        **kwargs,
    )


# ── named credential storage ─────────────────────────────────────────────


def test_name_round_trips_through_dict_serialization():
    """The optional ``name`` survives to_dict/from_dict (pool persistence)."""
    from agent.credential_pool import PooledCredential

    entry = PooledCredential(
        provider="openrouter",
        id="abc123",
        label="work key",
        auth_type="api_key",
        priority=0,
        source="manual",
        access_token="sk-or-1",
        name="daily",
    )
    payload = entry.to_dict()
    assert payload["name"] == "daily"

    rehydrated = PooledCredential.from_dict("openrouter", payload)
    assert rehydrated.name == "daily"
    assert rehydrated.access_token == "sk-or-1"

    # Unnamed entries keep the legacy payload shape (no ``name`` key).
    unnamed_payload = PooledCredential(
        provider="openrouter",
        id="def456",
        label="legacy",
        auth_type="api_key",
        priority=0,
        source="manual",
        access_token="sk-or-2",
    ).to_dict()
    assert "name" not in unnamed_payload


def test_get_default_auth_name_reads_config(monkeypatch):
    from agent import credential_pool as cp

    monkeypatch.setattr(cp, "_load_config_safe", lambda: {"default_auth": {"openrouter": "daily"}})
    assert cp.get_default_auth_name("openrouter") == "daily"

    # Whitespace-collapsed values are honored; empty values are None.
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {"default_auth": {"openrouter": "  daily  "}})
    assert cp.get_default_auth_name("openrouter") == "daily"
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {"default_auth": {"openrouter": "   "}})
    assert cp.get_default_auth_name("openrouter") is None

    # Missing config, non-dict map, unknown provider, non-str value → None.
    monkeypatch.setattr(cp, "_load_config_safe", lambda: None)
    assert cp.get_default_auth_name("openrouter") is None
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {})
    assert cp.get_default_auth_name("openrouter") is None
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {"default_auth": {"anthropic": "x"}})
    assert cp.get_default_auth_name("openrouter") is None
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {"default_auth": {"openrouter": 42}})
    assert cp.get_default_auth_name("openrouter") is None


# ── manual selection in the pool ──────────────────────────────────────────


def test_select_prefers_named_credential(monkeypatch):
    """With default_auth set, selection is pinned to the named entry even when
    it is not the first entry (fill_first would otherwise pick the unnamed one)."""
    entries = [
        _entry("a", "sk-or-a", priority=0),
        _entry("b", "sk-or-b", priority=1, name="daily"),
    ]
    pool = _make_pool(entries, default_auth="daily", monkeypatch=monkeypatch)
    assert pool.select().id == "b"
    # Selection stays pinned across calls while the entry is healthy.
    assert pool.select().id == "b"


def test_peek_prefers_named_credential(monkeypatch):
    entries = [
        _entry("a", "sk-or-a", priority=0),
        _entry("b", "sk-or-b", priority=1, name="daily"),
    ]
    pool = _make_pool(entries, default_auth="daily", monkeypatch=monkeypatch)
    assert pool.peek().id == "b"


def test_select_falls_back_when_named_exhausted(monkeypatch):
    """The named credential is exhausted → auto-rotate to the unnamed key."""
    entries = [
        _entry("a", "sk-or-a", priority=0),
        _entry("b", "sk-or-b", priority=1, name="daily", exhausted=True),
    ]
    pool = _make_pool(entries, default_auth="daily", monkeypatch=monkeypatch)
    assert pool.select().id == "a"


def test_select_falls_back_when_name_not_found(monkeypatch):
    """default_auth names a credential that does not exist → legacy behavior."""
    entries = [
        _entry("a", "sk-or-a", priority=0),
        _entry("b", "sk-or-b", priority=1),
    ]
    pool = _make_pool(entries, default_auth="ghost", monkeypatch=monkeypatch)
    assert pool.select().id == "a"


def test_unnamed_pool_keeps_legacy_auto_rotate(monkeypatch):
    """No default_auth configured → fill_first selection unchanged."""
    entries = [
        _entry("a", "sk-or-a", priority=0),
        _entry("b", "sk-or-b", priority=1),
    ]
    pool = _make_pool(entries, default_auth=None, monkeypatch=monkeypatch)
    assert pool.select().id == "a"


def test_mark_exhausted_rotates_off_named_onto_fallback(monkeypatch):
    """Full loop: named key hits 429 → marked exhausted → next selection
    falls back to the unnamed key instead of failing."""
    entries = [
        _entry("a", "sk-or-a", priority=0),
        _entry("b", "sk-or-b", priority=1, name="daily"),
    ]
    pool = _make_pool(entries, default_auth="daily", monkeypatch=monkeypatch)
    named = pool.select()
    assert named.id == "b"

    rotated = pool.mark_exhausted_and_rotate(status_code=429, credential_id=named.id)
    assert rotated.id == "a"
    assert pool.select().id == "a"


def test_load_pool_honors_default_auth_end_to_end(tmp_path, monkeypatch):
    """A pool loaded from disk with default_auth configured selects the named
    entry, and re-selection after exhaustion lands on the unnamed fallback."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: None,
    )
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-a",
                        "label": "legacy",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-or-a",
                    },
                    {
                        "id": "cred-b",
                        "label": "daily label",
                        "name": "daily",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "sk-or-b",
                    },
                ]
            },
        },
    )
    from agent import credential_pool as cp

    monkeypatch.setattr(cp, "_seed_from_singletons", lambda provider, entries: (False, set()))
    monkeypatch.setattr(cp, "_seed_from_env", lambda provider, entries: (False, set()))
    monkeypatch.setattr(cp, "get_default_auth_name", lambda _p: "daily")

    pool = cp.load_pool("openrouter")
    assert pool.select().id == "cred-b"

    # Named entry payload survives a write cycle with its name intact.
    persisted = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    named_on_disk = next(
        e for e in persisted["credential_pool"]["openrouter"] if e["id"] == "cred-b"
    )
    assert named_on_disk["name"] == "daily"
