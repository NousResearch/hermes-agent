"""Regression tests: per-request Anthropic credential refresh must honour the
pool's selected entry, and ``load_pool()`` must not undo user priority
reorders between file-backed OAuth accounts.

Background (2026-10-30): with a pool of [hermes_pkce_shared p0, claude_code p1]
every request still went out with the claude_code token because
``_try_refresh_anthropic_client_credentials`` consulted the singleton
``resolve_anthropic_token()`` (which prefers ~/.claude/.credentials.json) and
rebuilt the client with that token, overriding ``pool.select()``.  On top of
that ``_normalize_pool_priorities`` sorted seeded entries by source first, so
``pool-priority-shift.py night`` was reverted on the next ``load_pool()``.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


NOW_MS = int(time.time() * 1000)


def _oauth_entry(idx, label, source, access, priority):
    return {
        "id": f"cred-{idx}",
        "label": label,
        "auth_type": "oauth",
        "priority": priority,
        "source": source,
        "access_token": access,
        "refresh_token": f"{access}-refresh",
        "expires_at_ms": NOW_MS + 3_600_000,
    }


def _make_pool(tmp_path, monkeypatch, entries):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"anthropic": entries}})
    )
    from agent.credential_pool import load_pool

    return load_pool("anthropic")


@pytest.fixture
def no_autodiscovery(monkeypatch):
    """Keep ~/.claude/.credentials.json / shared PKCE files out of the pool."""
    # claude_code rows are file-backed singletons: load_pool() prunes them when
    # the backing file is gone, so back the seeded row with a stub file.
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: {
            "accessToken": "main-tok",
            "refreshToken": "main-tok-refresh",
            "expiresAt": NOW_MS + 3_600_000,
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_hermes_oauth_credentials", lambda: None
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_shared_hermes_oauth_credentials",
        lambda label=None: {
            "accessToken": "shared-tok",
            "refreshToken": "shared-tok-refresh",
            "expiresAt": NOW_MS + 3_600_000,
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.list_shared_hermes_oauth_credentials",
        lambda: {"cg.chae-claude": {"accessToken": "shared-tok"}},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True
    )
    monkeypatch.setattr("hermes_cli.auth.is_source_suppressed", lambda p, s: False)


# --------------------------------------------------------------------------
# _normalize_pool_priorities: explicit priority wins among file-backed accounts
# --------------------------------------------------------------------------


def test_load_pool_keeps_user_priority_between_shared_and_claude_code(
    tmp_path, monkeypatch, no_autodiscovery
):
    """claude_code p0 / hermes_pkce_shared p1 must survive load_pool()."""
    pool = _make_pool(
        tmp_path,
        monkeypatch,
        [
            _oauth_entry(0, "cg.chae-claude", "hermes_pkce_shared", "shared-tok", 1),
            _oauth_entry(1, "claude_code", "claude_code", "main-tok", 0),
        ],
    )
    order = [(e.label, e.priority) for e in sorted(pool.entries(), key=lambda e: e.priority)]
    assert order == [("claude_code", 0), ("cg.chae-claude", 1)]
    assert pool.select().label == "claude_code"


def test_load_pool_env_token_still_outranks_file_backed_accounts(
    tmp_path, monkeypatch, no_autodiscovery
):
    """Explicit env override keeps winning regardless of file-backed priority."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hermes" / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [
                        _oauth_entry(0, "claude_code", "claude_code", "main-tok", 0)
                    ]
                },
            }
        )
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_TOKEN", "env-override-token")
    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    assert pool.select().source == "env:ANTHROPIC_TOKEN"


# --------------------------------------------------------------------------
# CredentialPool.refresh_current_entry: never rotates behind the caller's back
# --------------------------------------------------------------------------


def test_refresh_current_entry_returns_selected_entry_without_rotating(
    tmp_path, monkeypatch, no_autodiscovery
):
    pool = _make_pool(
        tmp_path,
        monkeypatch,
        [
            _oauth_entry(0, "cg.chae-claude", "hermes_pkce_shared", "shared-tok", 0),
            _oauth_entry(1, "claude_code", "claude_code", "main-tok", 1),
        ],
    )
    # shared rows are hydrated from the grant file; stub it to the same token
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_shared_hermes_oauth_credentials",
        lambda label=None: {
            "accessToken": "shared-tok",
            "refreshToken": "shared-tok-refresh",
            "expiresAt": NOW_MS + 3_600_000,
        },
    )
    selected = pool.select()
    assert selected.label == "cg.chae-claude"

    entry = pool.refresh_current_entry(api_key_hint="shared-tok")
    assert entry is not None
    assert entry.id == selected.id
    assert entry.access_token == "shared-tok"
    assert pool.current().id == selected.id


def test_refresh_current_entry_none_when_nothing_selected(
    tmp_path, monkeypatch, no_autodiscovery
):
    pool = _make_pool(
        tmp_path,
        monkeypatch,
        [_oauth_entry(0, "claude_code", "claude_code", "main-tok", 0)],
    )
    assert pool.current() is None
    assert pool.refresh_current_entry(api_key_hint="unknown") is None
    # unambiguous hint re-anchors the cursor
    assert pool.refresh_current_entry(api_key_hint="main-tok").id == "cred-0"


# --------------------------------------------------------------------------
# AIAgent._try_refresh_anthropic_client_credentials: pool-bound, not singleton
# --------------------------------------------------------------------------


def _agent_with_pool(pool, current_key):
    from run_agent import AIAgent

    agent = SimpleNamespace(
        api_mode="anthropic_messages",
        provider="anthropic",
        _anthropic_api_key=current_key,
        _anthropic_base_url="https://api.anthropic.com",
        _credential_pool=pool,
        _credential_pool_entry_id=None,
        _anthropic_client=MagicMock(),
        model="claude-x",
    )
    agent._resolve_anthropic_refresh_token_from_pool = (
        AIAgent._resolve_anthropic_refresh_token_from_pool.__get__(agent)
    )
    agent._try_refresh_anthropic_client_credentials = (
        AIAgent._try_refresh_anthropic_client_credentials.__get__(agent)
    )
    return agent


def test_per_request_refresh_keeps_pool_selected_token(
    tmp_path, monkeypatch, no_autodiscovery
):
    """Singleton resolver says 'main-tok' (claude_code file); pool selected the
    shared account.  The client must stay on the pool's token."""
    pool = _make_pool(
        tmp_path,
        monkeypatch,
        [
            _oauth_entry(0, "cg.chae-claude", "hermes_pkce_shared", "shared-tok", 0),
            _oauth_entry(1, "claude_code", "claude_code", "main-tok", 1),
        ],
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_shared_hermes_oauth_credentials",
        lambda label=None: {
            "accessToken": "shared-tok",
            "refreshToken": "shared-tok-refresh",
            "expiresAt": NOW_MS + 3_600_000,
        },
    )
    assert pool.select().label == "cg.chae-claude"

    monkeypatch.setattr(
        "agent.anthropic_adapter.resolve_anthropic_token", lambda: "main-tok"
    )
    built = []
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        lambda *a, **kw: built.append((a, kw)) or MagicMock(),
    )

    agent = _agent_with_pool(pool, current_key="shared-tok")
    changed = agent._try_refresh_anthropic_client_credentials()

    assert changed is False, "pool token unchanged -> no client rebuild"
    assert built == []
    assert agent._anthropic_api_key == "shared-tok"


def test_per_request_refresh_falls_back_to_singleton_without_pool(monkeypatch):
    monkeypatch.setattr(
        "agent.anthropic_adapter.resolve_anthropic_token", lambda: "main-tok"
    )
    built = []
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        lambda *a, **kw: built.append((a, kw)) or MagicMock(),
    )
    agent = _agent_with_pool(None, current_key="stale-tok")
    assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent._anthropic_api_key == "main-tok"
    assert len(built) == 1
