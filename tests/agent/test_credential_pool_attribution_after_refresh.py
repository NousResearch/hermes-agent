"""A 401/429 carrying a pre-refresh OAuth token must still bench the entry that sent it.

Single-use OAuth refresh (possibly finished by a peer process) can swap an entry's
access token between request dispatch and failure recovery. The failed request's
``api_key_hint`` is then the *previous* token generation: it matches no live
``runtime_api_key``, ``_identify_failed_entry`` used to drop the (correct)
``credential_id`` too, and ``_rotate_unmatched()`` benched nothing — so a login
blocked by a weekly limit was re-selected on every turn.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _oauth_row(entry_id: str, label: str, priority: int, token: str) -> dict:
    return {
        "id": entry_id,
        "label": label,
        "auth_type": "oauth",
        "priority": priority,
        "source": "manual:hermes_pkce",
        "access_token": token,
        "refresh_token": f"refresh-{entry_id}",
        "expires_at_ms": int((time.time() + 6 * 3600) * 1000),
    }


def _load_two_oauth_logins(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    _oauth_row("cred-a", "a@example.com", 0, "sk-ant-oat01-a-gen1"),
                    _oauth_row("cred-b", "b@example.com", 1, "sk-ant-oat01-b-gen1"),
                ]
            },
        },
    )
    from agent.credential_pool import load_pool

    return load_pool("anthropic")


def _rotate_token(pool, entry_id: str, new_token: str) -> None:
    entry = next(e for e in pool.entries() if e.id == entry_id)
    pool._replace_entry(entry, replace(entry, access_token=new_token))


def _statuses(pool) -> dict:
    return {e.id: e.last_status for e in pool.entries()}


def test_id_plus_pre_refresh_key_benches_the_issuing_entry(tmp_path, monkeypatch):
    from agent.credential_pool import STATUS_EXHAUSTED

    pool = _load_two_oauth_logins(tmp_path, monkeypatch)
    _rotate_token(pool, "cred-a", "sk-ant-oat01-a-gen2")  # refresh lands mid-request

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429, api_key_hint="sk-ant-oat01-a-gen1", credential_id="cred-a",
    )

    statuses = _statuses(pool)
    assert statuses["cred-a"] == STATUS_EXHAUSTED
    assert statuses["cred-b"] != STATUS_EXHAUSTED
    assert next_entry is not None and next_entry.id == "cred-b"


def test_key_only_pre_refresh_identity_benches_the_issuing_entry(tmp_path, monkeypatch):
    from agent.credential_pool import STATUS_EXHAUSTED

    pool = _load_two_oauth_logins(tmp_path, monkeypatch)
    _rotate_token(pool, "cred-a", "sk-ant-oat01-a-gen2")

    pool.mark_exhausted_and_rotate(status_code=429, api_key_hint="sk-ant-oat01-a-gen1")

    statuses = _statuses(pool)
    assert statuses["cred-a"] == STATUS_EXHAUSTED
    assert statuses["cred-b"] != STATUS_EXHAUSTED


def test_unknown_key_with_stale_id_still_marks_nothing(tmp_path, monkeypatch):
    """A key this pool never held keeps the existing unmatched behaviour (#79156 guard)."""
    from agent.credential_pool import STATUS_EXHAUSTED

    pool = _load_two_oauth_logins(tmp_path, monkeypatch)
    _rotate_token(pool, "cred-a", "sk-ant-oat01-a-gen2")

    pool.mark_exhausted_and_rotate(
        status_code=429, api_key_hint="sk-ant-oat01-foreign", credential_id="cred-a",
    )

    assert STATUS_EXHAUSTED not in _statuses(pool).values()


def test_retired_key_map_stores_digests_only(tmp_path, monkeypatch):
    pool = _load_two_oauth_logins(tmp_path, monkeypatch)
    _rotate_token(pool, "cred-a", "sk-ant-oat01-a-gen2")

    stored = json.dumps(pool._retired_runtime_keys)
    assert "sk-ant-oat01-a-gen1" not in stored
    assert list(pool._retired_runtime_keys.values()) == ["cred-a"]



@pytest.mark.parametrize(
    "auth_type, key_prefix, expect_benched",
    [
        pytest.param("oauth", "sk-ant-oat01", True, id="oauth-refresh"),
        pytest.param("api_key", "sk-ant-api", False, id="manual-api-key-replacement"),
    ],
)
def test_retired_hint_attribution_requires_oauth(
    tmp_path, monkeypatch, auth_type, key_prefix, expect_benched,
):
    """Retired OAuth tokens share a login; a replaced manual API key may not."""
    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    rows = [
        {
            "id": f"cred-{ident}",
            "label": f"{ident}@example.com",
            "auth_type": auth_type,
            "priority": priority,
            "source": "manual:hermes_pkce" if auth_type == "oauth" else "manual",
            "access_token": f"{key_prefix}-{ident}-gen1",
            **({"refresh_token": f"refresh-{ident}",
                "expires_at_ms": int((time.time() + 6 * 3600) * 1000)}
               if auth_type == "oauth" else {}),
        }
        for priority, ident in enumerate(("a", "b"))
    ]
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {"anthropic": rows}})
    pool = load_pool("anthropic")

    old_a = next(e for e in pool.entries() if e.id == "cred-a")
    pool._replace_entry(old_a, replace(old_a, access_token=f"{key_prefix}-a-gen2"))
    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="cred-a", api_key_hint=f"{key_prefix}-a-gen1",
    )

    statuses = _statuses(pool)
    assert (statuses["cred-a"] == STATUS_EXHAUSTED) is expect_benched
    assert statuses["cred-b"] != STATUS_EXHAUSTED
