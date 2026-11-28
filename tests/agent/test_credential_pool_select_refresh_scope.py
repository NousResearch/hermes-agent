"""``select()`` must only spend the single-use refresh token of the credential it returns.

Proactive refresh of a single-use OAuth token rotates (and on failure may quarantine) that
credential. When several pooled credentials are due for refresh at once, ``select()`` used to
refresh every one of them, spending fallbacks it never returned; an unpinned ``acquire_lease()``
did the same. Both now refresh only the entry they hand out (or, when nothing is selectable yet,
only the highest-priority candidate before re-selecting).
"""

from __future__ import annotations

import base64
import json
import time


def _jwt(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


def _write_pool(tmp_path, entries: list[dict]) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"openai-codex": entries}}, indent=2)
    )


def _expired_entry(entry_id: str, priority: int) -> dict:
    return {
        "id": entry_id,
        "label": entry_id,
        "auth_type": "oauth",
        "priority": priority,
        "source": "manual:device_code",
        "access_token": _jwt({"exp": int(time.time()) - 60, "sub": entry_id}),
        "refresh_token": f"{entry_id}-refresh",
    }


def _fresh_entry(entry_id: str, priority: int) -> dict:
    entry = _expired_entry(entry_id, priority)
    entry["access_token"] = _jwt({"exp": int(time.time()) + 3600, "sub": entry_id})
    return entry


def _setup(tmp_path, monkeypatch, entries: list[dict]) -> list[str]:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    _write_pool(tmp_path, entries)
    spent: list[str] = []

    def _refresh(access_token, refresh_token, **_kwargs):
        spent.append(refresh_token)
        return {
            "access_token": _jwt({"exp": int(time.time()) + 3600, "sub": "rotated"}),
            "refresh_token": f"{refresh_token}-rotated",
            "last_refresh": "2026-01-01T00:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", _refresh)
    return spent


def _stored_refresh_token(tmp_path, entry_id: str) -> str:
    stored = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    return next(e for e in stored["credential_pool"]["openai-codex"] if e["id"] == entry_id)["refresh_token"]


def test_select_refreshes_only_the_credential_it_returns(tmp_path, monkeypatch):
    spent = _setup(tmp_path, monkeypatch, [_expired_entry("primary", 0), _expired_entry("fallback", 1)])

    from agent.credential_pool import load_pool

    selected = load_pool("openai-codex").select()

    assert selected is not None and selected.id == "primary"
    assert spent == ["primary-refresh"], "an unselected fallback's single-use token was spent"
    stored = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    fallback = next(e for e in stored["credential_pool"]["openai-codex"] if e["id"] == "fallback")
    assert fallback["refresh_token"] == "fallback-refresh"


def test_select_with_a_ready_entry_spends_no_pending_token(tmp_path, monkeypatch):
    spent = _setup(tmp_path, monkeypatch, [_fresh_entry("ready", 0), _expired_entry("pending", 1)])

    from agent.credential_pool import load_pool

    selected = load_pool("openai-codex").select()

    assert selected is not None and selected.id == "ready"
    assert spent == []
    assert _stored_refresh_token(tmp_path, "pending") == "pending-refresh"


def test_unpinned_acquire_lease_refreshes_only_the_leased_candidate(tmp_path, monkeypatch):
    spent = _setup(tmp_path, monkeypatch, [_expired_entry("primary", 0), _expired_entry("fallback", 1)])

    from agent.credential_pool import load_pool

    leased = load_pool("openai-codex").acquire_lease()

    assert leased == "primary"
    assert spent == ["primary-refresh"]
    assert _stored_refresh_token(tmp_path, "fallback") == "fallback-refresh"



def test_acquire_lease_with_a_ready_entry_spends_no_pending_token(tmp_path, monkeypatch):
    spent = _setup(tmp_path, monkeypatch, [_fresh_entry("ready", 0), _expired_entry("pending", 1)])

    from agent.credential_pool import load_pool

    assert load_pool("openai-codex").acquire_lease() == "ready"
    assert spent == []
    assert _stored_refresh_token(tmp_path, "pending") == "pending-refresh"
