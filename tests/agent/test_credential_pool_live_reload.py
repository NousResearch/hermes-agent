"""Cross-process credential-pool reload regression tests."""

from __future__ import annotations

import time

from agent.credential_pool import load_pool
from hermes_cli.auth import read_credential_pool, write_credential_pool


def _entry(
    credential_id: str,
    priority: int,
    *,
    status: str = "ok",
    status_at: float | None = None,
    reset_at: float | None = None,
) -> dict:
    return {
        "id": credential_id,
        "label": credential_id,
        "auth_type": "oauth",
        "priority": priority,
        "source": "manual:device_code",
        "access_token": f"token-{credential_id}",
        "refresh_token": f"refresh-{credential_id}",
        "last_status": status,
        "last_status_at": status_at,
        "last_error_code": 429 if status == "exhausted" else None,
        "last_error_reason": "rate_limit" if status == "exhausted" else None,
        "last_error_message": None,
        "last_error_reset_at": reset_at,
        "request_count": 0,
    }


def _isolate_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)


def test_select_adopts_external_priority_change(tmp_path, monkeypatch):
    """A process must not keep selecting the old first row after another writer reorders it."""
    _isolate_home(tmp_path, monkeypatch)
    write_credential_pool("openai-codex", [_entry("a", 0), _entry("b", 1)])
    pool = load_pool("openai-codex")

    write_credential_pool("openai-codex", [_entry("b", 0), _entry("a", 1)])

    assert pool.select().id == "b"
    assert [entry.id for entry in pool.entries()] == ["b", "a"]


def test_external_change_waits_until_active_lease_released(tmp_path, monkeypatch):
    """A persisted reorder must never replace the credential held by an in-flight lease."""
    _isolate_home(tmp_path, monkeypatch)
    write_credential_pool("openai-codex", [_entry("a", 0), _entry("b", 1)])
    pool = load_pool("openai-codex")
    assert pool.acquire_lease() == "a"

    write_credential_pool("openai-codex", [_entry("b", 0), _entry("a", 1)])

    assert [entry.id for entry in pool.entries()] == ["a", "b"]
    pool.release_lease("a")
    assert pool.acquire_lease() == "b"


def test_reload_adopts_newer_exhaustion_without_changing_tokens(tmp_path, monkeypatch):
    """A peer's cooldown must route away from that row without altering either OAuth token."""
    _isolate_home(tmp_path, monkeypatch)
    original = [_entry("a", 0), _entry("b", 1)]
    write_credential_pool("openai-codex", original)
    pool = load_pool("openai-codex")
    now = time.time()

    changed = [
        _entry("a", 0, status="exhausted", status_at=now, reset_at=now + 3600),
        _entry("b", 1),
    ]
    write_credential_pool("openai-codex", changed)

    assert pool.select().id == "b"
    persisted = {row["id"]: row for row in read_credential_pool("openai-codex")}
    assert persisted["a"]["access_token"] == "token-a"
    assert persisted["a"]["refresh_token"] == "refresh-a"
    assert persisted["b"]["access_token"] == "token-b"
    assert persisted["b"]["refresh_token"] == "refresh-b"
