"""CLI-level tests for `hermes auth reset` (interactive menu option 3).

The reset must clear exhaustion/rate-limit status on disk even while the
cooldowns are still active — the regression where the recency merge in
``write_credential_pool`` resurrected every still-binding on-disk cooldown.

All credentials here are synthetic mock data.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from agent.credential_pool import load_pool
from hermes_cli.auth_commands import auth_reset_command


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _exhausted_entry(entry_id, label, *, age_seconds, error_code=429):
    return {
        "id": entry_id,
        "label": label,
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": f"sk-mock-{entry_id}",
        "last_status": "exhausted",
        "last_status_at": time.time() - age_seconds,
        "last_error_code": error_code,
        "last_error_reason": "rate_limit_error",
        "last_error_message": "mock upstream message",
        "last_error_reset_at": None,
    }


def _setup_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    # Prevent auto-seeding from host Codex CLI tokens / anthropic config.
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
    )
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [
                    _exhausted_entry("cred-1", "alice", age_seconds=60),
                    _exhausted_entry("cred-2", "bob", age_seconds=120, error_code=402),
                ]
            },
        },
    )


def test_auth_reset_command_clears_active_cooldowns(tmp_path, monkeypatch, capsys):
    """Menu option 3: `hermes auth reset <provider>` must land on disk even
    while cooldowns are still active (the resurrection regression)."""
    _setup_pool(tmp_path, monkeypatch)

    pool = load_pool("openai-codex")
    assert pool.has_available() is False

    auth_reset_command(SimpleNamespace(provider="openai-codex"))

    out = capsys.readouterr().out
    assert "Reset status on 2 openai-codex credentials" in out

    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    for entry in auth_payload["credential_pool"]["openai-codex"]:
        assert entry.get("last_status") is None
        assert entry.get("last_status_at") is None
        assert entry.get("last_error_code") is None

    # A fresh pool load (what the runtime does) sees healthy credentials.
    reloaded = load_pool("openai-codex")
    assert reloaded.has_available() is True


def test_auth_reset_command_prints_zero_when_clean(tmp_path, monkeypatch, capsys):
    """A pool with no statuses reports 0 and leaves the file untouched."""
    _setup_pool(tmp_path, monkeypatch)
    auth_file = tmp_path / "hermes" / "auth.json"
    payload = json.loads(auth_file.read_text())
    for entry in payload["credential_pool"]["openai-codex"]:
        for key in (
            "last_status",
            "last_status_at",
            "last_error_code",
            "last_error_reason",
            "last_error_message",
            "last_error_reset_at",
        ):
            entry.pop(key, None)
    auth_file.write_text(json.dumps(payload))
    before = auth_file.read_bytes()

    auth_reset_command(SimpleNamespace(provider="openai-codex"))

    assert "Reset status on 0 openai-codex credentials" in capsys.readouterr().out
    assert auth_file.read_bytes() == before
