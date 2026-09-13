"""Codex OAuth refresh writes back to the store the grant was resolved FROM (#87503).

Codex refresh tokens are single-use with rotation-family reuse detection: a profile that refreshed
a root-borrowed grant must land the rotated chain in root — singleton AND ``credential_pool`` —
or root keeps the consumed refresh token and the next reader gets the whole family revoked.
Token values are synthetic placeholders.
"""

import json
from pathlib import Path

import pytest

from hermes_cli import auth


def _pair(prefix: str) -> dict:
    return {"access_token": f"{prefix}-at", "refresh_token": f"{prefix}-rt"}


def _write(path: Path, store: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    """Global root at tmp/.hermes, active profile at tmp/.hermes/profiles/work (real on-disk layout)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return profile / "auth.json", root / "auth.json"


def test_profile_refresh_of_root_grant_writes_through_to_root(profile_env):
    profile_path, root_path = profile_env
    _write(root_path, {
        "version": 1,
        "providers": {"openai-codex": {"auth_mode": "chatgpt", "tokens": _pair("old")}},
        "credential_pool": {"openai-codex": [
            {"provider": "openai-codex", "source": "device_code", **_pair("old")}]},
    })
    _write(profile_path, {"version": 1, "providers": {}})

    rotated = _pair("new")
    auth._save_codex_tokens(rotated, last_refresh="2026-08-16T00:00:00Z")

    root = _read(root_path)
    assert root["providers"]["openai-codex"]["tokens"] == rotated
    assert root["credential_pool"]["openai-codex"][0]["refresh_token"] == rotated["refresh_token"]
    assert root["credential_pool"]["openai-codex"][0]["access_token"] == rotated["access_token"]
    # A profile copy would shadow root and disable the write-through on the next refresh (#74339).
    assert "openai-codex" not in _read(profile_path).get("providers", {})


def test_profile_owned_grant_stays_local(profile_env):
    profile_path, root_path = profile_env
    _write(profile_path, {
        "version": 1,
        "providers": {"openai-codex": {"auth_mode": "chatgpt", "tokens": _pair("prof")}},
    })
    _write(root_path, {"version": 1, "providers": {}})

    rotated = _pair("next")
    auth._save_codex_tokens(rotated, last_refresh="2026-08-16T00:00:00Z")

    assert _read(profile_path)["providers"]["openai-codex"]["tokens"] == rotated
    assert "openai-codex" not in _read(root_path).get("providers", {})
