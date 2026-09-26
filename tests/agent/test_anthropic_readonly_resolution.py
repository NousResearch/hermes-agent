"""Regression coverage for the read-only Anthropic pool resolver."""

from __future__ import annotations

import json
from pathlib import Path

from agent import anthropic_credentials as credentials


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_resolving_hermes_anthropic_oauth_token_does_not_write_home(tmp_path, monkeypatch):
    """Removing the read-only resolver must not seed or initialize its stores."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    (home / "config.yaml").write_text("model:\n  provider: anthropic\n", encoding="utf-8")
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {}}), encoding="utf-8"
    )
    (home / ".anthropic_oauth.json").write_text(
        json.dumps({
            "accessToken": "sk-ant-oat01-readonly",
            "refreshToken": "sk-ant-ort01-readonly",
            "expiresAt": 4_102_444_800_000,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_keychain", lambda: None)
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_file", lambda: None)

    before = _file_snapshot(tmp_path)

    assert credentials._resolve_anthropic_pool_token(skip_borrowed=True) == "sk-ant-oat01-readonly"

    assert _file_snapshot(tmp_path) == before


def test_resolving_anthropic_pool_does_not_preserve_malformed_auth_store(tmp_path, monkeypatch):
    """A diagnostic read must not create ``auth.json.corrupt`` either."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    (home / "config.yaml").write_text("model:\n  provider: anthropic\n", encoding="utf-8")
    (home / "auth.json").write_text("not JSON", encoding="utf-8")
    (home / ".anthropic_oauth.json").write_text(
        json.dumps({"accessToken": "sk-ant-oat01-readonly"}), encoding="utf-8"
    )
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_keychain", lambda: None)
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_file", lambda: None)

    before = _file_snapshot(tmp_path)

    assert credentials._resolve_anthropic_pool_token(skip_borrowed=True) == "sk-ant-oat01-readonly"

    assert _file_snapshot(tmp_path) == before


def test_resolving_anthropic_env_pool_does_not_preserve_malformed_auth_store(tmp_path, monkeypatch):
    """Environment seeding must use the same non-mutating suppression lookup."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-readonly")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (home / "auth.json").write_text("not JSON", encoding="utf-8")
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_keychain", lambda: None)
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_file", lambda: None)

    before = _file_snapshot(tmp_path)

    assert credentials._resolve_anthropic_pool_token(skip_borrowed=True) == "sk-ant-oat01-readonly"

    assert _file_snapshot(tmp_path) == before


def test_resolving_explicit_pool_survives_malformed_config_without_writing(tmp_path, monkeypatch):
    """A broken config must not hide a separately explicit OAuth pool entry."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    (home / "config.yaml").write_text("model: [", encoding="utf-8")
    (home / "auth.json").write_text(
        json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {"anthropic": [{
                "id": "pkce", "source": "hermes_pkce", "auth_type": "oauth",
                "access_token": "sk-ant-oat01-readonly", "priority": 0,
            }]},
        }),
        encoding="utf-8",
    )
    (home / ".anthropic_oauth.json").write_text(
        json.dumps({"accessToken": "sk-ant-oat01-readonly"}), encoding="utf-8"
    )
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_keychain", lambda: None)
    monkeypatch.setattr(credentials, "_read_claude_code_credentials_from_file", lambda: None)

    before = _file_snapshot(tmp_path)

    assert credentials._resolve_anthropic_pool_token(skip_borrowed=True) == "sk-ant-oat01-readonly"

    assert _file_snapshot(tmp_path) == before
