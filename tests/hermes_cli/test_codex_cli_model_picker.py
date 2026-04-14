"""Regression test: openai-codex must appear in /model picker when
credentials are only in the Codex CLI shared file (~/.codex/auth.json)
and haven't been migrated to the Hermes auth store yet.

Root cause: list_authenticated_providers() checked the raw Hermes auth
store but didn't know about the Codex CLI fallback import path.

Fix: _seed_from_singletons() now imports from the Codex CLI when the
Hermes auth store has no openai-codex tokens, and
list_authenticated_providers() falls back to load_pool() for OAuth
providers.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_fake_jwt(expiry_offset: int = 3600) -> str:
    """Build a fake JWT with a future expiry."""
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    exp = int(time.time()) + expiry_offset
    payload_bytes = json.dumps({"exp": exp, "sub": "test"}).encode()
    payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    return f"{header}.{payload}.fakesig"


@pytest.fixture()
def codex_cli_only_env(tmp_path, monkeypatch):
    """Set up an environment where Codex tokens exist only in ~/.codex/auth.json,
    NOT in the Hermes auth store."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    # Empty Hermes auth store
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 2, "providers": {}})
    )

    # Valid Codex CLI tokens
    fake_jwt = _make_fake_jwt()
    (codex_home / "auth.json").write_text(
        json.dumps({
            "tokens": {
                "access_token": fake_jwt,
                "refresh_token": "fake-refresh-token",
            }
        })
    )

    # Clear provider env vars so only OAuth is a detection path
    for var in [
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "NOUS_API_KEY", "DEEPSEEK_API_KEY", "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN", "GEMINI_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)

    return hermes_home


def test_codex_cli_tokens_detected_by_model_picker(codex_cli_only_env):
    """openai-codex should appear when tokens only exist in ~/.codex/auth.json."""
    from hermes_cli.model_switch import list_authenticated_providers

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        max_models=10,
    )
    slugs = [p["slug"] for p in providers]
    assert "openai-codex" in slugs, (
        f"openai-codex not found in /model picker providers: {slugs}"
    )

    codex = next(p for p in providers if p["slug"] == "openai-codex")
    assert codex["is_current"] is True
    assert codex["total_models"] > 0


def test_codex_cli_tokens_migrated_after_detection(codex_cli_only_env):
    """After the /model picker detects Codex CLI tokens, they should be
    migrated into the Hermes auth store for subsequent fast lookups."""
    from hermes_cli.model_switch import list_authenticated_providers

    # First call triggers migration
    list_authenticated_providers(current_provider="openai-codex")

    # Verify tokens are now in Hermes auth store
    auth_path = codex_cli_only_env / "auth.json"
    store = json.loads(auth_path.read_text())
    providers = store.get("providers", {})
    assert "openai-codex" in providers, (
        f"openai-codex not migrated to Hermes auth store: {list(providers.keys())}"
    )
    tokens = providers["openai-codex"].get("tokens", {})
    assert tokens.get("access_token"), "access_token missing after migration"
    assert tokens.get("refresh_token"), "refresh_token missing after migration"


@pytest.fixture()
def hermes_auth_only_env(tmp_path, monkeypatch):
    """Tokens already in Hermes auth store (no Codex CLI needed)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Point CODEX_HOME to nonexistent dir to prove it's not needed
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no_codex"))

    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 2,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": _make_fake_jwt(),
                    "refresh_token": "fake-refresh",
                },
                "last_refresh": "2026-04-12T00:00:00Z",
            }
        },
    }))

    for var in [
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "NOUS_API_KEY", "DEEPSEEK_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)

    return hermes_home


def test_normal_path_still_works(hermes_auth_only_env):
    """openai-codex appears when tokens are already in Hermes auth store."""
    from hermes_cli.model_switch import list_authenticated_providers

    providers = list_authenticated_providers(
        current_provider="openai-codex",
        max_models=10,
    )
    slugs = [p["slug"] for p in providers]
    assert "openai-codex" in slugs


@pytest.fixture()
def claude_code_only_env(tmp_path, monkeypatch):
    """Set up an environment where Anthropic credentials only exist in
    ~/.claude/.credentials.json (Claude Code) — not in env vars or Hermes
    auth store."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # No Codex CLI
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no_codex"))

    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 2, "providers": {}})
    )

    # Claude Code credentials in the correct format
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": _make_fake_jwt(),
            "refreshToken": "fake-refresh",
            "expiresAt": int(time.time() * 1000) + 3_600_000,
        }
    }))

    # Patch Path.home() so the adapter finds the file
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    for var in [
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
        "NOUS_API_KEY", "DEEPSEEK_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)

    return hermes_home


def test_claude_code_file_detected_by_model_picker(claude_code_only_env):
    """anthropic should appear when credentials only exist in ~/.claude/.credentials.json."""
    from hermes_cli.model_switch import list_authenticated_providers

    providers = list_authenticated_providers(
        current_provider="anthropic",
        max_models=10,
    )
    slugs = [p["slug"] for p in providers]
    assert "anthropic" in slugs, (
        f"anthropic not found in /model picker providers: {slugs}"
    )

    anthropic = next(p for p in providers if p["slug"] == "anthropic")
    assert anthropic["is_current"] is True
    assert anthropic["total_models"] > 0


def test_no_codex_when_no_credentials(tmp_path, monkeypatch):
    """openai-codex should NOT appear when no credentials exist anywhere."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no_codex"))

    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 2, "providers": {}})
    )

    for var in [
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "NOUS_API_KEY", "DEEPSEEK_API_KEY", "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN", "GEMINI_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)

    from hermes_cli.model_switch import list_authenticated_providers

    providers = list_authenticated_providers(
        current_provider="openrouter",
        max_models=10,
    )
    slugs = [p["slug"] for p in providers]
    assert "openai-codex" not in slugs, (
        "openai-codex should not appear without any credentials"
    )
