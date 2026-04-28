"""Regression tests for the /model picker's credential-discovery paths.

Covers:
 - Normal path (tokens already in Hermes auth store)
 - Claude Code fallback (tokens only in ~/.claude/.credentials.json)
 - Negative case (no credentials anywhere)

Note: auto-import from ~/.codex/auth.json was removed in #12360 — Hermes
now owns its own openai-codex auth state, and users explicitly adopt
existing Codex CLI tokens via `hermes auth openai-codex`. The old
"Codex CLI shared file" discovery tests were removed with that change.
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
