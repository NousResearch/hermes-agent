"""Unit tests for claude_cli setup-token resolution (canonical root fallback).

Resolution order under test:
  1. explicit
  2. profile / process env CLAUDE_CODE_OAUTH_TOKEN (legacy ANTHROPIC_TOKEN)
  3. profile credential_pool OAuth
  4. canonical Hermes root ~/.hermes/.env (get_default_hermes_root)
  5. never rotating ~/.claude login

No live network / claude binary.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.transports.claude_cli import ClaudeCliError
from agent.transports.claude_cli_session import (
    _read_canonical_root_setup_token,
    _read_setup_token_keys_from_env_file,
    resolve_claude_cli_oauth_token,
)


# Masked-looking fakes — never real secrets.
_PROFILE_TOKEN = "sk-ant-oat01-PROFILE_TEST_TOKEN_NOT_REAL"
_ROOT_TOKEN = "sk-ant-oat01-ROOT_TEST_TOKEN_NOT_REAL"
_POOL_TOKEN = "sk-ant-oat01-POOL_TEST_TOKEN_NOT_REAL"
_EXPLICIT_TOKEN = "sk-ant-oat01-EXPLICIT_TEST_TOKEN_NOT_REAL"
_ROTATING_LOGIN_TOKEN = "sk-ant-oat01-ROTATING_LOGIN_TEST_TOKEN_NOT_REAL"


@pytest.fixture
def clean_token_env(monkeypatch):
    """Strip process env sources so resolution only sees what each test sets."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def no_pool(monkeypatch):
    """Stub credential_pool so tests do not touch real auth.json."""
    monkeypatch.setattr(
        "agent.anthropic_adapter._resolve_anthropic_pool_token",
        lambda: None,
    )


@pytest.fixture
def no_root_token(monkeypatch):
    monkeypatch.setattr(
        "agent.transports.claude_cli_session._read_canonical_root_setup_token",
        lambda: None,
    )


def test_canonical_root_token_when_profile_has_none(
    tmp_path, monkeypatch, clean_token_env, no_pool
):
    """Profile with no env/pool + root .env token → resolves root token."""
    root = tmp_path / "hermes_root"
    root.mkdir()
    (root / ".env").write_text(
        f"CLAUDE_CODE_OAUTH_TOKEN={_ROOT_TOKEN}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: root,
    )

    # Simulate a profile HERMES_HOME that does not contain the token.
    profile_home = tmp_path / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    got = resolve_claude_cli_oauth_token()
    assert got == _ROOT_TOKEN


def test_profile_env_wins_over_root(
    tmp_path, monkeypatch, clean_token_env, no_pool
):
    """Profile env token wins over canonical root .env (order preserved)."""
    root = tmp_path / "hermes_root"
    root.mkdir()
    (root / ".env").write_text(
        f"CLAUDE_CODE_OAUTH_TOKEN={_ROOT_TOKEN}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: root,
    )
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _PROFILE_TOKEN)

    got = resolve_claude_cli_oauth_token()
    assert got == _PROFILE_TOKEN


def test_explicit_wins_over_everything(
    tmp_path, monkeypatch, clean_token_env, no_pool
):
    root = tmp_path / "hermes_root"
    root.mkdir()
    (root / ".env").write_text(
        f"CLAUDE_CODE_OAUTH_TOKEN={_ROOT_TOKEN}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: root,
    )
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _PROFILE_TOKEN)

    got = resolve_claude_cli_oauth_token(explicit=_EXPLICIT_TOKEN)
    assert got == _EXPLICIT_TOKEN


def test_pool_wins_over_root(tmp_path, monkeypatch, clean_token_env):
    root = tmp_path / "hermes_root"
    root.mkdir()
    (root / ".env").write_text(
        f"CLAUDE_CODE_OAUTH_TOKEN={_ROOT_TOKEN}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: root,
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter._resolve_anthropic_pool_token",
        lambda: _POOL_TOKEN,
    )

    got = resolve_claude_cli_oauth_token()
    assert got == _POOL_TOKEN


def test_no_token_anywhere_raises_clean_error(
    monkeypatch, clean_token_env, no_pool, no_root_token
):
    with pytest.raises(ClaudeCliError, match="no Anthropic setup token"):
        resolve_claude_cli_oauth_token()


def test_rotating_claude_login_never_used(
    monkeypatch, clean_token_env, no_pool, no_root_token
):
    """Even if resolve_anthropic_token / ~/.claude would yield a token, ignore it."""

    def _boom_if_called():
        raise AssertionError(
            "resolve_anthropic_token must not be used for claude_cli "
            "(would pull rotating ~/.claude login)"
        )

    monkeypatch.setattr(
        "agent.anthropic_adapter.resolve_anthropic_token",
        _boom_if_called,
    )
    # Also stub the credentials readers so a mistaken import still fails loud.
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: {
            "accessToken": _ROTATING_LOGIN_TOKEN,
            "expiresAt": 9999999999999,
        },
    )

    with pytest.raises(ClaudeCliError, match="no Anthropic setup token"):
        resolve_claude_cli_oauth_token()


def test_root_env_absent_is_graceful(
    tmp_path, monkeypatch, clean_token_env, no_pool
):
    """Missing canonical root .env → no crash; falls through to clean error."""
    root = tmp_path / "hermes_root_empty"
    root.mkdir()
    # deliberately no .env file
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: root,
    )

    with pytest.raises(ClaudeCliError, match="no Anthropic setup token"):
        resolve_claude_cli_oauth_token()


def test_root_env_unreadable_is_graceful(
    tmp_path, monkeypatch, clean_token_env, no_pool
):
    """Unreadable root path → no crash."""
    missing = tmp_path / "does_not_exist_root"
    # do not create — get_default_hermes_root points at a non-existent dir
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: missing,
    )

    assert _read_canonical_root_setup_token() is None
    with pytest.raises(ClaudeCliError, match="no Anthropic setup token"):
        resolve_claude_cli_oauth_token()


def test_read_env_file_legacy_anthropic_token_alias(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_TOKEN=sk-ant-oat01-LEGACY_ALIAS_NOT_REAL\n",
        encoding="utf-8",
    )
    assert (
        _read_setup_token_keys_from_env_file(env_file)
        == "sk-ant-oat01-LEGACY_ALIAS_NOT_REAL"
    )


def test_read_env_file_prefers_claude_code_oauth_token(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_TOKEN=sk-ant-oat01-LEGACY_NOT_REAL\n"
        f"CLAUDE_CODE_OAUTH_TOKEN={_ROOT_TOKEN}\n",
        encoding="utf-8",
    )
    assert _read_setup_token_keys_from_env_file(env_file) == _ROOT_TOKEN


def test_read_env_file_missing_returns_none(tmp_path):
    assert _read_setup_token_keys_from_env_file(tmp_path / "nope.env") is None


def test_agent_setup_shaped_key_counts_as_passed(
    monkeypatch, clean_token_env, no_pool, no_root_token
):
    agent = SimpleNamespace(api_key=_PROFILE_TOKEN)
    assert resolve_claude_cli_oauth_token(agent=agent) == _PROFILE_TOKEN


def test_agent_setup_key_wins_over_root(
    tmp_path, monkeypatch, clean_token_env, no_pool
):
    root = tmp_path / "hermes_root"
    root.mkdir()
    (root / ".env").write_text(
        f"CLAUDE_CODE_OAUTH_TOKEN={_ROOT_TOKEN}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: root,
    )
    agent = SimpleNamespace(api_key=_PROFILE_TOKEN)
    assert resolve_claude_cli_oauth_token(agent=agent) == _PROFILE_TOKEN


def test_agent_plain_api_key_not_used_as_setup_token(
    monkeypatch, clean_token_env, no_pool, no_root_token
):
    """Regular console API keys are not setup tokens for claude_cli injection."""
    agent = SimpleNamespace(api_key="sk-ant-api03-NOT_A_SETUP_TOKEN")
    with pytest.raises(ClaudeCliError, match="no Anthropic setup token"):
        resolve_claude_cli_oauth_token(agent=agent)
