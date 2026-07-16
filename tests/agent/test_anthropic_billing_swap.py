"""Tests for the spend-guard billing swap in resolve_anthropic_token.

While the spend guard reports the api_key lane swapped (daily cap
exhausted), API-key-shaped tokens are demoted below the OAuth sources so a
freshly spawned worker bills the personal account; if no OAuth source
resolves, the API key is still returned (degrade to normal billing, never
to a broken worker).
"""
from __future__ import annotations

import pytest

from agent import anthropic_adapter
from agent.anthropic_adapter import resolve_anthropic_token

API_KEY = "sk-ant-api03-shared-key"
OAUTH_TOKEN = "sk-ant-oat01-personal"


@pytest.fixture(autouse=True)
def clean_sources(monkeypatch):
    """Neutralize every credential source; tests re-enable what they need."""
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(anthropic_adapter, "read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(
        anthropic_adapter, "_resolve_claude_code_token_from_credentials", lambda creds: None
    )
    monkeypatch.setattr(anthropic_adapter, "_resolve_anthropic_pool_token", lambda: None)


def _set_swap(monkeypatch, active: bool):
    monkeypatch.setattr(
        anthropic_adapter, "_spend_guard_api_key_swap_active", lambda: active
    )


def test_no_swap_pinned_api_key_wins(monkeypatch):
    _set_swap(monkeypatch, False)
    monkeypatch.setenv("ANTHROPIC_TOKEN", API_KEY)
    assert resolve_anthropic_token() == API_KEY


def test_swap_prefers_keychain_oauth_over_pinned_api_key(monkeypatch):
    _set_swap(monkeypatch, True)
    monkeypatch.setenv("ANTHROPIC_TOKEN", API_KEY)
    monkeypatch.setattr(
        anthropic_adapter,
        "_resolve_claude_code_token_from_credentials",
        lambda creds: OAUTH_TOKEN,
    )
    assert resolve_anthropic_token() == OAUTH_TOKEN


def test_swap_falls_back_to_api_key_when_no_oauth(monkeypatch):
    _set_swap(monkeypatch, True)
    monkeypatch.setenv("ANTHROPIC_TOKEN", API_KEY)
    assert resolve_anthropic_token() == API_KEY


def test_swap_demotes_anthropic_api_key_env_too(monkeypatch):
    """Worker .envs carry the shared key in ANTHROPIC_API_KEY as fallback —
    under swap it must not shadow the keychain OAuth source."""
    _set_swap(monkeypatch, True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    monkeypatch.setattr(
        anthropic_adapter,
        "_resolve_claude_code_token_from_credentials",
        lambda creds: OAUTH_TOKEN,
    )
    assert resolve_anthropic_token() == OAUTH_TOKEN
    # ...but with no OAuth source it is still the fallback.
    monkeypatch.setattr(
        anthropic_adapter,
        "_resolve_claude_code_token_from_credentials",
        lambda creds: None,
    )
    assert resolve_anthropic_token() == API_KEY


def test_swap_does_not_demote_oauth_shaped_token(monkeypatch):
    _set_swap(monkeypatch, True)
    monkeypatch.setenv("ANTHROPIC_TOKEN", OAUTH_TOKEN)
    assert resolve_anthropic_token() == OAUTH_TOKEN


def test_swap_check_failure_means_no_swap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_TOKEN", API_KEY)

    def boom():
        raise RuntimeError("spend_meter broken")

    # The real guard swallows exceptions; simulate via the real function with
    # a broken spend_meter import path.
    monkeypatch.setattr(anthropic_adapter, "_spend_guard_api_key_swap_active",
                        anthropic_adapter._spend_guard_api_key_swap_active)
    import agent.spend_meter as sm
    monkeypatch.setattr(sm, "read_throttle", boom)
    assert resolve_anthropic_token() == API_KEY
