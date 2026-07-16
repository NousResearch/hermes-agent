"""Tests for spend-guard billing routing in resolve_anthropic_token.

The spend poller writes a routing directive (split / api_key_only /
personal_only). When a process prefers personal billing, API-key-shaped
tokens are demoted below the OAuth sources so it bills the personal
account; if no OAuth source resolves, the API key is still returned
(degrade to normal billing, never to a broken worker).
"""
from __future__ import annotations

import pytest

from agent import anthropic_adapter, spend_meter
from agent.anthropic_adapter import (
    _spend_guard_prefers_personal_billing,
    resolve_anthropic_token,
)

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
    monkeypatch.setattr(anthropic_adapter, "_billing_split_choice", None)


def _set_personal(monkeypatch, active: bool):
    monkeypatch.setattr(
        anthropic_adapter, "_spend_guard_prefers_personal_billing", lambda: active
    )


def _set_routing(monkeypatch, routing: dict):
    monkeypatch.setattr(
        spend_meter, "read_throttle", lambda path=None: spend_meter.ThrottleState(routing=routing)
    )


# ─── Resolver behavior given a billing preference ────────────────────────────


def test_no_preference_pinned_api_key_wins(monkeypatch):
    _set_personal(monkeypatch, False)
    monkeypatch.setenv("ANTHROPIC_TOKEN", API_KEY)
    assert resolve_anthropic_token() == API_KEY


def test_personal_prefers_keychain_oauth_over_pinned_api_key(monkeypatch):
    _set_personal(monkeypatch, True)
    monkeypatch.setenv("ANTHROPIC_TOKEN", API_KEY)
    monkeypatch.setattr(
        anthropic_adapter,
        "_resolve_claude_code_token_from_credentials",
        lambda creds: OAUTH_TOKEN,
    )
    assert resolve_anthropic_token() == OAUTH_TOKEN


def test_personal_falls_back_to_api_key_when_no_oauth(monkeypatch):
    _set_personal(monkeypatch, True)
    monkeypatch.setenv("ANTHROPIC_TOKEN", API_KEY)
    assert resolve_anthropic_token() == API_KEY


def test_personal_demotes_anthropic_api_key_env_too(monkeypatch):
    """Worker .envs carry the shared key in ANTHROPIC_API_KEY as fallback —
    under personal routing it must not shadow the keychain OAuth source."""
    _set_personal(monkeypatch, True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    monkeypatch.setattr(
        anthropic_adapter,
        "_resolve_claude_code_token_from_credentials",
        lambda creds: OAUTH_TOKEN,
    )
    assert resolve_anthropic_token() == OAUTH_TOKEN
    monkeypatch.setattr(
        anthropic_adapter,
        "_resolve_claude_code_token_from_credentials",
        lambda creds: None,
    )
    assert resolve_anthropic_token() == API_KEY


def test_personal_does_not_demote_oauth_shaped_token(monkeypatch):
    _set_personal(monkeypatch, True)
    monkeypatch.setenv("ANTHROPIC_TOKEN", OAUTH_TOKEN)
    assert resolve_anthropic_token() == OAUTH_TOKEN


# ─── Directive: routing mode → preference ────────────────────────────────────


def test_directive_personal_only(monkeypatch):
    _set_routing(monkeypatch, {"mode": "personal_only", "personal_share": 1.0})
    assert _spend_guard_prefers_personal_billing() is True


def test_directive_api_key_only(monkeypatch):
    _set_routing(monkeypatch, {"mode": "api_key_only", "personal_share": 0.0})
    assert _spend_guard_prefers_personal_billing() is False


@pytest.mark.parametrize("coin,expected", [(0.5, True), (0.95, False)])
def test_directive_split_uses_per_process_coin(monkeypatch, coin, expected):
    _set_routing(monkeypatch, {"mode": "split", "personal_share": 0.8})
    import random

    monkeypatch.setattr(random, "random", lambda: coin)
    assert _spend_guard_prefers_personal_billing() is expected
    # Stable within the process: a different coin later must not flip it.
    monkeypatch.setattr(random, "random", lambda: 1.0 - coin)
    assert _spend_guard_prefers_personal_billing() is expected


def test_directive_legacy_swapped_flag(monkeypatch):
    monkeypatch.setattr(
        spend_meter,
        "read_throttle",
        lambda path=None: spend_meter.ThrottleState(
            swapped_lanes={"api_key": {"to": "personal_oauth"}}
        ),
    )
    assert _spend_guard_prefers_personal_billing() is True


def test_directive_failure_means_no_preference(monkeypatch):
    def boom(path=None):
        raise RuntimeError("spend_meter broken")

    monkeypatch.setattr(spend_meter, "read_throttle", boom)
    monkeypatch.setenv("ANTHROPIC_TOKEN", API_KEY)
    assert _spend_guard_prefers_personal_billing() is False
    assert resolve_anthropic_token() == API_KEY
