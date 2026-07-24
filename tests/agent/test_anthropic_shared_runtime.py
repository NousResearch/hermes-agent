"""Runtime gate: bypasses, official target predicate, epoch preflight."""

from __future__ import annotations

import os

import pytest

from tests.agent.anthropic_shared_test_helpers import (
    enable_marker,
    shared_root,
    stage_three,
)


def test_official_target_predicate_matrix():
    from agent.anthropic_shared_pool import is_official_anthropic_oauth_target as pred

    assert pred("anthropic", "inference", "anthropic_messages", None) is True
    assert pred("anthropic", "inference", None, "https://api.anthropic.com") is True
    assert pred("anthropic", "inference", None, "https://api.anthropic.com/v1") is True
    assert pred("anthropic", "model_discovery", None, None) is True
    # Non-official
    assert pred("anthropic", "inference", None, "https://my-azure.openai.azure.com") is False
    assert pred("anthropic", "inference", None, "https://api.anthropic.com.evil.com") is False
    assert pred("anthropic", "inference", None, "https://user:pass@api.anthropic.com") is False
    assert pred("anthropic", "inference", None, "http://api.anthropic.com") is False
    assert pred("anthropic", "inference", None, "https://api.anthropic.com:8443") is False
    assert pred("minimax", "inference", "anthropic_messages", None) is False
    assert pred("anthropic", "inference", "chat_completions", None) is False
    assert pred("bedrock", "inference", None, None) is False


def test_shared_mode_ignores_env_and_explicit_key(shared_root, monkeypatch):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-api-key-ENVKEYSHOULDNOTWIN0000")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "fixture-should-not-win-token-xxxxxxxxxxxx")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fixture-should-not-win-cc-xxxxxxxxxxxx")
    from agent import anthropic_shared_pool as sp
    from agent.anthropic_adapter import resolve_anthropic_token

    sp.reset_startup_epoch_for_tests()
    explicit = "fixture-" + "explicit-should-not-win-00000000"
    token = resolve_anthropic_token(
        explicit_api_key=explicit,
        provider="anthropic",
    )
    assert token.startswith("fixture-oauth-access-token")
    assert "ENVKEY" not in token
    assert "EXPLICIT" not in token


def test_shared_mode_skips_seeding(shared_root, monkeypatch):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-api-key-seedmeplease000000000")
    from agent import anthropic_shared_pool as sp
    from agent.credential_pool import load_pool

    sp.reset_startup_epoch_for_tests()
    pool = load_pool("anthropic")
    assert len(pool.entries()) == 3
    assert all(e.auth_type == "oauth" for e in pool.entries())
    assert all(e.source == "manual:hermes_pkce" for e in pool.entries())


def test_non_official_target_does_not_use_shared(shared_root, monkeypatch):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-api-key-azureishkey00000000000")
    from agent import anthropic_shared_pool as sp
    from agent.anthropic_adapter import resolve_anthropic_token

    sp.reset_startup_epoch_for_tests()
    token = resolve_anthropic_token(
        provider="anthropic",
        base_url="https://myresource.openai.azure.com/anthropic",
        explicit_api_key="fixture-api-key-azureishkey00000000000",
    )
    # Non-official: shared gate skipped; explicit/env may win
    assert token == "fixture-api-key-azureishkey00000000000"


def test_epoch_change_fails_new_requests(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root, epoch="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    from agent import anthropic_shared_pool as sp
    from hermes_cli.auth import AuthError

    sp.reset_startup_epoch_for_tests()
    sp.observe_startup_epoch()
    # Change epoch on disk
    enable_marker(shared_root, epoch="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    with pytest.raises(AuthError) as ei:
        sp.resolve_shared_anthropic_credential()
    assert ei.value.code == "shared_scope_changed"


def test_empty_shared_pool_no_local_fallback(shared_root, monkeypatch):
    # Marker present but no pool
    enable_marker(shared_root)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-api-key-shouldnotfallback0000")
    from agent import anthropic_shared_pool as sp
    from agent.anthropic_adapter import resolve_anthropic_token
    from hermes_cli.auth import AuthError

    sp.reset_startup_epoch_for_tests()
    with pytest.raises(AuthError):
        resolve_anthropic_token(provider="anthropic")


def test_get_anthropic_key_shared(shared_root, monkeypatch):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-api-key-nope0000000000000000")
    from agent import anthropic_shared_pool as sp
    from hermes_cli.auth import get_anthropic_key

    sp.reset_startup_epoch_for_tests()
    key = get_anthropic_key()
    assert key.startswith("fixture-oauth-access-token")
