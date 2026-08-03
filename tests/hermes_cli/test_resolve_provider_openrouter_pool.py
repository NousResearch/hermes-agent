"""Regression tests for issue #42130.

A credential added via `hermes auth add openrouter` lives in the credential
pool, NOT as an OPENROUTER_API_KEY env var. Before the fix, resolve_provider()
auto-detection only checked env vars, so such a credential was invisible:
the provider failed to resolve (AuthError) or resolved without a key, and
requests went out with no Authorization header — OpenRouter's
"HTTP 401: Missing Authentication header".

These tests lock in that auto-detection consults the OpenRouter pool.
"""

import uuid

import pytest


@pytest.fixture(autouse=True)
def _clean_inference_env(monkeypatch):
    """Strip credential-shaped env vars so the pool is the only source."""
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "NOUS_API_KEY",
        "HERMES_INFERENCE_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


def _seed_openrouter_pool(token: str = "sk-or-FAKEKEY123") -> None:
    """Mimic `hermes auth add openrouter <token>` — a manual pool entry."""
    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )

    pool = load_pool("openrouter")
    pool.add_entry(
        PooledCredential(
            provider="openrouter",
            id=uuid.uuid4().hex[:6],
            label="api-key-1",
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token=token,
            base_url="https://openrouter.ai/api/v1",
        )
    )


def test_auto_detects_openrouter_from_pool(tmp_path, monkeypatch):
    """With only a pool credential (no env var), auto-detection finds it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    _seed_openrouter_pool()

    from hermes_cli.auth import resolve_provider

    assert resolve_provider("auto") == "openrouter"


def test_no_credentials_still_raises(tmp_path, monkeypatch):
    """Empty pool + no env var must still fail to resolve — no false positive."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    from hermes_cli.auth import AuthError, resolve_provider

    with pytest.raises(AuthError):
        resolve_provider("auto")
