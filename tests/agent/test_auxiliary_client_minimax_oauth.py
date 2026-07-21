"""Tests for MiniMax OAuth auxiliary client routing.

Mirrors the xAI OAuth auxiliary routing contract tests
(``test_auth_xai_oauth_provider.py``).  Without the ``minimax-oauth`` branch
in ``resolve_provider_client``, a minimax-oauth main provider falls through
every ``auth_type`` check (it is not ``api_key``, ``oauth_external``,
``oauth_external_process``, ``vertex``, or ``aws_sdk``) and returns
``(None, None)`` — silently breaking every auxiliary task (vision, title
generation, compression, etc.) configured with
``auxiliary.<task>.provider: minimax-oauth``.

These tests pin three routing contracts:

1. **Authenticated (sync + async)** → returns a non-None
   ``AnthropicAuxiliaryClient`` (sync) / ``AsyncAnthropicAuxiliaryClient``
   (async) with ``is_oauth=False`` (MiniMax is a third-party
   Anthropic-compatible endpoint, NOT Claude Code OAuth — the ``is_oauth``
   flag injects Claude Code identity transforms that must not apply).
2. **Unauthenticated** → returns ``(None, None)`` so auto-fallback engages.
3. **Construction failure** → returns ``(None, None)`` rather than falling
   back to an OpenAI-wire client (the ``/anthropic`` endpoint speaks
   Anthropic Messages, not OpenAI chat.completions).
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.auxiliary_client import (
    AnthropicAuxiliaryClient,
    AsyncAnthropicAuxiliaryClient,
    resolve_provider_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAX_OAUTH_INFERENCE_URL = "https://api.minimax.io/anthropic"
MINIMAX_OAUTH_PORTAL_URL = "https://api.minimax.io"


def _setup_minimax_oauth_auth(
    hermes_home: Path,
    *,
    access_token: str = "mm_access",
    refresh_token: str = "mm_refresh",
    inference_base_url: str = MINIMAX_OAUTH_INFERENCE_URL,
) -> Path:
    """Write MiniMax OAuth state into the Hermes auth store at ``hermes_home``."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Set expiry well in the future so _refresh_minimax_oauth_state is a no-op.
    expires_dt = datetime.now(timezone.utc) + timedelta(hours=2)
    state = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "inference_base_url": inference_base_url,
        "portal_base_url": MINIMAX_OAUTH_PORTAL_URL,
        "client_id": "test-client-id",
        "region": "global",
        "obtained_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires_dt.isoformat(),
        "expires_in": 7200,
        "auth_mode": "oauth_pkce",
    }
    auth_store = {
        "version": 1,
        "active_provider": "minimax-oauth",
        "providers": {"minimax-oauth": state},
    }
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps(auth_store, indent=2))
    return auth_file


def _empty_auth_store(hermes_home: Path) -> Path:
    """Write an auth store with no providers configured."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps({"version": 1, "providers": {}}))
    return auth_file


# ---------------------------------------------------------------------------
# Routing contract tests
# ---------------------------------------------------------------------------


def test_auxiliary_client_routes_minimax_oauth_through_anthropic(
    tmp_path, monkeypatch
):
    """``resolve_provider_client("minimax-oauth", model)`` must return a
    non-None ``AnthropicAuxiliaryClient``.

    Without the minimax-oauth branch, the provider falls through every
    ``auth_type`` check and returns ``(None, None)``, silently re-routing
    every auxiliary task to whatever fallback chain the user has configured.
    """
    hermes_home = tmp_path / "hermes"
    _setup_minimax_oauth_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    client, model = resolve_provider_client(
        "minimax-oauth", model="MiniMax-M2.7"
    )
    assert client is not None, (
        "minimax-oauth must route to an AnthropicAuxiliaryClient; falling "
        "through silently swaps providers for every auxiliary task."
    )
    assert isinstance(client, AnthropicAuxiliaryClient)
    assert model == "MiniMax-M2.7"
    # MiniMax OAuth is a third-party Anthropic-compatible endpoint, NOT
    # Claude Code OAuth.  The is_oauth flag is Claude-Code-specific (it
    # injects Claude Code system-prompt identity + tool-name transforms via
    # _AnthropicCompletionsAdapter).  Pin is_oauth=False so those transforms
    # do not apply.  (PR #61585 review feedback.)
    assert client.chat.completions._is_oauth is False, (
        "is_oauth must be False for MiniMax OAuth — it is a third-party "
        "Anthropic-compatible endpoint, not Claude Code OAuth."
    )


def test_auxiliary_client_minimax_oauth_async_routes_through_anthropic(
    tmp_path, monkeypatch
):
    """Async mode must return an ``AsyncAnthropicAuxiliaryClient`` wrapping
    the same ``is_oauth=False`` adapter."""
    hermes_home = tmp_path / "hermes"
    _setup_minimax_oauth_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    client, model = resolve_provider_client(
        "minimax-oauth", model="MiniMax-M2.7", async_mode=True
    )
    assert client is not None
    assert isinstance(client, AsyncAnthropicAuxiliaryClient)
    assert model == "MiniMax-M2.7"
    # The async adapter delegates to the sync adapter, so the is_oauth flag
    # lives on the underlying _AnthropicCompletionsAdapter.
    assert client.chat.completions._sync._is_oauth is False


def test_auxiliary_client_minimax_oauth_returns_none_when_unauthenticated(
    tmp_path, monkeypatch
):
    """No MiniMax OAuth tokens in the auth store → must return ``(None, None)``
    so ``_resolve_auto`` falls through to the next provider in the chain."""
    hermes_home = tmp_path / "hermes"
    _empty_auth_store(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    client, model = resolve_provider_client(
        "minimax-oauth", model="MiniMax-M2.7"
    )
    assert client is None
    assert model is None


def test_auxiliary_client_minimax_oauth_no_openai_fallback_on_failure(
    tmp_path, monkeypatch
):
    """When Anthropic client construction fails, the builder must return
    ``(None, None)`` — NOT fall back to an OpenAI-wire client.

    The MiniMax ``/anthropic`` endpoint speaks Anthropic Messages, not OpenAI
    chat.completions.  A previous version of this code fell back to
    ``_create_openai_client`` on construction failure, which would emit
    misformatted OpenAI-wire requests to the wrong API format.  Returning
    ``(None, None)`` lets the caller's auto-fallback chain pick the next
    configured provider cleanly.  (PR #61585 review feedback.)
    """
    hermes_home = tmp_path / "hermes"
    _setup_minimax_oauth_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Force build_anthropic_client to raise a non-ImportError so the
    # construction-failure branch is exercised (not the ImportError branch).
    import agent.anthropic_adapter as _aa

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated SDK construction failure")

    monkeypatch.setattr(_aa, "build_anthropic_client", _boom)

    client, model = resolve_provider_client(
        "minimax-oauth", model="MiniMax-M2.7"
    )
    assert client is None, (
        "Construction failure must return (None, None), not an OpenAI-wire "
        "client — the /anthropic endpoint does not accept OpenAI wire format."
    )
    assert model is None
