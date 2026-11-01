"""Tests for the minimax-oauth explicit branch in auxiliary_client.

``minimax-oauth`` (auth_type=``oauth_minimax``) previously fell through
``_EXPLICIT_PROVIDER_BRANCHES`` into the generic registry arm, which returns
``(None, None)`` for every OAuth provider.  The dedicated branch added in
``_resolve_minimax_oauth_branch`` routes it to an Anthropic-compatible
``AnthropicAuxiliaryClient`` so auxiliary tasks (compression, vision, title
generation, …) honour ``auxiliary.<task>.provider: minimax-oauth``.

These tests pin the wiring contract:

1. ``_EXPLICIT_PROVIDER_BRANCHES`` maps ``"minimax-oauth"`` to the new branch
   (the registry OAuth dead-end is no longer reachable for this provider).
2. ``_build_minimax_oauth_aux_client`` returns ``(None, None)`` without a model
   (mirrors the xai-oauth / codex contract — a pinned default would rot).
3. With a token-provider stub, the builder returns an ``AnthropicAuxiliaryClient``
   pointed at ``api.minimax.io/anthropic`` — never a raw OpenAI client (the
   endpoint is Anthropic-format only).
"""

import pytest


# ── Branch registration ────────────────────────────────────────────────────


class TestMinimaxOauthBranchRegistered:
    """``_EXPLICIT_PROVIDER_BRANCHES`` must route ``minimax-oauth`` explicitly."""

    def test_minimax_oauth_in_explicit_branches(self):
        from agent.auxiliary_client import _EXPLICIT_PROVIDER_BRANCHES
        assert "minimax-oauth" in _EXPLICIT_PROVIDER_BRANCHES

    def test_branch_callable_matches_resolve_function(self):
        from agent.auxiliary_client import (
            _EXPLICIT_PROVIDER_BRANCHES,
            _resolve_minimax_oauth_branch,
        )
        assert _EXPLICIT_PROVIDER_BRANCHES["minimax-oauth"] is _resolve_minimax_oauth_branch


# ── Builder contract ───────────────────────────────────────────────────────


class TestBuildMinimaxOauthAuxClient:
    """``_build_minimax_oauth_aux_client`` mirrors the xai-oauth builder contract."""

    def test_no_model_returns_none_none(self):
        from agent.auxiliary_client import _build_minimax_oauth_aux_client
        client, model = _build_minimax_oauth_aux_client("")
        assert client is None
        assert model is None

    def test_with_token_provider_returns_anthropic_adapter(self, monkeypatch):
        """When OAuth credentials resolve, the builder must produce an
        ``AnthropicAuxiliaryClient`` bound to the MiniMax Anthropic endpoint —
        NOT a raw OpenAI client (MiniMax serves /anthropic only)."""
        from agent.auxiliary_client import (
            _build_minimax_oauth_aux_client,
            AnthropicAuxiliaryClient,
        )

        def _fake_provider():
            return "sk-fake-token"

        monkeypatch.setattr(
            "hermes_cli.auth.build_minimax_oauth_token_provider",
            lambda: _fake_provider,
        )

        client, model = _build_minimax_oauth_aux_client("MiniMax-M3")

        assert client is not None
        assert isinstance(client, AnthropicAuxiliaryClient)
        assert model == "MiniMax-M3"
        assert client.base_url == "https://api.minimax.io/anthropic"

    def test_token_provider_failure_returns_none_none(self, monkeypatch):
        """A broken credential source must degrade to (None, None) — never raise."""
        from agent.auxiliary_client import _build_minimax_oauth_aux_client

        def _boom():
            raise RuntimeError("auth.json missing")

        monkeypatch.setattr(
            "hermes_cli.auth.build_minimax_oauth_token_provider",
            _boom,
        )

        client, model = _build_minimax_oauth_aux_client("MiniMax-M3")
        assert client is None
        assert model is None


# ── Branch routing ─────────────────────────────────────────────────────────


class TestResolveMinimaxOauthBranch:
    """``resolve_provider_client`` must not fall into the registry OAuth dead-end."""

    def test_resolve_returns_anthropic_client(self, monkeypatch):
        from agent.auxiliary_client import (
            resolve_provider_client,
            AnthropicAuxiliaryClient,
        )

        def _fake_provider():
            return "sk-fake-token"

        monkeypatch.setattr(
            "hermes_cli.auth.build_minimax_oauth_token_provider",
            lambda: _fake_provider,
        )

        client, model = resolve_provider_client(
            "minimax-oauth", model="MiniMax-M3", task="compression",
        )
        assert isinstance(client, AnthropicAuxiliaryClient)
        assert model == "MiniMax-M3"
