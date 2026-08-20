"""Regression test for the minimax-oauth auxiliary routing bug.

Symptom: auxiliary tasks configured with ``provider: minimax-oauth`` (e.g.
``auxiliary.title_generation.provider: minimax-oauth``) raised:

    RuntimeError: Provider 'minimax-oauth' is set in config.yaml but no
    API key was found. Set the MINIMAX-OAUTH_API_KEY environment variable,
    or switch to a different provider with `hermes model`.

Root cause: ``resolve_provider_client()`` in
``agent/auxiliary_client.py`` had early branches for the ``nous``,
``openai-codex``, and ``xai-oauth`` OAuth providers, but no branch for
``minimax-oauth``.  Requests fell through to the generic
``pconfig.auth_type in {"oauth_device_code", "oauth_external"}`` arm
which doesn't list ``oauth_minimax``, and then to a final
``unhandled auth_type oauth_minimax`` warning that returns ``(None, None)``.
The caller then raised the misleading "no API key" error.

These tests verify the fix:

1. ``_build_minimax_oauth_aux_client`` returns a working client when
   auth.json has a valid MiniMax OAuth token.
2. ``_build_minimax_oauth_aux_client`` returns ``(None, None)`` when the
   user is not logged in (no entry in auth.json).
3. ``resolve_provider_client("minimax-oauth", ...)`` no longer falls
   through to the misleading "no API key" path.
4. The ``oauth_minimax`` auth_type routes through the dedicated branch
   (defense in depth — the early branch above is the primary path).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# ── helpers ────────────────────────────────────────────────────────────────


def _fake_auth_state(logged_in: bool = True) -> dict:
    """Mimic the shape of ``get_minimax_oauth_auth_status()``."""
    return {
        "logged_in": logged_in,
        "provider": "minimax-oauth",
        "region": "global",
    }


def _fake_oauth_creds() -> dict:
    """Mimic the shape of ``resolve_minimax_oauth_runtime_credentials()``."""
    return {
        "provider": "minimax-oauth",
        "api_key": "fake-access-token-abc123",
        "base_url": "https://api.minimax.io/anthropic",
        "source": "oauth",
    }


# ── _build_minimax_oauth_aux_client: happy path ─────────────────────────────


class TestBuildMinimaxOauthAuxClient:
    """Verify _build_minimax_oauth_aux_client returns a working client."""

    def _import(self):
        from agent.auxiliary_client import _build_minimax_oauth_aux_client
        return _build_minimax_oauth_aux_client

    def test_returns_client_when_logged_in(self):
        build = self._import()
        with patch(
            "hermes_cli.auth.get_minimax_oauth_auth_status",
            return_value=_fake_auth_state(logged_in=True),
        ), patch(
            "hermes_cli.auth.build_minimax_oauth_token_provider",
            return_value=lambda: "fresh-token",
        ), patch(
            "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
            return_value=_fake_oauth_creds(),
        ):
            client, model = build("MiniMax-M3")

        assert client is not None, "client must not be None when logged in"
        assert model == "MiniMax-M3", f"expected caller-provided model, got {model!r}"

    def test_returns_client_with_model_fallback(self):
        """When no model is provided, falls back to a sensible default."""
        build = self._import()
        with patch(
            "hermes_cli.auth.get_minimax_oauth_auth_status",
            return_value=_fake_auth_state(logged_in=True),
        ), patch(
            "hermes_cli.auth.build_minimax_oauth_token_provider",
            return_value=lambda: "fresh-token",
        ), patch(
            "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
            return_value=_fake_oauth_creds(),
        ):
            client, model = build(None)

        assert client is not None
        assert model, "model must be non-empty even when caller passed None"

    def test_returns_none_when_not_logged_in(self):
        """When auth.json has no MiniMax entry, return (None, None) cleanly."""
        build = self._import()
        with patch(
            "hermes_cli.auth.get_minimax_oauth_auth_status",
            return_value=_fake_auth_state(logged_in=False),
        ):
            client, model = build("MiniMax-M3")

        assert client is None
        assert model is None

    def test_returns_none_when_creds_resolution_raises(self):
        """If resolve_minimax_oauth_runtime_credentials raises AuthError,
        the helper must return (None, None) instead of propagating."""
        from hermes_cli.auth import AuthError

        build = self._import()
        with patch(
            "hermes_cli.auth.get_minimax_oauth_auth_status",
            return_value=_fake_auth_state(logged_in=True),
        ), patch(
            "hermes_cli.auth.build_minimax_oauth_token_provider",
            return_value=lambda: "token",
        ), patch(
            "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
            side_effect=AuthError("not logged in", provider="minimax-oauth"),
        ):
            client, model = build("MiniMax-M3")

        assert client is None
        assert model is None


# ── resolve_provider_client dispatch ───────────────────────────────────────


class TestResolveProviderClientMinimaxOauth:
    """Verify resolve_provider_client('minimax-oauth', ...) routes correctly.

    Before the fix, this would fall through to the
    ``unhandled auth_type oauth_minimax`` warning and return ``(None, None)``,
    which the caller then translated into the misleading "no API key" error.
    """

    def test_dispatches_to_minimax_oauth_helper(self):
        """The provider dispatch should reach the new branch instead of
        falling through to the generic OAuth arm."""
        from agent.auxiliary_client import resolve_provider_client

        fake_client = MagicMock(name="minimax_oauth_client")
        with patch(
            "agent.auxiliary_client._build_minimax_oauth_aux_client",
            return_value=(fake_client, "MiniMax-M3"),
        ) as mock_build:
            client, model = resolve_provider_client("minimax-oauth", "MiniMax-M3")

        mock_build.assert_called_once()
        assert client is fake_client, "should return the client built by the helper"
        assert model == "MiniMax-M3"

    def test_returns_none_cleanly_when_helper_returns_none(self):
        """When the helper returns (None, None) (user not logged in),
        resolve_provider_client should also return (None, None) — *not*
        raise the misleading "no API key" RuntimeError."""
        from agent.auxiliary_client import resolve_provider_client

        with patch(
            "agent.auxiliary_client._build_minimax_oauth_aux_client",
            return_value=(None, None),
        ):
            client, model = resolve_provider_client("minimax-oauth", "MiniMax-M3")

        assert client is None
        assert model is None

    def test_oauth_minimax_auth_type_branch_exists(self):
        """Defense in depth: the auth_type dispatch in
        resolve_provider_client should have an ``elif pconfig.auth_type
        == 'oauth_minimax'`` arm.  Source-grep test — the branch is
        short and unambiguous, and a future refactor is more likely to
        preserve the semantic than to preserve a mockable surface."""
        import inspect
        from agent.auxiliary_client import resolve_provider_client

        source = inspect.getsource(resolve_provider_client)
        assert 'pconfig.auth_type == "oauth_minimax"' in source, (
            "the oauth_minimax auth_type fallback has been removed or "
            "refactored; verify the minimax-oauth dispatch still works"
        )


# ── regression: the original error message must no longer fire ──────────────


class TestNoMisleadingApiKeyError:
    """Before the fix, callers raised:

        RuntimeError: Provider 'minimax-oauth' is set in config.yaml but
        no API key was found. Set the MINIMAX-OAUTH_API_KEY environment
        variable, or switch to a different provider with `hermes model`.

    This regression test asserts that the error string is *not* raised
    when the user is correctly authenticated with MiniMax OAuth.
    """

    def test_does_not_raise_api_key_error_when_logged_in(self):
        from agent.auxiliary_client import resolve_provider_client

        fake_client = MagicMock(name="minimax_oauth_client_no_error")
        with patch(
            "agent.auxiliary_client._build_minimax_oauth_aux_client",
            return_value=(fake_client, "MiniMax-M3"),
        ):
            try:
                client, model = resolve_provider_client("minimax-oauth", "MiniMax-M3")
            except RuntimeError as exc:
                msg = str(exc)
                assert "MINIMAX-OAUTH_API_KEY" not in msg, (
                    "regression: the misleading 'no API key' error is back"
                )
                raise

        assert client is fake_client
