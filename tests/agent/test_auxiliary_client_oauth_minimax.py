"""Regression tests: oauth_minimax auth_type must route to the MiniMax OAuth
auxiliary client instead of falling through to the unhandled-auth_type
(None, None) return (#21521, #45241, #45242, #58231, #36091, #29910, #38685).

The minimax-oauth provider uses the Anthropic Messages wire with a
callable MiniMax OAuth token provider. Before the fix, the dispatch block
at resolve_provider_client only handled oauth_device_code / oauth_external,
so oauth_minimax fell through to 'unhandled auth_type' and every auxiliary
task (compression, vision, title_generation, session_search, ...) returned
(None, None).
"""

from unittest.mock import patch

import pytest


def _import_resolve():
    from agent.auxiliary_client import resolve_provider_client

    return resolve_provider_client


class TestOauthMinimaxAuthTypeDispatch:
    """oauth_minimax auth_type routes to the minimax-oauth builder."""

    @pytest.fixture(autouse=True)
    def _import(self):
        self.resolve = _import_resolve()

    def test_oauth_minimax_dispatch_returns_client(self, monkeypatch):
        """resolve_provider_client(minimax-oauth, ...) must build a client via
        _build_minimax_oauth_aux_client, not return (None, None)."""
        class _FakeClient:
            def __init__(self, model):
                self.model = model

        with patch(
            "agent.auxiliary_client._build_minimax_oauth_aux_client",
            return_value=(_FakeClient("minimax-m2.7"), "minimax-m2.7"),
        ) as mock_build:
            client, model = self.resolve("minimax-oauth", "minimax/minimax-m2.7")
        mock_build.assert_called_once()
        assert client is not None
        assert model == "minimax/minimax-m2.7"

    def test_oauth_minimax_unauthenticated_returns_none_none(self, monkeypatch):
        """No valid token -> (None, None), matching the other OAuth branches."""
        with patch(
            "agent.auxiliary_client._build_minimax_oauth_aux_client",
            return_value=(None, None),
        ):
            client, model = self.resolve("minimax-oauth", "minimax/minimax-m2.7")
        assert client is None
        assert model is None


class TestBuildMinimaxOauthAuxClient:
    """The real builder path: AnthropicAuxiliaryClient, is_oauth=False,
    profile model fallback — the semantics that make MiniMax work as a
    third-party Anthropic-compatible endpoint."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agent.auxiliary_client import _build_minimax_oauth_aux_client

        self.build = _build_minimax_oauth_aux_client

    def _patch_creds(self, monkeypatch, api_key="fake-token-callable"):
        from hermes_cli.auth import AuthError

        def _resolve_creds(*, min_token_ttl_seconds=None, as_token_provider=False):
            return {
                "provider": "minimax-oauth",
                "api_key": (lambda: api_key) if as_token_provider else api_key,
                "base_url": "https://api.minimax.io/anthropic/",
                "source": "oauth",
            }

        def _raise_no_auth(*, min_token_ttl_seconds=None, as_token_provider=False):
            raise AuthError(
                "Not logged into MiniMax OAuth.", provider="minimax-oauth",
                code="not_logged_in", relogin_required=True,
            )

        # The builder imports resolve_minimax_oauth_runtime_credentials fresh
        # from hermes_cli.auth at call time — patch the source module.
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
            _resolve_creds,
        )
        return _raise_no_auth

    def test_builds_anthropic_auxiliary_client_with_is_oauth_false(
        self, monkeypatch,
    ):
        """The wrapper must carry is_oauth=False: MiniMax is a third-party
        Anthropic-compatible endpoint and must never trip the Claude Code
        identity OAuth code paths (agent_init.py guard, #1739)."""
        class _FakeRealClient:
            def close(self):
                pass

        def _fake_build_anthropic_client(api_key, base_url, **kwargs):
            assert callable(api_key), "token provider must stay callable"
            return _FakeRealClient()

        self._patch_creds(monkeypatch)
        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client",
            _fake_build_anthropic_client,
        )

        from agent.auxiliary_client import AnthropicAuxiliaryClient

        client, model = self.build("minimax/minimax-m2.7")
        assert isinstance(client, AnthropicAuxiliaryClient)
        assert client.chat.completions._is_oauth is False
        assert client.base_url == "https://api.minimax.io/anthropic"
        assert model == "minimax/minimax-m2.7"

    def test_unauthenticated_returns_none_none(self, monkeypatch):
        """resolve_minimax_oauth_runtime_credentials raising AuthError
        propagates as (None, None) — the same contract as xai-oauth."""
        from hermes_cli.auth import AuthError

        def _raise(*, min_token_ttl_seconds=None, as_token_provider=False):
            raise AuthError(
                "Not logged into MiniMax OAuth.", provider="minimax-oauth",
                code="not_logged_in", relogin_required=True,
            )

        monkeypatch.setattr(
            "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
            _raise,
        )
        client, model = self.build("minimax/minimax-m2.7")
        assert client is None
        assert model is None

    def test_model_falls_back_to_provider_profile_default(self, monkeypatch):
        """No explicit model -> the provider profile's default_aux_model
        (MiniMax-M2.7) is used, so the auto-detection chain (#29910) and
        auxiliary.<task>.provider: minimax-oauth with empty model work."""
        class _FakeRealClient:
            def close(self):
                pass

        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client",
            lambda api_key, base_url, **kwargs: _FakeRealClient(),
        )
        self._patch_creds(monkeypatch)

        client, model = self.build(None)
        assert client is not None
        assert model == "MiniMax-M2.7"
