"""Regression tests for ACP adapter detection under Azure Foundry Entra ID.

The ACP adapter's ``detect_provider`` previously gated on
``isinstance(api_key, str)`` and returned ``None`` for any runtime that
returned a callable ``api_key`` — i.e. Azure Foundry with
``auth_mode=entra_id``. Downstream, ACP would default to
``"openrouter"`` and reject the legitimate provider in its auth handshake.
This test pins the callable-aware fix so it never regresses.
"""

from __future__ import annotations

from unittest.mock import patch


class TestDetectProviderEntra:
    def test_callable_api_key_is_a_valid_credential(self):
        """A runtime returning a callable ``api_key`` (Entra bearer token
        provider) must be detected as a configured provider, not
        ``None``."""
        from acp_adapter import auth as _acp_auth

        def _fake_runtime(**_kwargs):
            return {
                "provider": "azure-foundry",
                "api_mode": "chat_completions",
                "auth_mode": "entra_id",
                "base_url": "https://r.openai.azure.com/openai/v1",
                "api_key": lambda: "jwt-fresh",
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_fake_runtime,
        ):
            assert _acp_auth.detect_provider() == "azure-foundry"
            assert _acp_auth.has_provider() is True

    def test_string_api_key_still_works(self):
        from acp_adapter import auth as _acp_auth

        def _fake_runtime(**_kwargs):
            return {
                "provider": "openrouter",
                "api_key": "sk-or-static-key",
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_fake_runtime,
        ):
            assert _acp_auth.detect_provider() == "openrouter"

    def test_empty_string_api_key_returns_none(self):
        from acp_adapter import auth as _acp_auth

        def _fake_runtime(**_kwargs):
            return {"provider": "openrouter", "api_key": ""}

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_fake_runtime,
        ):
            assert _acp_auth.detect_provider() is None

    def test_missing_provider_returns_none(self):
        """A callable api_key without a provider is still ``None`` —
        we don't synthesize a provider name from the credential shape."""
        from acp_adapter import auth as _acp_auth

        def _fake_runtime(**_kwargs):
            return {"api_key": lambda: "jwt-fresh", "provider": ""}

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_fake_runtime,
        ):
            assert _acp_auth.detect_provider() is None

    def test_resolver_exception_returns_none(self):
        from acp_adapter import auth as _acp_auth

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=RuntimeError("simulated"),
        ):
            assert _acp_auth.detect_provider() is None
