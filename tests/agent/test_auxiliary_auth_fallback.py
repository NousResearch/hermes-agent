"""Regression tests for 401 auth-error fallback in auxiliary_client.

When the primary provider returns 401 and the auth refresh either fails or
is not applicable (non-Nous provider), the fallback chain should try
alternative providers instead of raising.

Regression: https://github.com/NousResearch/hermes-agent/issues/21165
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSyncAuthErrorFallback:
    """call_llm should fall back to alternative providers on 401."""

    def test_auth_error_triggers_fallback_in_auto_mode(self, monkeypatch):
        """401 on primary provider → fallback to next provider in chain."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        err = Exception("401 Unauthorized")
        err.status_code = 401  # type: ignore[attr-defined]

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="fallback response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
        ) as mock_resolve, patch(
            "agent.auxiliary_client._refresh_nous_auxiliary_client",
            return_value=(None, None),  # auth refresh fails
        ), patch(
            "agent.auxiliary_client._try_payment_fallback",
            return_value=(fallback_client, "fallback-model", "openrouter"),
        ) as mock_fallback:
            # Primary client raises 401
            primary_client = MagicMock()
            primary_client.chat.completions.create.side_effect = err
            mock_resolve.return_value = (primary_client, "test-model")

            from agent.auxiliary_client import call_llm

            result = call_llm(
                messages=[{"role": "user", "content": "test"}],
                task="compression",
            )

        # Fallback should have been called with reason="auth error"
        mock_fallback.assert_called_once()
        assert mock_fallback.call_args.kwargs.get("reason") == "auth error" or \
               mock_fallback.call_args[1].get("reason") == "auth error" or \
               "auth error" in str(mock_fallback.call_args)

    def test_auth_error_no_fallback_when_explicit_provider(self, monkeypatch):
        """401 with explicit provider → should NOT fall back (is_auto=False)."""
        err = Exception("401 Unauthorized")
        err.status_code = 401  # type: ignore[attr-defined]

        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("nous", None, None, None, None),  # explicit
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
        ) as mock_resolve, patch(
            "agent.auxiliary_client._refresh_nous_auxiliary_client",
            return_value=(None, None),
        ):
            primary_client = MagicMock()
            primary_client.chat.completions.create.side_effect = err
            mock_resolve.return_value = (primary_client, "test-model")

            from agent.auxiliary_client import call_llm

            with pytest.raises(Exception, match="401"):
                call_llm(
                    messages=[{"role": "user", "content": "test"}],
                    task="compression",
                )

    def test_payment_error_still_works(self, monkeypatch):
        """402 payment error → fallback still works (regression guard)."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        err = Exception("402 Payment Required")
        err.status_code = 402  # type: ignore[attr-defined]

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="fallback"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
        ) as mock_resolve, patch(
            "agent.auxiliary_client._try_payment_fallback",
            return_value=(fallback_client, "fb-model", "openrouter"),
        ) as mock_fallback:
            primary_client = MagicMock()
            primary_client.chat.completions.create.side_effect = err
            mock_resolve.return_value = (primary_client, "test-model")

            from agent.auxiliary_client import call_llm

            call_llm(
                messages=[{"role": "user", "content": "test"}],
                task="compression",
            )

        mock_fallback.assert_called_once()
        args, kwargs = mock_fallback.call_args
        assert kwargs.get("reason") == "payment error" or \
               "payment error" in str(mock_fallback.call_args)


class TestAsyncAuthErrorFallback:
    """async_call_llm should fall back to alternative providers on 401."""

    @pytest.mark.asyncio
    async def test_auth_error_triggers_fallback_in_auto_mode(self, monkeypatch):
        """401 on primary provider → async fallback to next provider."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        err = Exception("401 Unauthorized")
        err.status_code = 401  # type: ignore[attr-defined]

        # Build an async-capable fallback client
        mock_response = MagicMock(
            choices=[MagicMock(message=MagicMock(content="fallback"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        fallback_client = MagicMock()
        fallback_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
        ) as mock_resolve, patch(
            "agent.auxiliary_client._refresh_nous_auxiliary_client",
            return_value=(None, None),
        ), patch(
            "agent.auxiliary_client._try_payment_fallback",
            return_value=(fallback_client, "fb-model", "openrouter"),
        ) as mock_fallback, patch(
            "agent.auxiliary_client._to_async_client",
            return_value=(fallback_client, "fb-model"),
        ):
            primary_client = MagicMock()
            primary_client.chat.completions.create.side_effect = err
            mock_resolve.return_value = (primary_client, "test-model")

            from agent.auxiliary_client import async_call_llm

            await async_call_llm(
                messages=[{"role": "user", "content": "test"}],
                task="compression",
            )

        mock_fallback.assert_called_once()
        args, kwargs = mock_fallback.call_args
        assert kwargs.get("reason") == "auth error" or \
               "auth error" in str(mock_fallback.call_args)
