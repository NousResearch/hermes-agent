"""Regression tests for the universal "unsupported temperature" retry in
``agent.auxiliary_client``.

Auxiliary callers (context compression, session search,
web extract summarisation, etc.) hardcode ``temperature=0.3`` for historical
reasons. Several provider/model combinations reject ``temperature`` with a
400:

  * OpenAI Responses (gpt-5/o-series reasoning models)
  * Copilot Responses (reasoning models)
  * OpenRouter reasoning models (gpt-5.5, some anthropic via OAI-compat)
  * Anthropic Opus 4.7+ via OpenAI-compat endpoints
  * Kimi/Moonshot (server-managed)

``_fixed_temperature_for_model`` catches Kimi up front, and
``build_chat_completion_kwargs`` drops temperature for Anthropic Opus 4.7+,
but the same backend can accept ``temperature`` for some models and reject
it for others (for example gpt-5.4 accepts but gpt-5.5 rejects on the same
endpoint). An allow/deny-list is not maintainable across providers.

The universal fix is reactive: when a call returns an
``Unsupported parameter: temperature`` 400, retry once without temperature.
These tests lock in that behaviour for both sync and async paths.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    _is_unsupported_temperature_error,
)


class TestIsUnsupportedTemperatureError:
    """The detector must match the phrasings providers actually return."""

    @pytest.mark.parametrize("message", [
        # OpenAI / Codex Responses
        "HTTP 400: Unsupported parameter: temperature",
        "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'temperature'\"}}",
        # Copilot / OpenAI error-code form
        "Error code: 400 - {'error': {'code': 'unsupported_parameter', 'param': 'temperature'}}",
        # OpenRouter-style
        "Provider returned error: temperature is not supported for this model",
        "this model does not support temperature",
        # Anthropic-style via OAI-compat
        "temperature: unknown parameter",
        # Some gateways
        "unrecognized request argument supplied: temperature",
    ])
    def test_matches_real_provider_messages(self, message):
        assert _is_unsupported_temperature_error(RuntimeError(message)) is True

    @pytest.mark.parametrize("message", [
        # Unrelated 400s must NOT trigger a silent-retry
        "HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant'...",
        "max_tokens is too large for this model",
        "Rate limit exceeded",
        "Connection reset by peer",
        # Temperature value error is a different class of problem
        "temperature must be between 0 and 2",
    ])
    def test_does_not_match_unrelated_errors(self, message):
        assert _is_unsupported_temperature_error(RuntimeError(message)) is False


def _dummy_response():
    # The real code calls _validate_llm_response which inspects
    # response.choices[0].message.  The tests here patch that out, so
    # any sentinel object is fine.
    return {"ok": True}


class TestCallLlmUnsupportedTemperatureRetry:
    """``call_llm`` retries once without temperature and returns on success."""

    def _setup(self, first_exc):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create.side_effect = [first_exc, _dummy_response()]
        return client

    @pytest.mark.parametrize("error_message", [
        "HTTP 400: Unsupported parameter: temperature",
        "Error code: 400 - {'error': {'code': 'unsupported_parameter', 'param': 'temperature'}}",
        "Provider error: this model does not support temperature",
    ])
    def test_retries_once_without_temperature(self, error_message):
        client = self._setup(RuntimeError(error_message))

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task: resp),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "remember this"}],
                temperature=0.3,
                max_tokens=500,
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert first_kwargs["temperature"] == 0.3
        assert "temperature" not in retry_kwargs
        # other kwargs preserved
        assert retry_kwargs["max_tokens"] == 500

    def test_non_temperature_400_does_not_retry_as_temperature(self):
        """Unrelated 400s (e.g. bad tool role) must not silently drop temp."""
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        non_temp_err = RuntimeError(
            "HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant'..."
        )
        client.chat.completions.create.side_effect = non_temp_err

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(RuntimeError, match="Invalid value"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "x"}],
                    temperature=0.3,
                    max_tokens=500,
                )
        # Should NOT have retried (non-temperature 400 doesn't match)
        assert client.chat.completions.create.call_count == 1

    def test_no_retry_when_temperature_not_in_kwargs(self):
        """If caller didn't send temperature, don't invent a temperature-retry."""
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        # Provider complains about temperature even though we didn't send it.
        # (Pathological but possible with misleading error text.)  The guard
        # ``"temperature" in kwargs`` must prevent an unnecessary retry.
        err = RuntimeError("HTTP 400: Unsupported parameter: temperature")
        client.chat.completions.create.side_effect = err

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(RuntimeError):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "x"}],
                    temperature=None,  # explicit: no temperature sent
                    max_tokens=500,
                )
        assert client.chat.completions.create.call_count == 1


class TestAsyncCallLlmUnsupportedTemperatureRetry:
    """``async_call_llm`` mirror of the sync retry semantics."""

    @pytest.mark.asyncio
    async def test_async_retries_once_without_temperature(self):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create = AsyncMock(side_effect=[
            RuntimeError("HTTP 400: Unsupported parameter: temperature"),
            _dummy_response(),
        ])

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task: resp),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "query"}],
                temperature=0.3,
                max_tokens=500,
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.await_count == 2
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert first_kwargs["temperature"] == 0.3
        assert "temperature" not in retry_kwargs
        assert retry_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_async_non_temperature_400_does_not_retry(self):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("HTTP 400: Invalid value: 'tool'"),
        )

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(RuntimeError, match="Invalid value"):
                await async_call_llm(
                    task="session_search",
                    messages=[{"role": "user", "content": "x"}],
                    temperature=0.3,
                    max_tokens=500,
                )
        assert client.chat.completions.create.await_count == 1
