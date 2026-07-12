"""Regression tests for the generic unsupported-parameter detector in
``agent.auxiliary_client``.

The original temperature-specific detector (PR #15621) was generalized so the
same reactive-retry strategy covers any provider that rejects an arbitrary
request parameter — ``max_tokens``, ``seed``, ``top_p``, future quirks — not
just ``temperature``. Credit @nicholasrae (PR #15416) for the generalization
pattern.

These tests lock in:
  * ``_is_unsupported_parameter_error(exc, param)`` across common phrasings
  * the back-compat wrapper ``_is_unsupported_temperature_error`` still works
  * the max_tokens retry branch no longer pops a key that was never set
    (``max_tokens is None`` gate)
  * the max_tokens retry branch matches via the generic helper on top of the
    legacy ``"max_tokens"`` / ``"unsupported_parameter"`` substring checks
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    _is_unsupported_parameter_error,
    _is_unsupported_temperature_error,
)


class TestIsUnsupportedParameterError:
    """The generic detector must match real provider phrasings for any param."""

    @pytest.mark.parametrize("param,message", [
        # temperature phrasings (regression coverage via the generic API)
        ("temperature", "HTTP 400: Unsupported parameter: temperature"),
        ("temperature", "Error code: 400 - {'error': {'code': 'unsupported_parameter', 'param': 'temperature'}}"),
        ("temperature", "this model does not support temperature"),
        # max_tokens phrasings
        ("max_tokens", "HTTP 400: Unsupported parameter: max_tokens"),
        ("max_tokens", "Unknown parameter: max_tokens — use max_completion_tokens"),
        ("max_tokens", "Invalid parameter: max_tokens is not supported"),
        # arbitrary future params
        ("seed", "HTTP 400: unrecognized parameter: seed"),
        ("top_p", "Error: top_p is not supported for this model"),
    ])
    def test_matches_real_provider_messages(self, param, message):
        assert _is_unsupported_parameter_error(RuntimeError(message), param) is True

    @pytest.mark.parametrize("param,message", [
        # Param not mentioned at all
        ("temperature", "HTTP 400: max_tokens is too large"),
        # Param mentioned but not flagged as unsupported
        ("temperature", "temperature must be between 0 and 2"),
        # Totally unrelated 400
        ("max_tokens", "Rate limit exceeded"),
        # Connection-level errors
        ("temperature", "Connection reset by peer"),
    ])
    def test_does_not_match_unrelated_errors(self, param, message):
        assert _is_unsupported_parameter_error(RuntimeError(message), param) is False

    def test_empty_param_returns_false(self):
        assert _is_unsupported_parameter_error(
            RuntimeError("HTTP 400: Unsupported parameter: temperature"), ""
        ) is False

    def test_temperature_wrapper_delegates_to_generic(self):
        """Back-compat: ``_is_unsupported_temperature_error`` still routes through."""
        msg = "HTTP 400: Unsupported parameter: temperature"
        assert _is_unsupported_temperature_error(RuntimeError(msg)) is True
        # And the unrelated-case still holds
        assert _is_unsupported_temperature_error(
            RuntimeError("max_tokens is too large")) is False


def _dummy_response():
    """Sentinel — real code calls ``_validate_llm_response`` which we patch out."""
    return {"ok": True}


class TestMaxTokensRetryHardening:
    """The max_tokens retry branch now (a) gates on ``max_tokens is not None``
    and (b) also matches the generic phrasings via the helper.
    """

    def test_sync_max_tokens_retry_skipped_when_max_tokens_is_none(self):
        """No max_tokens kwarg → must not pop/retry even if the error mentions it.

        Before the hardening, ``kwargs.pop("max_tokens", None)`` was safe but
        ``kwargs["max_completion_tokens"] = max_tokens`` would set a None
        value and hit the provider again. The gate skips the whole branch.
        """
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        err = RuntimeError("HTTP 400: Unsupported parameter: max_tokens")
        client.chat.completions.create.side_effect = err

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task: resp),
        ):
            with pytest.raises(RuntimeError):
                call_llm(
                    task="session_search",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.3,
                    # max_tokens omitted on purpose
                )

        # Only the initial attempt — no retry because the gate blocked it
        assert client.chat.completions.create.call_count == 1


    @pytest.mark.asyncio
    async def test_async_max_tokens_retry_skipped_when_max_tokens_is_none(self):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        err = RuntimeError("HTTP 400: Unsupported parameter: max_tokens")
        client.chat.completions.create = AsyncMock(side_effect=err)

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task: resp),
        ):
            with pytest.raises(RuntimeError):
                await async_call_llm(
                    task="session_search",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.3,
                )

        assert client.chat.completions.create.call_count == 1

