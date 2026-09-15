"""Regression tests for the structured-output rejection retry in
``agent.auxiliary_client``.

Auxiliary callers (title generation, plugin structured completions) send an
OpenAI ``response_format`` request field. Some providers reject the field, or
its Anthropic translation, with a hard 400:

  * vLLM gateways translate ``response_format: json_schema`` into
    ``guided_grammar`` and fail when the grammar backend is absent
    (``compile_grammar_error: No module named 'xgrammar'``, #82816).
  * Some OpenAI-compatible endpoints answer
    ``This response_format type is unavailable now`` (#82816).
  * Anthropic-compatible gateways that predate structured outputs reject the
    translated ``output_config`` field with
    ``output_config: Extra inputs are not permitted`` (the documented case is
    the ``bedrock-mantle`` Messages endpoint).

Callers tolerate an unconstrained reply: the title prompt demands bare JSON
and ``_extract_title_text`` has a loose-JSON fallback. The fix is reactive,
like the temperature retry: when the provider rejects the structured-output
field, retry once without it. These tests lock in that behaviour for both
sync and async paths.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    _is_structured_output_rejection,
    _without_structured_output_format,
)


_TITLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "session_title",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}


class TestIsStructuredOutputRejection:
    """The detector must match the phrasings providers actually return."""

    @pytest.mark.parametrize("message", [
        # vLLM guided_grammar / xgrammar (#82816, verbatim from the report)
        (
            "Error code: 400 - {'error': {'message': 'guided_grammar "
            '\'{"additionalProperties":false}\' has compile_grammar_error: '
            "No module named 'xgrammar'', 'type': 'invalid_request_error'}}"
        ),
        # Second endpoint from the same report
        "HTTP 400: This response_format type is unavailable now",
        # Strict Anthropic-wire gateways rejecting the raw OpenAI field
        "HTTP 400: response_format: Extra inputs are not permitted",
        # Gateways that predate output_config (bedrock-mantle documented case)
        "HTTP 400: output_config: Extra inputs are not permitted",
        # Generic unsupported-parameter phrasings for both field names
        "Unsupported parameter: response_format",
        "output_config is not supported",
    ])
    def test_matches_real_provider_messages(self, message):
        assert _is_structured_output_rejection(RuntimeError(message)) is True

    @pytest.mark.parametrize("message", [
        # Unrelated 400s must NOT trigger a silent schema downgrade
        "HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant'",
        "HTTP 400: Unsupported parameter: temperature",
        "max_tokens is too large for this model",
        "Rate limit exceeded",
        "Connection reset by peer",
        # Alternation errors that happen to mention messages
        "messages: Extra inputs are not permitted",
    ])
    def test_does_not_match_unrelated_errors(self, message):
        assert _is_structured_output_rejection(RuntimeError(message)) is False

    def test_does_not_match_non_400_statuses(self):
        exc = RuntimeError("output_config: Extra inputs are not permitted")
        exc.status_code = 500
        assert _is_structured_output_rejection(exc) is False


class TestWithoutStructuredOutputFormat:
    """The kwargs scrubber removes the field on both call shapes."""

    def test_removes_extra_body_entry_and_keeps_siblings(self):
        kwargs = {
            "model": "m",
            "extra_body": {
                "response_format": dict(_TITLE_RESPONSE_FORMAT),
                "metadata": {"user_id": "u1"},
            },
        }
        result = _without_structured_output_format(kwargs)
        assert result is not None
        assert result["extra_body"] == {"metadata": {"user_id": "u1"}}
        # The input dict is not mutated.
        assert "response_format" in kwargs["extra_body"]

    def test_drops_extra_body_entirely_when_it_becomes_empty(self):
        kwargs = {
            "model": "m",
            "extra_body": {"response_format": dict(_TITLE_RESPONSE_FORMAT)},
        }
        result = _without_structured_output_format(kwargs)
        assert result is not None
        assert "extra_body" not in result

    def test_removes_top_level_kwarg(self):
        kwargs = {"model": "m", "response_format": dict(_TITLE_RESPONSE_FORMAT)}
        result = _without_structured_output_format(kwargs)
        assert result is not None
        assert "response_format" not in result

    def test_returns_none_when_nothing_to_remove(self):
        assert _without_structured_output_format({"model": "m"}) is None
        assert _without_structured_output_format(
            {"model": "m", "extra_body": {"metadata": {}}}
        ) is None


def _dummy_response():
    return {"ok": True}


class TestCallLlmStructuredOutputRetry:
    """``call_llm`` retries once without the field and returns on success."""

    def _setup(self, first_exc):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create.side_effect = [
            first_exc, _dummy_response(),
        ]
        return client

    @pytest.mark.parametrize("error_message", [
        # vLLM guided_grammar (#82816)
        "Error code: 400 - guided_grammar has compile_grammar_error: "
        "No module named 'xgrammar'",
        # Second endpoint flavor from the same report
        "HTTP 400: This response_format type is unavailable now",
        # Strict gateway that rejects the translated Anthropic field
        "HTTP 400: output_config: Extra inputs are not permitted",
    ])
    def test_retries_once_without_response_format(self, error_message):
        client = self._setup(RuntimeError(error_message))

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
        ):
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"response_format": dict(_TITLE_RESPONSE_FORMAT)},
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        first_eb = first_kwargs.get("extra_body") or {}
        retry_eb = retry_kwargs.get("extra_body") or {}
        assert "response_format" in first_eb
        assert "response_format" not in retry_eb
        assert "response_format" not in retry_kwargs
        assert retry_kwargs["model"] == first_kwargs["model"]

    def test_unrelated_400_does_not_strip_response_format(self):
        """Unrelated 400s must not silently downgrade the schema contract."""
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create.side_effect = RuntimeError(
            "HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant'"
        )

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(RuntimeError, match="Invalid value"):
                call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=64,
                    extra_body={
                        "response_format": dict(_TITLE_RESPONSE_FORMAT),
                    },
                )
        assert client.chat.completions.create.call_count == 1

    def test_no_retry_when_no_response_format_was_sent(self):
        """A rejection with no field in the request must not loop a retry."""
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create.side_effect = RuntimeError(
            "HTTP 400: output_config: Extra inputs are not permitted"
        )

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(RuntimeError):
                call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=64,
                )
        assert client.chat.completions.create.call_count == 1


class TestAsyncCallLlmStructuredOutputRetry:
    """``async_call_llm`` mirror of the sync retry semantics."""

    @pytest.mark.asyncio
    async def test_async_retries_once_without_response_format(self):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create = AsyncMock(side_effect=[
            RuntimeError(
                "Error code: 400 - guided_grammar has compile_grammar_error: "
                "No module named 'xgrammar'"
            ),
            _dummy_response(),
        ])

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
        ):
            result = await async_call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"response_format": dict(_TITLE_RESPONSE_FORMAT)},
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.await_count == 2
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert "response_format" in (first_kwargs.get("extra_body") or {})
        assert "response_format" not in (retry_kwargs.get("extra_body") or {})
        assert "response_format" not in retry_kwargs

    @pytest.mark.asyncio
    async def test_async_unrelated_400_does_not_retry(self):
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
                  side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(RuntimeError, match="Invalid value"):
                await async_call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=64,
                    extra_body={
                        "response_format": dict(_TITLE_RESPONSE_FORMAT),
                    },
                )
        assert client.chat.completions.create.await_count == 1
