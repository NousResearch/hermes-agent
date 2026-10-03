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
  * OpenAI-compatible gateways that validate the body with a strict pydantic
    model reject the OBJECT-form ``response_format.json_schema`` by shape --
    ``422 ... body.response_format.json_schema: str type expected`` -- instead
    of naming the feature, so the error never mentions an unsupported option.

Callers tolerate an unconstrained reply: the title prompt demands bare JSON
and ``_extract_title_text`` has a loose-JSON fallback. The fix is reactive,
like the temperature retry: when the provider rejects the structured-output
field, retry once without it. These tests lock in that behaviour for both
sync and async paths.
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent import auxiliary_structured_output as structured_output
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
        # Strict pydantic gateway rejecting the object-form json_schema by SHAPE
        # (422, verbatim body from the gateway) -- no unsupported-parameter wording
        'HTTP 422: {"detail":[{"loc":["body","response_format","json_schema"],'
        '"msg":"str type expected","type":"type_error.str"}]}',
        # Gemini native generationConfig wording (verbatim 400s from generativelanguage)
        "Gemini HTTP 400 (INVALID_ARGUMENT): Function calling with a response mime type: 'application/json' is unsupported",
        "Gemini HTTP 400 (INVALID_ARGUMENT): Invalid JSON payload received. Unknown name \"response_json_schema\" at 'generation_config'",
        "Gemini HTTP 400 (INVALID_ARGUMENT): Invalid value at 'generation_config.response_schema.properties[0].value.type'",
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


# --- diagnostic-free rejections -------------------------------------------------------------
# A relay that spreads one model over several upstreams can refuse ``response_format`` with a payload
# that names no reason at all: an empty body, or a bare routing echo. opencode.ai/zen/go answers
# ``400 {"model": "deepseek-v4.1-flash"}`` for deepseek-v4.1-flash requests carrying the field
# (probed 2026-10-03: 23 of 24 fresh sessions 400'd, and every one of those sessions returned 200 on
# the same request without the field). Nothing in such a body matches the phrasings above, so the
# retry rung never fired: title generation hard-failed with
# ``Title generation failed: Error code: 400 - {'model': 'deepseek-v4.1-flash'}``.
_ECHO_ONLY_BODY = '{"model": "deepseek-v4.1-flash"}'
_BARE_BODY_400 = "Error code: 400"  # what the SDK renders for an EMPTY body: no separator, nothing after
_ECHO_ONLY_400 = f"Error code: 400 - {_ECHO_ONLY_BODY}"


def _status_error(text, status=400, raw_body=None):
    """An SDK-shaped rejection: ``str()`` renders the parsed body, ``.response.text`` the raw one.

    ``openai.APIStatusError`` built from an empty body renders the bare ``"Error code: 400"`` and keeps
    ``response.text == ""`` (``_base_client._make_status_error_from_response``), so both halves have to
    be reproduced to test the detector against what production raises.
    """
    exc = RuntimeError(text)
    exc.status_code = status
    exc.response = SimpleNamespace(text="" if raw_body is None else raw_body, status_code=status)
    return exc


class TestDiagnosticFreeRejectionDetection:
    """A payload that names nothing is a rejection on a 400/422 -- and only there."""

    @pytest.mark.parametrize("text,status,raw_body", [
        (_BARE_BODY_400, 400, ""),
        (_ECHO_ONLY_400, 400, _ECHO_ONLY_BODY),
        ('Error code: 422 - {"model": "m"}', 422, '{"model": "m"}'),
        ('Error code: 400 - {"model": "m", "id": "chatcmpl-1", "object": "error", "created": 1}',
         400, '{"model": "m", "id": "chatcmpl-1", "object": "error", "created": 1}'),
    ])
    def test_payload_that_names_no_reason_is_a_rejection(self, text, status, raw_body):
        assert _is_structured_output_rejection(_status_error(text, status, raw_body=raw_body)) is True

    @pytest.mark.parametrize("body", [
        # A body that names something is never assumed to be about the field.
        '{"error": {"message": "Rate limit exceeded"}}',
        '{"detail": "quota exhausted"}',
        '{"reason": "model is unavailable"}',
        '{"unexpected": "field"}',
        '{"error": {"message": "Upstream request failed: Model is unavailable."}}',
        # A code-only payload names something the caller can read and act on.
        '{"error_code": "missing_session_id"}',
        '{"type": "rate_limit_error"}',
        '{"code": "invalid_api_key"}',
        '{"status": "quota_exceeded"}',
    ])
    def test_payload_that_names_something_is_not(self, body):
        error = _status_error(f"Error code: 400 - {body}", raw_body=body)
        assert _is_structured_output_rejection(error) is False

    def test_unparsed_prose_is_not(self):
        """Unparsed prose may well name the problem -- never assume it does not."""
        assert _is_structured_output_rejection(
            _status_error("HTTP 400: upstream connection reset", raw_body="HTTP 400: upstream connection reset")
        ) is False

    @pytest.mark.parametrize("status", [401, 403, 404, 408, 429, 500, 502, 503])
    def test_a_silent_payload_on_another_status_is_not_a_rejection(self, status):
        assert _is_structured_output_rejection(
            _status_error(_BARE_BODY_400, status, raw_body="")) is False

    @pytest.mark.parametrize("status", [401, 429, 500, 503])
    def test_a_silent_body_does_not_slip_through_when_the_status_rides_on_the_response(self, status):
        """Some wrappers expose the status only on ``error.response``; the guard has to read it there
        too, or an empty-bodied 401/429/500 would be read as a field rejection."""
        error = RuntimeError("")  # no status_code attribute at all
        error.response = SimpleNamespace(text="", status_code=status)
        assert _is_structured_output_rejection(error) is False

    def test_repr_only_body_is_not_guessed_when_no_raw_response_is_exposed(self):
        """Without ``.response`` the payload reaches us as ``str(error)``, and a Python-repr body is not
        JSON -- so it reads as prose and the caller stays conservative. The OpenAI SDK always exposes
        ``.response`` (``_make_status_error_from_response``), which is the shape the tests above use."""
        assert _is_structured_output_rejection(
            RuntimeError("Error code: 400 - {'model': 'deepseek-v4.1-flash'}")) is False


class TestCallLlmDiagnosticFreeRetry:
    """The ladder must degrade to prompt compliance for a silent 400, exactly like a named one."""

    def _silent_client(self, *effects):
        client = MagicMock()
        client.base_url = "https://opencode.ai/zen/go/v1"
        client.chat.completions.create.side_effect = list(effects)
        return client

    def test_retries_once_without_response_format(self):
        client = self._silent_client(_status_error(_ECHO_ONLY_400, raw_body=_ECHO_ONLY_BODY), _dummy_response())

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("opencode-go", "deepseek-v4.1-flash", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "deepseek-v4.1-flash")),
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
        assert "response_format" in (first_kwargs.get("extra_body") or {})
        assert "response_format" not in (retry_kwargs.get("extra_body") or {})
        assert "response_format" not in retry_kwargs

    def test_silent_400_without_the_field_does_not_loop_a_retry(self):
        client = self._silent_client(_status_error(_BARE_BODY_400, raw_body=""))

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("opencode-go", "deepseek-v4.1-flash", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "deepseek-v4.1-flash")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._try_payment_fallback", return_value=None),
        ):
            with pytest.raises(RuntimeError):
                call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=64,
                )
        assert client.chat.completions.create.call_count == 1

    def test_empty_body_400_with_the_field_retries_and_succeeds(self):
        """The shape the relay answers most often: an EMPTY body, which the SDK renders as the bare
        ``"Error code: 400"``. It has to buy the same retry the echo-only body buys."""
        client = self._silent_client(_status_error(_BARE_BODY_400, raw_body=""), _dummy_response())

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("opencode-go", "deepseek-v4.1-flash", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "deepseek-v4.1-flash")),
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
        retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert "response_format" not in (retry_kwargs.get("extra_body") or {})

    def test_a_failed_stripped_retry_reports_that_failure_not_a_wrapper(self):
        """A diagnostic-free 400 buys exactly one retry. If the retry fails too, the caller sees the
        retry's own provider error (status and body intact) -- never a synthetic "retry failed"
        wrapper, and never a phantom success."""
        rate_limit = _status_error(
            'Error code: 429 - {"error": {"message": "Rate limit exceeded"}}', 429,
            raw_body='{"error": {"message": "Rate limit exceeded"}}')
        calls = {"n": 0}
        client = MagicMock()
        client.base_url = "https://opencode.ai/zen/go/v1"

        def create(**_kwargs):
            calls["n"] += 1
            raise _status_error(_BARE_BODY_400, raw_body="") if calls["n"] == 1 else rate_limit

        client.chat.completions.create.side_effect = create

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("opencode-go", "deepseek-v4.1-flash", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "deepseek-v4.1-flash")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._try_payment_fallback", return_value=None),
        ):
            with pytest.raises(RuntimeError) as caught:
                call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=64,
                    extra_body={"response_format": dict(_TITLE_RESPONSE_FORMAT)},
                )

        assert getattr(caught.value, "status_code", None) == 429
        assert "Rate limit exceeded" in str(caught.value)
        assert calls["n"] >= 2  # the silent 400 was retried without the field before failing
        assert structured_output._REJECTED_ROUTES == set()  # a failed retry must not condemn the route

    def test_a_successful_stripped_retry_memoises_the_route(self):
        """Integration: the ladder's success path feeds the memo, so the next structured call on that
        route+model drops the field up front instead of paying a second wasted 400."""
        client = self._silent_client(_status_error(_ECHO_ONLY_400, raw_body=_ECHO_ONLY_BODY), _dummy_response())

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("opencode-go", "deepseek-v4.1-flash", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "deepseek-v4.1-flash")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
        ):
            call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"response_format": dict(_TITLE_RESPONSE_FORMAT)},
            )

        assert any(entry[1] == "deepseek-v4.1-flash" for entry in structured_output._REJECTED_ROUTES)
        scrubbed = structured_output.without_unsupported_response_format(
            {"response_format": dict(_TITLE_RESPONSE_FORMAT)},
            "opencode-go", "https://opencode.ai/zen/go/v1", "deepseek-v4.1-flash")
        assert "response_format" not in scrubbed


class TestAsyncDiagnosticFreeRetry:
    """``async_call_llm`` parity: the async ladder takes the same rung for a silent 400."""

    @pytest.mark.asyncio
    async def test_async_retries_once_without_response_format(self):
        client = MagicMock()
        client.base_url = "https://opencode.ai/zen/go/v1"
        client.chat.completions.create = AsyncMock(
            side_effect=[_status_error(_BARE_BODY_400, raw_body=""), _dummy_response()])

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("opencode-go", "deepseek-v4.1-flash", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "deepseek-v4.1-flash")),
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
