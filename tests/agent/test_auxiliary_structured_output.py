"""Structured-output (``response_format``) handling on auxiliary routes that reject it.

Covers NousResearch/hermes-agent#83390 / #105191 / #113064: a fallback candidate that rejects
``json_schema`` gets the same retry-without-the-field rung as the primary path instead of aborting
the task, and a route known to reject a ``response_format`` type (provider profile or a rejection
already seen in this process) never pays the guaranteed-fail first request.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import auxiliary_structured_output as structured_output
from agent.auxiliary_client import (
    _build_call_kwargs,
    _call_fallback_candidate_async,
    _call_fallback_candidate_sync,
)

_JSON_SCHEMA = {"type": "json_schema", "json_schema": {"name": "t", "strict": True, "schema": {"type": "object"}}}
_DEEPSEEK_400 = ("Error code: 400 - {'error': {'message': 'This response_format type is unavailable now', "
                 "'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}")


class _Rejects400(Exception):
    status_code = 400


def _ok_response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"title": "x"}'))])


def _rejecting_client(base_url, *, async_mode=False):
    """Fake OpenAI client: 400 while ``extra_body.response_format`` is present, 200 once it is gone."""
    client = MagicMock(base_url=base_url)
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        if "response_format" in (kwargs.get("extra_body") or {}):
            raise _Rejects400(_DEEPSEEK_400)
        return _ok_response()

    async def acreate(**kwargs):
        return create(**kwargs)

    client.chat.completions.create.side_effect = acreate if async_mode else create
    return client, sent


@pytest.mark.parametrize("async_mode", [False, True])
def test_fallback_candidate_retries_without_response_format_on_rejection(async_mode):
    """The fallback path used to re-raise the 400; now it degrades to prompt compliance like the primary path."""
    client, sent = _rejecting_client("http://relay.example:8080/v1", async_mode=async_mode)
    call = _call_fallback_candidate_async if async_mode else _call_fallback_candidate_sync
    common = dict(
        task="title_generation", messages=[{"role": "user", "content": "Reply with JSON only"}],
        temperature=None, max_tokens=64, tools=None, effective_timeout=30.0,
        effective_extra_body={"response_format": dict(_JSON_SCHEMA)}, reasoning_config=None,
    )
    result = call(client, "relay-model", "fallback_chain[0](custom)", **common)
    if async_mode:
        result = asyncio.run(result)

    assert result.choices[0].message.content == '{"title": "x"}'
    assert [("response_format" in (k.get("extra_body") or {})) for k in sent] == [True, False]
    # The rejection is remembered: the next request to this route skips the field up front.
    retry_kwargs = _build_call_kwargs(
        "custom", "relay-model", [{"role": "user", "content": "hi"}],
        extra_body={"response_format": dict(_JSON_SCHEMA)}, base_url="http://relay.example:8080/v1")
    assert "response_format" not in retry_kwargs.get("extra_body", {})


def test_known_unsupported_route_skips_response_format_before_first_request():
    """DeepSeek's profile declares json_schema unsupported; json_object and other providers are untouched."""
    messages = [{"role": "user", "content": "hi"}]
    deepseek = _build_call_kwargs(
        "deepseek", "deepseek-flash", messages, extra_body={"response_format": dict(_JSON_SCHEMA)},
        base_url="https://api.deepseek.com/v1", task="title_generation")
    assert "response_format" not in deepseek.get("extra_body", {})

    json_object = _build_call_kwargs(
        "deepseek", "deepseek-flash", messages, extra_body={"response_format": {"type": "json_object"}},
        base_url="https://api.deepseek.com/v1")
    assert json_object["extra_body"]["response_format"] == {"type": "json_object"}

    openai_kwargs = _build_call_kwargs(
        "openai", "gpt-5-mini", messages, extra_body={"response_format": dict(_JSON_SCHEMA)},
        base_url="https://api.openai.com/v1")
    assert openai_kwargs["extra_body"]["response_format"] == _JSON_SCHEMA

    # A remembered rejection is scoped to the endpoint (host:port), not to every local server.
    structured_output.remember_structured_output_rejection(
        "custom", "http://127.0.0.1:1234/v1", {"extra_body": {"response_format": dict(_JSON_SCHEMA)}})
    same = _build_call_kwargs("custom", "m", messages, extra_body={"response_format": dict(_JSON_SCHEMA)},
                              base_url="http://127.0.0.1:1234/v1")
    other = _build_call_kwargs("custom", "m", messages, extra_body={"response_format": dict(_JSON_SCHEMA)},
                               base_url="http://127.0.0.1:11434/v1")
    assert "response_format" not in same.get("extra_body", {})
    assert other["extra_body"]["response_format"] == _JSON_SCHEMA
