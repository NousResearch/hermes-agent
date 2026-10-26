"""Structured-output translation for Anthropic auxiliary calls."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _capture_anthropic_kwargs(
    extra_body: dict | None, *, model: str = "claude-sonnet-4-6",
    async_call: bool = False, top_level_kwargs: dict | None = None,
) -> dict:
    from agent.auxiliary_client import (
        _AnthropicCompletionsAdapter,
        _AsyncAnthropicCompletionsAdapter,
    )

    captured = {}
    sync_adapter = _AnthropicCompletionsAdapter(
        MagicMock(name="anthropic_client"), model, is_oauth=False,
    )
    adapter = (
        _AsyncAnthropicCompletionsAdapter(sync_adapter)
        if async_call
        else sync_adapter
    )

    def _fake_create(_client, api_kwargs, **_kwargs):
        captured.update(api_kwargs)
        return SimpleNamespace()

    normalized = SimpleNamespace(
        content="ok",
        tool_calls=None,
        reasoning=None,
        finish_reason="stop",
    )
    call_kwargs = dict(top_level_kwargs or {})
    if extra_body is not None:
        call_kwargs["extra_body"] = extra_body
    with patch(
        "agent.anthropic_adapter.create_anthropic_message",
        side_effect=_fake_create,
    ), patch("agent.transports.get_transport") as mock_get_transport:
        mock_get_transport.return_value.normalize_response.return_value = normalized
        call = adapter.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
            **call_kwargs,
        )
        if async_call:
            asyncio.run(call)

    return captured


def _assert_no_raw_response_format(api_kwargs: dict) -> None:
    assert "response_format" not in api_kwargs
    assert "response_format" not in api_kwargs.get("extra_body", {})


def test_json_schema_response_format_uses_native_output_config():
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    api_kwargs = _capture_anthropic_kwargs({
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "thread_title",
                "schema": schema,
                "strict": False,
            },
        },
    })

    assert api_kwargs["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }
    _assert_no_raw_response_format(api_kwargs)


def test_json_object_response_format_uses_permissive_object_schema():
    api_kwargs = _capture_anthropic_kwargs({
        "response_format": {"type": "json_object"},
    })

    assert api_kwargs["output_config"]["format"] == {
        "type": "json_schema",
        "schema": {"type": "object"},
    }
    _assert_no_raw_response_format(api_kwargs)


def test_response_format_merges_with_adaptive_thinking_effort():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    api_kwargs = _capture_anthropic_kwargs({
        "reasoning": {"enabled": True, "effort": "high"},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": schema},
        },
    })

    assert api_kwargs["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": schema},
    }
    assert api_kwargs["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    _assert_no_raw_response_format(api_kwargs)


def test_unrelated_extra_body_keys_still_pass_through():
    api_kwargs = _capture_anthropic_kwargs({
        "response_format": {"type": "json_object"},
        "metadata": {"user_id": "thread-autotitle"},
        "vendor_option": True,
    })

    assert api_kwargs["extra_body"] == {
        "metadata": {"user_id": "thread-autotitle"},
        "vendor_option": True,
    }
    _assert_no_raw_response_format(api_kwargs)


def test_async_anthropic_adapter_uses_the_same_translation():
    schema = {"type": "object"}
    api_kwargs = _capture_anthropic_kwargs(
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": schema},
            },
        },
        async_call=True,
    )

    assert api_kwargs["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }
    _assert_no_raw_response_format(api_kwargs)


def test_top_level_response_format_kwarg_is_translated_not_dropped():
    """#85626 review, point 2: the top-level kwarg shape must also translate.

    ``client.chat.completions.create(..., response_format=...)`` is the
    OpenAI SDK's documented call shape. The adapter builds the Messages body
    from a fixed allow-list of kwargs, so before this an unrecognized
    top-level kwarg was dropped on the floor: the request succeeded, but the
    schema contract silently became prompt compliance. Pin-test pattern from
    PR #85626 (Matt McClean), adapted from strip to translate semantics.
    """
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    api_kwargs = _capture_anthropic_kwargs(
        None,
        top_level_kwargs={
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "session_title", "schema": schema},
            },
        },
    )

    assert api_kwargs["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }
    _assert_no_raw_response_format(api_kwargs)


def test_extra_body_response_format_wins_over_top_level_kwarg():
    """When both shapes are present, extra_body wins.

    Every in-tree caller uses the extra_body shape. The top-level kwarg is
    the compatibility path, so it must not override an explicit extra_body
    value when a caller somehow sends both.
    """
    eb_schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    top_schema = {"type": "object", "properties": {"b": {"type": "string"}}}
    api_kwargs = _capture_anthropic_kwargs(
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": eb_schema},
            },
        },
        top_level_kwargs={
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": top_schema},
            },
        },
    )

    assert api_kwargs["output_config"]["format"] == {
        "type": "json_schema",
        "schema": eb_schema,
    }
    _assert_no_raw_response_format(api_kwargs)
