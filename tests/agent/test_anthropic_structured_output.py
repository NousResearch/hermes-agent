"""Structured-output translation for Anthropic auxiliary calls."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _capture_anthropic_kwargs(
    extra_body: dict, *, model: str = "claude-sonnet-4-6", async_call: bool = False,
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
    with patch(
        "agent.anthropic_adapter.create_anthropic_message",
        side_effect=_fake_create,
    ), patch("agent.transports.get_transport") as mock_get_transport:
        mock_get_transport.return_value.normalize_response.return_value = normalized
        call = adapter.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=64,
            extra_body=extra_body,
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
