"""Unit tests for gateway.response_prefix — the opt-in response prefix
prepended to the first gateway reply."""

from __future__ import annotations

import pytest

from gateway.response_prefix import (
    _model_short,
    build_prefix_line,
    interpolate_prefix_template,
    resolve_prefix_config,
)


# ---------------------------------------------------------------------------
# _model_short
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("github-copilot/claude-opus-4.6", "claude-opus-4.6"),
        ("openai-codex/gpt-5.4", "gpt-5.4"),
        ("anthropic/claude-sonnet-4.6", "claude-sonnet-4.6"),
        ("gpt-5.4", "gpt-5.4"),  # no prefix
        ("", ""),
        (None, ""),
    ],
)
def test_model_short_drops_vendor_prefix(model, expected):
    assert _model_short(model) == expected


# ---------------------------------------------------------------------------
# interpolate_prefix_template
# ---------------------------------------------------------------------------

def test_interpolate_all_variables():
    out = interpolate_prefix_template(
        "[{provider}/{model}]",
        model="github-copilot/claude-opus-4.6",
        provider="github-copilot",
        thinking="high",
    )
    assert out == "[github-copilot/claude-opus-4.6]"


def test_interpolate_short_model():
    out = interpolate_prefix_template(
        "[{model}]",
        model="github-copilot/claude-opus-4.6",
        provider="github-copilot",
    )
    assert out == "[claude-opus-4.6]"


def test_interpolate_model_full():
    out = interpolate_prefix_template(
        "{modelFull}",
        model="github-copilot/claude-opus-4.6",
        provider="github-copilot",
    )
    assert out == "github-copilot/claude-opus-4.6"


def test_interpolate_provider_only():
    out = interpolate_prefix_template(
        "via {provider}",
        model="github-copilot/claude-opus-4.6",
        provider="github-copilot",
    )
    assert out == "via github-copilot"


def test_interpolate_thinking_variants():
    # Case-insensitive variable names
    for var in ("thinking", "thinkingLevel", "thinking_level"):
        out = interpolate_prefix_template(
            f"{{{var}}}",
            thinking="high",
        )
        assert out == "high"


def test_interpolate_unresolved_vars_remain_literal():
    out = interpolate_prefix_template(
        "[{model}] {unknown}",
        model="openai/gpt-5.4",
    )
    assert out == "[gpt-5.4] {unknown}"


def test_interpolate_missing_values_remain_literal():
    out = interpolate_prefix_template(
        "[{provider}/{model}]",
        model=None,
        provider=None,
    )
    # Both unresolved → remain as literal text
    assert out == "[{provider}/{model}]"


def test_interpolate_partial_resolution():
    out = interpolate_prefix_template(
        "[{provider}/{model}]",
        model="openai/gpt-5.4",
        provider=None,
    )
    # model resolves, provider doesn't
    assert out == "[{provider}/gpt-5.4]"


def test_interpolate_with_surrounding_text():
    out = interpolate_prefix_template(
        "🤖 {model} │ ",
        model="github-copilot/claude-opus-4.6",
    )
    assert out == "🤖 claude-opus-4.6 │ "


def test_interpolate_no_variables_returns_as_is():
    out = interpolate_prefix_template(
        "[HERMES] ",
        model="openai/gpt-5.4",
    )
    assert out == "[HERMES] "


def test_interpolate_empty_template():
    assert interpolate_prefix_template("", model="x") == ""


def test_interpolate_case_insensitive():
    # Variable names are case-insensitive
    out = interpolate_prefix_template(
        "[{MODEL}/{PROVIDER}]",
        model="openai/gpt-5.4",
        provider="openai",
    )
    assert out == "[gpt-5.4/openai]"


# ---------------------------------------------------------------------------
# resolve_prefix_config
# ---------------------------------------------------------------------------

def test_resolve_defaults_off_empty_config():
    cfg = resolve_prefix_config({}, "telegram")
    assert cfg == {"enabled": False, "template": ""}


def test_resolve_global_string_template():
    user = {"messages": {"response_prefix": "[{provider}/{model}] "}}
    cfg = resolve_prefix_config(user, "telegram")
    assert cfg["enabled"] is True
    assert cfg["template"] == "[{provider}/{model}] "


def test_resolve_global_empty_string_disabled():
    user = {"messages": {"response_prefix": ""}}
    cfg = resolve_prefix_config(user, "telegram")
    assert cfg["enabled"] is False
    assert cfg["template"] == ""


def test_resolve_global_dict_format():
    user = {"messages": {"response_prefix": {"enabled": True, "template": "[{model}] "}}}
    cfg = resolve_prefix_config(user, "telegram")
    assert cfg["enabled"] is True
    assert cfg["template"] == "[{model}] "


def test_resolve_platform_override_wins():
    user = {
        "messages": {
            "response_prefix": "[{model}] ",
            "platforms": {
                "slack": {"response_prefix": "[SLACK {model}] "},
            },
        },
    }
    # Telegram picks up the global template
    tg = resolve_prefix_config(user, "telegram")
    assert tg["template"] == "[{model}] "
    # Slack overrides
    sl = resolve_prefix_config(user, "slack")
    assert sl["template"] == "[SLACK {model}] "


def test_resolve_platform_can_disable():
    user = {
        "messages": {
            "response_prefix": "[{model}] ",
            "platforms": {
                "slack": {"response_prefix": ""},
            },
        },
    }
    tg = resolve_prefix_config(user, "telegram")
    assert tg["enabled"] is True
    sl = resolve_prefix_config(user, "slack")
    assert sl["enabled"] is False


def test_resolve_platform_dict_format():
    user = {
        "messages": {
            "response_prefix": "[{model}] ",
            "platforms": {
                "discord": {"response_prefix": {"enabled": True, "template": "[DC {model}] "}},
            },
        },
    }
    dc = resolve_prefix_config(user, "discord")
    assert dc["enabled"] is True
    assert dc["template"] == "[DC {model}] "


def test_resolve_ignores_malformed_config():
    # Non-dict/non-string response_prefix shouldn't crash
    user = {"messages": {"response_prefix": 123}}
    cfg = resolve_prefix_config(user, "telegram")
    assert cfg["enabled"] is False


def test_resolve_no_messages_key():
    cfg = resolve_prefix_config({}, "telegram")
    assert cfg == {"enabled": False, "template": ""}


# ---------------------------------------------------------------------------
# build_prefix_line — top-level entry point used by gateway/run.py
# ---------------------------------------------------------------------------

def test_build_prefix_empty_when_disabled():
    out = build_prefix_line(
        user_config={},
        platform_key="telegram",
        model="openai/gpt-5.4",
        provider="openai",
    )
    assert out == ""


def test_build_prefix_returns_rendered_when_enabled():
    out = build_prefix_line(
        user_config={"messages": {"response_prefix": "[{provider}/{model}] "}},
        platform_key="telegram",
        model="openai/gpt-5.4",
        provider="openai",
    )
    assert out == "[openai/gpt-5.4] "


def test_build_prefix_per_platform_suppresses():
    user = {
        "messages": {
            "response_prefix": "[{model}] ",
            "platforms": {"slack": {"response_prefix": ""}},
        },
    }
    out = build_prefix_line(
        user_config=user,
        platform_key="slack",
        model="openai/gpt-5.4",
        provider="openai",
    )
    assert out == ""


def test_build_prefix_provider_from_config():
    """Provider derived from config model.provider when agent result lacks it."""
    config = {
        "messages": {"response_prefix": "[{provider}/{model}]"},
        "model": {"provider": "alibaba", "default": "qwen3.6-plus"},
    }
    out = build_prefix_line(
        user_config=config,
        platform_key=None,
        model="qwen3.6-plus",
        provider="",
    )
    assert out == "[alibaba/qwen3.6-plus]"


def test_build_prefix_no_model_provider_returns_template():
    # Template variables remain literal when no data
    out = build_prefix_line(
        user_config={"messages": {"response_prefix": "[{provider}/{model}] "}},
        platform_key="telegram",
        model=None,
        provider=None,
    )
    # Unresolved vars remain as literal text
    assert out == "[{provider}/{model}] "


def test_build_prefix_thinking_level():
    out = build_prefix_line(
        user_config={"messages": {"response_prefix": "[{model} | think:{thinking}] "}},
        platform_key="telegram",
        model="openai/gpt-5.4",
        thinking="high",
    )
    assert out == "[gpt-5.4 | think:high] "


# ---------------------------------------------------------------------------
# Streaming prefix — GatewayStreamConsumer prepends prefix to first chunk
# ---------------------------------------------------------------------------

import asyncio
from unittest.mock import AsyncMock, MagicMock

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def test_stream_consumer_prepends_prefix_to_first_chunk():
    """Stream consumer should prepend prefix to the first sent chunk."""
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.SUPPORTS_MESSAGE_EDITING = True

    send_result = MagicMock()
    send_result.success = True
    send_result.message_id = "msg123"
    adapter.send = AsyncMock(return_value=send_result)
    adapter.truncate_message = lambda text, limit: [text]

    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(buffer_threshold=1000),  # high threshold so it doesn't flush early
        prefix="[openai/gpt-5.4]",
    )

    async def _run():
        consumer.on_delta("Hello world")
        consumer.finish()
        await consumer.run()

    asyncio.run(_run())

    # The first send should have the prefix prepended
    adapter.send.assert_called()
    first_call_content = adapter.send.call_args_list[0].kwargs.get("content", "")
    assert first_call_content.startswith("[openai/gpt-5.4]"), f"Expected prefix, got: {first_call_content}"
    assert "Hello world" in first_call_content


def test_stream_consumer_prefix_only_on_first_chunk():
    """Prefix should only appear on the first chunk, not subsequent ones."""
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.SUPPORTS_MESSAGE_EDITING = True

    send_result = MagicMock()
    send_result.success = True
    send_result.message_id = "msg123"
    adapter.send = AsyncMock(return_value=send_result)
    adapter.truncate_message = lambda text, limit: [text]

    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(buffer_threshold=5),  # low threshold to force multiple sends
        prefix="[TEST]",
    )

    async def _run():
        consumer.on_delta("First chunk of text ")
        consumer.on_delta("Second chunk of text here")
        consumer.finish()
        await consumer.run()

    asyncio.run(_run())

    # Check all sends — only the first should have the prefix
    calls = adapter.send.call_args_list
    assert len(calls) >= 1
    first_content = calls[0].kwargs.get("content", "")
    assert first_content.startswith("[TEST]"), f"First send should have prefix: {first_content}"
    # Subsequent sends (if any) should NOT have the prefix
    for call in calls[1:]:
        content = call.kwargs.get("content", "")
        assert not content.startswith("[TEST]"), f"Subsequent send should not have prefix: {content}"


def test_stream_consumer_no_prefix_when_empty():
    """When prefix is empty/None, no prefix should be prepended."""
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.SUPPORTS_MESSAGE_EDITING = True

    send_result = MagicMock()
    send_result.success = True
    send_result.message_id = "msg123"
    adapter.send = AsyncMock(return_value=send_result)
    adapter.truncate_message = lambda text, limit: [text]

    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(buffer_threshold=1000),
        prefix="",  # empty prefix
    )

    async def _run():
        consumer.on_delta("Hello world")
        consumer.finish()
        await consumer.run()

    asyncio.run(_run())

    adapter.send.assert_called()
    first_call_content = adapter.send.call_args_list[0].kwargs.get("content", "")
    assert first_call_content == "Hello world", f"No prefix expected, got: {first_call_content}"
