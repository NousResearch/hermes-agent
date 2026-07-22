"""Tests for async_call_llm provider/model logging (issue #48618).

The sync call_llm() logs ``Auxiliary <task>: using <provider> (<model>)``
before making the API call.  async_call_llm() was missing the equivalent
line, making it impossible to verify from agent.log which provider/model
handled an async auxiliary call (e.g. vision pre-analysis).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_chat_response(model: str = "gpt-4.1-nano") -> MagicMock:
    """Build a minimal object matching ``_validate_llm_response``'s input."""
    choice = MagicMock()
    choice.message.content = "ok"
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return resp


def _make_mock_client(model: str = "gpt-4.1-nano", base_url: str = "https://api.example.com/v1"):
    """Return an async mock client whose .chat.completions.create works."""
    client = AsyncMock()
    client.base_url = base_url
    client.chat.completions.create = AsyncMock(return_value=_fake_chat_response(model))
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_call_llm_logs_provider_and_model(caplog):
    """async_call_llm should log provider + model for each task call."""
    mock_client = _make_mock_client(model="gpt-4.1-nano", base_url="https://api.example.com/v1")

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openai", "gpt-4.1-nano", None, None, "openai")),
        patch("agent.auxiliary_client._get_task_extra_body", return_value={}),
        patch("agent.auxiliary_client.resolve_vision_provider_client",
              return_value=("openai", mock_client, "gpt-4.1-nano")),
        patch("agent.auxiliary_client._build_call_kwargs", return_value={"model": "gpt-4.1-nano", "messages": []}),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda r, t: r),
        patch("agent.auxiliary_client._is_anthropic_compat_endpoint", return_value=False),
        caplog.at_level(logging.INFO, logger="agent.auxiliary_client"),
    ):
        from agent.auxiliary_client import async_call_llm
        await async_call_llm(task="vision", messages=[{"role": "user", "content": "hi"}])

    assert any("Auxiliary vision (async): using openai (gpt-4.1-nano)" in r.message
               for r in caplog.records), (
        f"Expected 'Auxiliary vision (async): using openai (gpt-4.1-nano)' in logs, "
        f"got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_async_call_llm_logs_with_base_url(caplog):
    """When base_url is non-OpenRouter, it should appear in the log line."""
    mock_client = _make_mock_client(model="qwen-max", base_url="https://dashscope.aliyuncs.com/v1")

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("custom", "qwen-max", "https://dashscope.aliyuncs.com/v1", None, "openai")),
        patch("agent.auxiliary_client._get_task_extra_body", return_value={}),
        patch("agent.auxiliary_client._get_cached_client", return_value=(mock_client, "qwen-max")),
        patch("agent.auxiliary_client._build_call_kwargs", return_value={"model": "qwen-max", "messages": []}),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda r, t: r),
        patch("agent.auxiliary_client._is_anthropic_compat_endpoint", return_value=False),
        caplog.at_level(logging.INFO, logger="agent.auxiliary_client"),
    ):
        from agent.auxiliary_client import async_call_llm
        await async_call_llm(task="web_extract", messages=[{"role": "user", "content": "hi"}])

    assert any("https://dashscope.aliyuncs.com/v1" in r.message for r in caplog.records), (
        f"Expected base_url in log, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_async_call_llm_no_log_without_task(caplog):
    """When task is None, no Auxiliary log line should be emitted."""
    mock_client = _make_mock_client()

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openai", "gpt-4.1-nano", None, None, "openai")),
        patch("agent.auxiliary_client._get_task_extra_body", return_value={}),
        patch("agent.auxiliary_client._get_cached_client", return_value=(mock_client, "gpt-4.1-nano")),
        patch("agent.auxiliary_client._build_call_kwargs", return_value={"model": "gpt-4.1-nano", "messages": []}),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda r, t: r),
        patch("agent.auxiliary_client._is_anthropic_compat_endpoint", return_value=False),
        caplog.at_level(logging.INFO, logger="agent.auxiliary_client"),
    ):
        from agent.auxiliary_client import async_call_llm
        await async_call_llm(task=None, messages=[{"role": "user", "content": "hi"}])

    assert not any("Auxiliary" in r.message for r in caplog.records), (
        f"Should not log 'Auxiliary' when task=None, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_async_call_llm_hides_openrouter_base_url(caplog):
    """OpenRouter base_url should be suppressed in the log (matches sync path)."""
    mock_client = _make_mock_client(model="claude-sonnet-4", base_url="https://openrouter.ai/api/v1")

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openrouter", "claude-sonnet-4", None, None, "openai")),
        patch("agent.auxiliary_client._get_task_extra_body", return_value={}),
        patch("agent.auxiliary_client.resolve_vision_provider_client",
              return_value=("openrouter", mock_client, "claude-sonnet-4")),
        patch("agent.auxiliary_client._build_call_kwargs", return_value={"model": "claude-sonnet-4", "messages": []}),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda r, t: r),
        patch("agent.auxiliary_client._is_anthropic_compat_endpoint", return_value=False),
        caplog.at_level(logging.INFO, logger="agent.auxiliary_client"),
    ):
        from agent.auxiliary_client import async_call_llm
        await async_call_llm(task="vision", messages=[{"role": "user", "content": "hi"}])

    log_msg = next((r.message for r in caplog.records if "Auxiliary vision (async)" in r.message), "")
    assert "openrouter" not in log_msg.lower() or "https://openrouter.ai" not in log_msg, (
        f"OpenRouter base_url should be suppressed, got: {log_msg}"
    )
