from types import SimpleNamespace

import pytest

from agent.conversation_loop import _looks_like_upstream_timeout_empty_stream


SANDBOX_URL = (
    "https://8765-r0rr8sr9.agent-sandbox-fe.baidu-int.com/"
    "api/proxy/69aa17cf52811a2e/llm/v1"
)


def _agent(**overrides):
    values = {
        "model": "gpt-5.5",
        "provider": "custom:sandbox",
        "base_url": SANDBOX_URL,
        "context_compressor": SimpleNamespace(last_prompt_tokens=162_518),
        "_last_stream_diag": {
            "duration": 68.0,
            "content_chunks": 0,
            "tool_call_chunks": 0,
            "reasoning_chunks": 0,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sandbox_long_context_empty_stream_near_timeout_is_classified():
    assert _looks_like_upstream_timeout_empty_stream(_agent()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"_last_stream_diag": {"duration": 3.0}},
        {"context_compressor": SimpleNamespace(last_prompt_tokens=20_000)},
        {"base_url": "https://api.openai.com/v1", "provider": "openai"},
    ],
)
def test_short_low_context_or_non_proxy_empty_stream_remains_plain_empty(overrides):
    assert _looks_like_upstream_timeout_empty_stream(_agent(**overrides)) is False
