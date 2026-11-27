"""Anthropic clients put exactly ONE User-Agent header on the wire.

The SDK merges ``default_headers`` over its own defaults as a case-sensitive dict and httpx keeps
both spellings, so a lowercase ``user-agent`` next to the SDK's ``User-Agent`` used to send two
headers on OAuth routes (``Anthropic/Python …`` and ``claude-code/…``). The dict-level checks in
test_anthropic_route_oauth_identity.py could not see that; these tests read the raw request.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("anthropic")

from agent.anthropic_adapter import build_anthropic_client

OAUTH = "sk-ant-oat01-single-ua-test"
_REPLY = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
    "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _wire_user_agents(client) -> list[tuple[bytes, bytes]]:
    seen: list[list[tuple[bytes, bytes]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append([(k, v) for k, v in request.headers.raw if k.lower() == b"user-agent"])
        return httpx.Response(200, json=_REPLY)

    mocked = client.with_options(http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)
    mocked.messages.create(model="claude-test", max_tokens=1, messages=[{"role": "user", "content": "hi"}])
    assert len(seen) == 1
    return seen[0]


def test_static_oauth_token_sends_one_claude_code_user_agent():
    agents = _wire_user_agents(build_anthropic_client(OAUTH, "https://api.anthropic.com"))
    assert len(agents) == 1, agents
    assert agents[0][1].startswith(b"claude-code/")


def test_key_cmd_oauth_token_sends_one_claude_code_user_agent():
    agents = _wire_user_agents(build_anthropic_client(lambda: OAUTH, "https://api.anthropic.com"))
    assert len(agents) == 1, agents
    assert agents[0][1].startswith(b"claude-code/")


def test_lowercase_custom_provider_user_agent_replaces_the_sdk_default():
    with patch("agent.anthropic_adapter._custom_provider_extra_headers", return_value={"user-agent": "RelayUA/1.0"}):
        client = build_anthropic_client("sk-test-key", "https://proxy.example.com/anthropic")
    assert _wire_user_agents(client) == [(b"User-Agent", b"RelayUA/1.0")]
