"""Anthropic API key families must not be routed through OAuth bearer auth."""

import httpx
import pytest

from agent.anthropic_adapter import build_anthropic_client
from agent.anthropic_credentials import _is_oauth_token


@pytest.mark.parametrize("family", ["usr", "admin", "workspace", "future", "api03"])
def test_console_key_families_are_not_oauth(family):
    key = "sk-" + "ant-" + family + "-synthetic-test-only"
    assert not _is_oauth_token(key)


@pytest.mark.parametrize(
    ("key", "oauth"),
    [
        ("sk-" + "ant-usr-synthetic-test-only", False),
        ("sk-" + "ant-oat01-synthetic-test-only", True),
        ("eyJsynthetic-test-only", True),
        ("cc-synthetic-test-only", True),
    ],
)
def test_native_messages_sends_exactly_the_matching_auth_header(key, oauth):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    with build_anthropic_client(key, "https://api.anthropic.com") as base:
        with base.with_options(http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            result = client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=16,
                messages=[{"role": "user", "content": "Reply only OK."}],
            )
    assert result.content[0].text == "OK"
    headers = requests[0].headers
    if oauth:
        assert headers["authorization"] == "Bearer " + key
        assert "x-api-key" not in headers
    else:
        assert headers["x-api-key"] == key
        assert "authorization" not in headers
        assert "oauth-2025-04-20" not in headers.get("anthropic-beta", "")
