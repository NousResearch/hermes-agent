"""Wire-identity tests for Z.ai's Claude Code-compatible Anthropic route."""
from __future__ import annotations

from unittest.mock import patch
from uuid import UUID

from agent.anthropic_adapter import build_anthropic_client, build_anthropic_kwargs

ZAI_ANTHROPIC_URL = "https://api.z.ai/api/anthropic"


def test_zai_anthropic_client_matches_captured_claude_code_headers():
    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk:
        build_anthropic_client("plan-token", base_url=ZAI_ANTHROPIC_URL)

    kwargs = sdk.Anthropic.call_args.kwargs
    assert kwargs["auth_token"] == "plan-token"
    assert "api_key" not in kwargs
    assert kwargs["base_url"] == ZAI_ANTHROPIC_URL

    headers = kwargs["default_headers"]
    assert headers["User-Agent"].startswith("claude-cli/")
    assert headers["User-Agent"].endswith(" (external, sdk-cli)")
    assert headers["x-app"] == "cli"
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"
    assert headers["x-stainless-lang"] == "js"
    assert headers["x-stainless-runtime"] == "node"
    assert headers["x-stainless-os"] == "Windows"
    assert headers["x-stainless-arch"] == "x64"
    assert headers["x-stainless-package-version"] == "0.94.0"
    assert headers["x-stainless-runtime-version"] == "v26.3.0"
    UUID(headers["x-claude-code-session-id"])
    assert headers["anthropic-beta"] == (
        "claude-code-20250219,interleaved-thinking-2025-05-14,"
        "thinking-token-count-2026-05-13,context-management-2025-06-27,"
        "prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,"
        "advisor-tool-2026-03-01,effort-2025-11-24"
    )


def test_zai_identity_removes_all_product_fingerprints_from_payload():
    kwargs = build_anthropic_kwargs(
        model="glm-5.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Hermes Agent by Nous Research. "
                    "Use hermes-agent and visit hermes-agent.nousresearch.com."
                ),
            },
            {"role": "user", "content": "Ask Hermes by Nous Research"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "A Hermes tool from nousresearch.com",
                    "parameters": {"type": "object"},
                },
            }
        ],
        max_tokens=256,
        reasoning_config=None,
        is_oauth=True,
        base_url=ZAI_ANTHROPIC_URL,
    )

    import json

    wire_payload = json.dumps(kwargs).lower()
    assert "hermes" not in wire_payload
    assert "nous" not in wire_payload
    system_text = "\n".join(block["text"] for block in kwargs["system"])
    assert system_text.startswith("x-anthropic-billing-header: cc_version=")
    assert "You are a Claude agent, built on Anthropic's Claude Agent SDK." in system_text
    assert kwargs["tools"][0]["name"] == "mcp__read_file"
