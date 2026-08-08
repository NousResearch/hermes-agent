"""Perplexity provider profile tests.

Covers two contracts:

1. The bundled profile registers and resolves (name + aliases) with the
   expected endpoint/auth wiring.
2. The transport honours ``sanitize_tool_schemas`` — profiles carrying the
   flag get the object-properties repair applied to outgoing tool schemas;
   profiles without it are untouched.

Reproduction behind the flag: Perplexity's Chat Completions endpoint returns
HTTP 400 "invalid request" for any object-typed tool parameter that omits
``properties`` (Hermes's ``tool_call.arguments``). Verified live against
api.perplexity.ai 2026-08-07.
"""

from __future__ import annotations

import pytest
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


def _msgs():
    return [{"role": "user", "content": "hello"}]


def _tool_call_tool():
    return {
        "type": "function",
        "function": {
            "name": "tool_call",
            "description": "Invoke a deferred tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    }


class TestPerplexityProfileRegistration:
    def test_resolves_by_name_and_aliases(self):
        profile = get_provider_profile("perplexity")
        assert profile is not None
        assert get_provider_profile("pplx") is profile
        assert get_provider_profile("sonar") is profile

    def test_wiring(self):
        profile = get_provider_profile("perplexity")
        assert profile.base_url == "https://api.perplexity.ai"
        assert "PERPLEXITY_API_KEY" in profile.env_vars
        assert profile.sanitize_tool_schemas is True
        # Behavior contract, not a snapshot: every fallback model must be a
        # sonar-family id (the only family Perplexity serves).
        assert profile.fallback_models
        assert all(m.startswith("sonar") for m in profile.fallback_models)


class TestSanitizeToolSchemasGating:
    def test_flagged_profile_repairs_schemaless_objects(self, transport):
        kwargs = transport.build_kwargs(
            model="sonar-pro",
            messages=_msgs(),
            tools=[_tool_call_tool()],
            provider_profile=get_provider_profile("perplexity"),
        )
        args = kwargs["tools"][0]["function"]["parameters"]["properties"]["arguments"]
        assert args["properties"] == {}

    def test_unflagged_profile_leaves_schemas_alone(self, transport):
        kwargs = transport.build_kwargs(
            model="deepseek-v4-pro",
            messages=_msgs(),
            tools=[_tool_call_tool()],
            provider_profile=get_provider_profile("deepseek"),
        )
        args = kwargs["tools"][0]["function"]["parameters"]["properties"]["arguments"]
        assert "properties" not in args
