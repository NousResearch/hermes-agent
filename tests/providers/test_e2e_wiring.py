"""E2E tests: verify _build_kwargs_from_profile produces correct output.

These tests call _build_kwargs_from_profile on the transport directly,
without importing run_agent (which would cause xdist worker contamination).
"""

import pytest
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


def _msgs():
    return [{"role": "user", "content": "hi"}]


class TestNvidiaProfileWiring:


    def test_nvidia_model_passed(self, transport):
        profile = get_provider_profile("nvidia")
        kwargs = transport.build_kwargs(
            model="nvidia/test-model",
            messages=_msgs(),
            tools=None,
            provider_profile=profile,
            max_tokens=None,
            max_tokens_param_fn=lambda x: {"max_tokens": x} if x else {},
            timeout=300,
            reasoning_config=None,
            request_overrides=None,
            session_id="test",
            ollama_num_ctx=None,
        )
        assert kwargs["model"] == "nvidia/test-model"


    def test_nvidia_tool_messages_drop_name_fields(self, transport):
        profile = get_provider_profile("nvidia")
        msgs = [
            {"role": "user", "content": "run a command"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "terminal",
                "tool_name": "terminal",
                "tool_call_id": "call_1",
                "content": "ok",
            },
        ]
        kwargs = transport.build_kwargs(
            model="mistralai/mistral-large-3-675b-instruct-2512",
            messages=msgs,
            tools=None,
            provider_profile=profile,
            max_tokens=None,
            max_tokens_param_fn=lambda x: {"max_tokens": x} if x else {},
            timeout=300,
            reasoning_config=None,
            request_overrides=None,
            session_id="test",
            ollama_num_ctx=None,
        )

        assert kwargs["messages"][2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "ok",
        }
        assert msgs[2]["name"] == "terminal"
        assert msgs[2]["tool_name"] == "terminal"


class TestDeepSeekProfileWiring:
    """#110126 layer 1 — the profile's declared output cap must reach the request."""

    def _build(self, transport, **params):
        return transport.build_kwargs(
            model="deepseek-v4-flash",
            messages=_msgs(),
            tools=None,
            provider_profile=get_provider_profile("deepseek"),
            max_tokens_param_fn=lambda x: {"max_tokens": x} if x else {},
            timeout=300,
            reasoning_config=None,
            request_overrides=None,
            session_id="test",
            ollama_num_ctx=None,
            **params,
        )

    def test_profile_default_cap_is_sent_without_explicit_config(self, transport):
        kwargs = self._build(transport, max_tokens=None)
        assert kwargs["max_tokens"] == 65536

    def test_explicit_max_tokens_wins_over_profile_default(self, transport):
        kwargs = self._build(transport, max_tokens=1234)
        assert kwargs["max_tokens"] == 1234

    def test_ephemeral_retry_budget_wins_over_profile_default(self, transport):
        kwargs = self._build(transport, max_tokens=None, ephemeral_max_output_tokens=77)
        assert kwargs["max_tokens"] == 77

