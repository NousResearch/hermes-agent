"""Regression guard: MiMo thinking history needs reasoning_content padding.

Xiaomi MiMo rejects replayed assistant tool-call messages in thinking mode when
``reasoning_content`` is missing.  Old sessions and cross-provider model
switches can produce such assistant messages, so the provider-facing API copy
must receive a single-space placeholder rather than omitting the field.
"""

from __future__ import annotations

import pytest

from run_agent import AIAgent


def _agent(provider: str = "xiaomi", model: str = "mimo-v2.5-pro", base_url: str = "") -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    return agent


class TestXiaomiMiMoReasoningReplayPadding:
    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("xiaomi", "mimo-v2.5-pro", "https://token-plan-sgp.xiaomimimo.com/anthropic"),
            ("mimo", "mimo-v2.5-pro", ""),
            ("custom", "mimo-v2.5-pro", ""),
            ("custom", "xiaomi-mimo-v2.5-pro", ""),
            ("custom", "xiaomi_mimo_v2.5_pro", ""),
            ("custom", "xiaomi/mimo-v2.5-pro", ""),
            ("custom", "openrouter/xiaomi-mimo-v2.5-pro", ""),
            ("custom", "other-model", "https://token-plan-sgp.xiaomimimo.com/anthropic"),
        ],
    )
    def test_mimo_provider_requires_thinking_reasoning_pad(
        self, provider: str, model: str, base_url: str
    ) -> None:
        agent = _agent(provider=provider, model=model, base_url=base_url)

        assert agent._needs_thinking_reasoning_pad() is True

    def test_mimo_missing_reasoning_content_is_padded_on_api_copy(self) -> None:
        agent = _agent()
        source_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        }
        api_msg = {"role": "assistant", "content": "", "tool_calls": source_msg["tool_calls"]}

        agent._copy_reasoning_content_for_api(source_msg, api_msg)

        assert api_msg["reasoning_content"] == " "

    def test_mimo_empty_reasoning_content_is_upgraded_to_space(self) -> None:
        agent = _agent()
        source_msg = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        }
        api_msg = {"role": "assistant", "content": "", "tool_calls": source_msg["tool_calls"]}

        agent._copy_reasoning_content_for_api(source_msg, api_msg)

        assert api_msg["reasoning_content"] == " "

    def test_non_mimo_provider_does_not_pad_missing_reasoning_content(self) -> None:
        agent = _agent(provider="openrouter", model="anthropic/claude-sonnet-4", base_url="")
        source_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        }
        api_msg = {"role": "assistant", "content": "", "tool_calls": source_msg["tool_calls"]}

        agent._copy_reasoning_content_for_api(source_msg, api_msg)

        assert "reasoning_content" not in api_msg


class TestLookalikeModelsNotDetectedAsMiMo:
    """Adjacent model families must NOT trigger MiMo's reasoning enforcement."""

    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("openrouter", "minimax-text-01", ""),
            ("openrouter", "MiniMax-M2.7", ""),
            ("openrouter", "mistralai/mistral-medium", ""),
            ("openrouter", "microsoft/phi-4", ""),
        ],
    )
    def test_lookalike_not_mimo(
        self, provider: str, model: str, base_url: str
    ) -> None:
        agent = _agent(provider=provider, model=model, base_url=base_url)
        assert agent._needs_mimo_tool_reasoning() is False


class TestDeepNamespacedMiMoModels:
    """Deep-namespaced model names must still be detected as MiMo."""

    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("custom", "vendor/sub/mimo-v3", ""),
            ("custom", "a/b/c/xiaomi-mimo-v2.5-pro", ""),
            ("custom", "openrouter/xiaomi/mimo-v2.5-pro", ""),
        ],
    )
    def test_deep_namespace_detected(
        self, provider: str, model: str, base_url: str
    ) -> None:
        agent = _agent(provider=provider, model=model, base_url=base_url)
        assert agent._needs_mimo_tool_reasoning() is True
