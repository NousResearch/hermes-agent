"""Regression tests for rejecting-provider reasoning_content stripping.

Cross-provider fallback can replay assistant turns carrying
``reasoning_content`` from a thinking model into strict providers that reject
that field outright. The outgoing API copy must strip the echo before the
request is built, while preserving the existing DeepSeek/Kimi/MiMo add-side
behavior.

Refs #45332.
"""

from __future__ import annotations

from run_agent import AIAgent


def _make_agent(provider: str = "", model: str = "", base_url: str = "") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.verbose_logging = False
    return agent


class TestRejectsReasoningContentEcho:
    def test_custom_groq_provider(self) -> None:
        agent = _make_agent(provider="custom:groq", model="llama-3.1-8b-instant")
        assert agent._rejects_reasoning_content_echo() is True

    def test_cerebras_base_url(self) -> None:
        agent = _make_agent(
            provider="custom",
            model="gpt-oss-120b",
            base_url="https://api.cerebras.ai/v1",
        )
        assert agent._rejects_reasoning_content_echo() is True

    def test_mistral_base_url(self) -> None:
        agent = _make_agent(
            provider="custom",
            model="mistral-small-latest",
            base_url="https://api.mistral.ai/v1",
        )
        assert agent._rejects_reasoning_content_echo() is True

    def test_sambanova_provider(self) -> None:
        agent = _make_agent(
            provider="custom:sambanova",
            model="Meta-Llama-3.3-70B-Instruct",
        )
        assert agent._rejects_reasoning_content_echo() is True

    def test_openrouter_alias_does_not_match_model_name(self) -> None:
        agent = _make_agent(
            provider="openrouter",
            model="mistral/mistral-small-3.2-24b-instruct",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent._rejects_reasoning_content_echo() is False

    def test_cache_invalidates_when_provider_changes(self) -> None:
        agent = _make_agent(
            provider="custom:cerebras",
            model="gpt-oss-120b",
            base_url="https://api.cerebras.ai/v1",
        )
        assert agent._rejects_reasoning_content_echo() is True

        agent.provider = "deepseek"
        agent.model = "deepseek-v4-flash"
        agent.base_url = "https://api.deepseek.com/v1"
        assert agent._rejects_reasoning_content_echo() is False


class TestCopyReasoningContentForApi:
    def test_rejecting_provider_strips_explicit_reasoning_content(self) -> None:
        agent = _make_agent(
            provider="custom:groq",
            model="llama-3.1-8b-instant",
            base_url="https://api.groq.com/openai/v1",
        )
        source = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "chain of thought",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
        }
        api_msg = {"reasoning_content": "stale value"}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg

    def test_rejecting_provider_does_not_promote_reasoning_field(self) -> None:
        agent = _make_agent(
            provider="custom:mistral",
            model="mistral-small-latest",
            base_url="https://api.mistral.ai/v1",
        )
        source = {
            "role": "assistant",
            "content": "",
            "reasoning": "glm thinking text",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert "reasoning_content" not in api_msg

    def test_non_rejecting_provider_preserves_reasoning_content(self) -> None:
        agent = _make_agent(
            provider="openrouter",
            model="z-ai/glm-5.1",
            base_url="https://openrouter.ai/api/v1",
        )
        source = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "keep me",
        }
        api_msg: dict = {}
        agent._copy_reasoning_content_for_api(source, api_msg)
        assert api_msg["reasoning_content"] == "keep me"
