"""Regression test for #63815: copilot-acp quota exhaustion must activate
the fallback provider chain.

The Copilot ACP server reports monthly-quota exhaustion by streaming the
backend error text as a normal message chunk and ending the turn with
stopReason "refusal".  agent/copilot_acp_client.py converts such turns into
a RuntimeError; this test pins the rest of the chain — classification as
billing and eager fallback activation in the conversation loop — so the
configured fallback provider answers instead of the raw error being
returned to the user.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

_QUOTA_ERROR = RuntimeError(
    "Copilot ACP session/prompt failed: turn ended with stopReason='refusal': "
    "Error: You have exceeded your monthly quota "
    "(Request ID: A540:22A007:3849658:44F52D9:6A541F53)"
)


def _make_copilot_acp_agent(fb_chain):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="copilot-acp",
            base_url="acp://copilot",
            provider="copilot-acp",
            model="auto",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fb_chain,
        )
        return agent


def _mock_response(content: str):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="fallback/model", usage=None)


class TestCopilotACPQuotaFallback:
    def test_quota_refusal_activates_fallback_chain(self):
        fb_chain = [
            {
                "provider": "openrouter",
                "model": "deepseek/deepseek-v4-flash",
                "base_url": "https://openrouter.ai/api/v1",
            }
        ]
        agent = _make_copilot_acp_agent(fb_chain)

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = _QUOTA_ERROR
        agent.client = primary_client

        fallback_client = MagicMock()
        fallback_client.base_url = "https://openrouter.ai/api/v1"
        fallback_client.api_key = "fb-key"
        fallback_client.chat.completions.create.return_value = _mock_response(
            "hello from fallback"
        )

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "deepseek/deepseek-v4-flash"),
        ):
            result = agent.chat("hi there")

        assert agent._fallback_activated is True
        assert agent.provider == "openrouter"
        # The user gets the fallback provider's answer — not the raw
        # Copilot quota error surfaced as if it were the agent's reply.
        assert result == "hello from fallback"
        assert fallback_client.chat.completions.create.call_count == 1
        # Billing errors are non-retryable: the primary must not be
        # hammered again once the quota error is classified.
        assert primary_client.chat.completions.create.call_count == 1

    def test_quota_refusal_without_fallback_surfaces_error(self):
        """Without a chain the turn still terminates with an error result
        (no infinite retry loop) and mentions the quota problem."""
        agent = _make_copilot_acp_agent(None)

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = _QUOTA_ERROR
        agent.client = primary_client

        result = agent.chat("hi there")

        assert agent._fallback_activated is False
        assert primary_client.chat.completions.create.call_count == 1
        assert "quota" in str(result).lower()
