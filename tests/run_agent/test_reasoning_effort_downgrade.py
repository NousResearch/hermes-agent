"""Tests for reasoning-effort downgrade logic in _github_models_reasoning_extra_body.

Regression test for https://github.com/NousResearch/hermes-agent/issues/10391:
When a provider's catalog reports that it supports "xhigh", the agent must
**not** silently downgrade the user's requested "xhigh" effort to "high".
"""

from unittest.mock import patch

import pytest

from run_agent import AIAgent


# -- helpers -----------------------------------------------------------------


def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def github_agent():
    """Minimal AIAgent targeting GitHub Models endpoint."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://models.github.ai/v1",
            model="gpt-5.4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        return a


# -- tests -------------------------------------------------------------------


class TestXhighReasoningEffortDowngrade:
    """Verify that xhigh is preserved when the provider supports it."""

    def test_xhigh_preserved_when_supported(self, github_agent):
        """If the provider catalog includes xhigh, keep it."""
        github_agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
        catalog = [
            {
                "id": "gpt-5.4",
                "capabilities": {
                    "type": "chat",
                    "supports": {
                        "reasoning_effort": ["low", "medium", "high", "xhigh"],
                    },
                },
            }
        ]
        with patch(
            "hermes_cli.models.github_model_reasoning_efforts",
            return_value=["low", "medium", "high", "xhigh"],
        ):
            result = github_agent._github_models_reasoning_extra_body()
        assert result == {"effort": "xhigh"}

    def test_xhigh_downgraded_to_high_when_not_supported(self, github_agent):
        """If the provider does NOT support xhigh but supports high, downgrade to high."""
        github_agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
        with patch(
            "hermes_cli.models.github_model_reasoning_efforts",
            return_value=["low", "medium", "high"],
        ):
            result = github_agent._github_models_reasoning_extra_body()
        assert result == {"effort": "high"}

    def test_xhigh_downgraded_to_medium_when_no_high(self, github_agent):
        """If the provider supports neither xhigh nor high, fall back to medium."""
        github_agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
        with patch(
            "hermes_cli.models.github_model_reasoning_efforts",
            return_value=["low", "medium"],
        ):
            result = github_agent._github_models_reasoning_extra_body()
        assert result == {"effort": "medium"}

    def test_high_not_affected(self, github_agent):
        """Normal high effort is passed through unchanged."""
        github_agent.reasoning_config = {"enabled": True, "effort": "high"}
        with patch(
            "hermes_cli.models.github_model_reasoning_efforts",
            return_value=["low", "medium", "high"],
        ):
            result = github_agent._github_models_reasoning_extra_body()
        assert result == {"effort": "high"}

    def test_default_medium_when_no_config(self, github_agent):
        """Without explicit config, effort defaults to medium."""
        github_agent.reasoning_config = None
        with patch(
            "hermes_cli.models.github_model_reasoning_efforts",
            return_value=["low", "medium", "high"],
        ):
            result = github_agent._github_models_reasoning_extra_body()
        assert result == {"effort": "medium"}
