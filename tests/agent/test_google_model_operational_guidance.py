"""Regression tests for the Google model operational guidance (#136497).

GOOGLE_MODEL_OPERATIONAL_GUIDANCE is adapted from OpenCode's gemini.txt, and the
adaptation dropped the upstream negative constraint against chitchat/preambles.
Gemini then read the Conciseness bullet as a mandate to open every action with
an English preamble or step header emitted in message content — which messaging
gateways cannot suppress, since it is not reasoning content. These tests pin
the restored constraint and its Gemini/Gemma-only injection.
"""

from unittest.mock import patch

from agent.prompt_builder import GOOGLE_MODEL_OPERATIONAL_GUIDANCE
from agent.system_prompt import build_system_prompt_parts

from tests.agent.test_system_prompt import _make_agent


def _stable_prompt(agent):
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestNoChitchatConstraint:
    def test_guidance_carries_upstream_no_chitchat_bullet(self):
        assert "- **No chitchat:**" in GOOGLE_MODEL_OPERATIONAL_GUIDANCE
        assert "preambles" in GOOGLE_MODEL_OPERATIONAL_GUIDANCE
        assert "postambles" in GOOGLE_MODEL_OPERATIONAL_GUIDANCE

    def test_injected_for_gemini_model(self):
        agent = _make_agent(
            valid_tool_names=["read_file"],
            _tool_use_enforcement="auto",
            model="google/gemini-3.8-flash",
        )
        stable = _stable_prompt(agent)
        assert "No chitchat" in stable

    def test_not_injected_for_non_google_model(self):
        # grok passes the enforcement gate, so only the Gemini/Gemma check can
        # keep the Google guidance (and its no-chitchat bullet) out.
        agent = _make_agent(
            valid_tool_names=["read_file"],
            _tool_use_enforcement="auto",
            model="x-ai/grok-4",
        )
        stable = _stable_prompt(agent)
        assert "No chitchat" not in stable
        assert "Google model operational directives" not in stable
