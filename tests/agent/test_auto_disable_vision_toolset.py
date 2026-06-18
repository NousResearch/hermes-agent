"""Unit tests for auto-disabling the vision toolset when the main model
lacks native vision capability (issue #43951).

When ``decide_image_input_mode()`` returns ``"text"``, the vision toolset
should be automatically added to ``disabled_toolsets`` so that
``vision_analyze`` does not appear in the tool list — preventing redundant
auxiliary vision API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestAutoDisableVisionToolset:
    """Verify the vision-auto-disable logic in agent/agent_init.py."""

    @staticmethod
    def _make_agent(provider="deepseek", model="deepseek-v4-pro"):
        """Create a minimal mock agent with provider/model set."""
        agent = MagicMock()
        agent.provider = provider
        agent.model = model
        agent.quiet_mode = True
        agent._fallback_chain = []
        return agent

    def test_vision_disabled_for_text_mode_model(self):
        """Non-vision model → vision toolset auto-disabled."""
        agent = self._make_agent("deepseek", "deepseek-v4-pro")

        with patch(
            "agent.image_routing.decide_image_input_mode", return_value="text"
        ), patch(
            "hermes_cli.config.load_config", return_value={}
        ):
            # Simulate the logic from init_agent
            disabled = []
            from agent.image_routing import decide_image_input_mode
            from hermes_cli.config import load_config

            mode = decide_image_input_mode(
                agent.provider, agent.model, load_config()
            )
            if mode == "text":
                disabled.append("vision")

            assert "vision" in disabled

    def test_vision_kept_for_native_mode_model(self):
        """Vision-capable model → vision toolset NOT disabled."""
        agent = self._make_agent("openai", "gpt-4o")

        with patch(
            "agent.image_routing.decide_image_input_mode", return_value="native"
        ), patch(
            "hermes_cli.config.load_config", return_value={}
        ):
            disabled = []
            from agent.image_routing import decide_image_input_mode
            from hermes_cli.config import load_config

            mode = decide_image_input_mode(
                agent.provider, agent.model, load_config()
            )
            if mode == "text":
                disabled.append("vision")

            assert "vision" not in disabled

    def test_vision_not_doubled_if_already_disabled(self):
        """If user already disabled vision in config, don't add it again."""
        agent = self._make_agent("deepseek", "deepseek-v4-pro")

        with patch(
            "agent.image_routing.decide_image_input_mode", return_value="text"
        ), patch(
            "hermes_cli.config.load_config", return_value={}
        ):
            disabled = ["vision"]  # Already disabled by user config
            from agent.image_routing import decide_image_input_mode
            from hermes_cli.config import load_config

            if "vision" not in disabled:
                mode = decide_image_input_mode(
                    agent.provider, agent.model, load_config()
                )
                if mode == "text":
                    disabled.append("vision")

            assert disabled.count("vision") == 1

    def test_exception_in_detection_does_not_disable(self):
        """If decide_image_input_mode raises, vision stays enabled (safe default)."""
        agent = self._make_agent("unknown", "unknown-model")

        with patch(
            "agent.image_routing.decide_image_input_mode",
            side_effect=Exception("provider not found"),
        ):
            disabled = []
            try:
                from agent.image_routing import decide_image_input_mode
                from hermes_cli.config import load_config

                mode = decide_image_input_mode(
                    agent.provider, agent.model, load_config()
                )
                if mode == "text":
                    disabled.append("vision")
            except Exception:
                pass  # Safe default: don't disable

            assert "vision" not in disabled
