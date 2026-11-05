"""Memory-provider status/warning callbacks reach every interactive local surface, never a chat."""

from unittest.mock import patch

import pytest


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def _build_agent(platform):
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=False, session_id="sess-surface", platform=platform,
        )
    return agent, provider


@pytest.mark.parametrize("platform", ["tui", "desktop"])
def test_tui_and_desktop_providers_get_late_bound_status_and_warning_lines(platform):
    agent, provider = _build_agent(platform)

    events = []
    # The TUI/Desktop bind (or rebind) agent.status_callback after construction; the provider's
    # callbacks resolve it at emission time.
    agent.status_callback = lambda kind, text: events.append((kind, text))
    provider.init_kwargs["status_callback"]("saving to memory")
    provider.init_kwargs["warning_callback"]("backend unreachable")

    assert events == [("lifecycle", "saving to memory"), ("warn", "backend unreachable")]


@pytest.mark.parametrize("platform", ["telegram", "slack", "discord"])
def test_messaging_providers_get_no_status_or_warning_callbacks(platform):
    _agent, provider = _build_agent(platform)

    assert "status_callback" not in provider.init_kwargs
    assert "warning_callback" not in provider.init_kwargs
