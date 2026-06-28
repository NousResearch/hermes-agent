from unittest.mock import patch

from run_agent import AIAgent


def test_invalid_controlled_rebuild_budgets_fall_back(capsys):
    """Invalid compression controlled-rebuild budgets warn and use safe defaults."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "hermes_cli.config.load_config",
            return_value={
                "compression": {
                    "controlled_rebuild_budget": "12K",
                    "controlled_rebuild_checkpoint_budget": 0,
                }
            },
        ),
    ):
        agent = AIAgent(
            api_key="test-k...7890",
            provider="custom",
            model="claude-opus-4-6-thinking",
            base_url="http://proxy.example/v1",
            quiet_mode=False,
            skip_context_files=True,
            skip_memory=True,
        )

    out = capsys.readouterr().out
    assert "Invalid compression.controlled_rebuild_budget" in out
    assert "Invalid compression.controlled_rebuild_checkpoint_budget" in out
    assert getattr(agent, "controlled_context_rebuild_packet_budget") == 12_000
    assert getattr(agent, "controlled_context_rebuild_checkpoint_budget") == 16_000
