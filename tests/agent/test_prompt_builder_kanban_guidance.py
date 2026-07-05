from unittest.mock import MagicMock, patch

from agent.prompt_builder import KANBAN_GUIDANCE, resolve_kanban_guidance
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def test_kanban_guidance_default_unchanged():
    assert (
        resolve_kanban_guidance(profile_name="custom-orchestrator", config={})
        == KANBAN_GUIDANCE
    )


def test_kanban_guidance_append_for_matching_profile():
    custom_text = "Always create analyst then coder tasks."
    config = {
        "kanban": {
            "guidance_override": {
                "custom-orchestrator": {
                    "mode": "append",
                    "text": custom_text,
                },
            },
        },
    }

    guidance = resolve_kanban_guidance(
        profile_name="custom-orchestrator",
        config=config,
    )

    assert KANBAN_GUIDANCE in guidance
    assert custom_text in guidance


def test_kanban_guidance_replace_for_matching_profile():
    custom_text = "Use the exact deterministic decomposition pipeline."
    config = {
        "kanban": {
            "guidance_override": {
                "custom-orchestrator": {
                    "mode": "replace",
                    "text": custom_text,
                },
            },
        },
    }

    guidance = resolve_kanban_guidance(
        profile_name="custom-orchestrator",
        config=config,
    )

    assert guidance == custom_text
    assert KANBAN_GUIDANCE not in guidance


def test_kanban_guidance_non_matching_profile_falls_back():
    config = {
        "kanban": {
            "guidance_override": {
                "custom-orchestrator": {
                    "mode": "replace",
                    "text": "Only custom orchestrators should see this.",
                },
            },
        },
    }

    assert (
        resolve_kanban_guidance(profile_name="coder", config=config)
        == KANBAN_GUIDANCE
    )


def test_kanban_guidance_invalid_mode_defaults_to_append():
    custom_text = "Invalid mode should still append this valid text."
    config = {
        "kanban": {
            "guidance_override": {
                "custom-orchestrator": {
                    "mode": "sideways",
                    "text": custom_text,
                },
            },
        },
    }

    guidance = resolve_kanban_guidance(
        profile_name="custom-orchestrator",
        config=config,
    )

    assert KANBAN_GUIDANCE in guidance
    assert custom_text in guidance


def test_kanban_worker_agent_uses_profile_override():
    custom_text = "Always create analyst then coder tasks."
    config = {
        "kanban": {
            "guidance_override": {
                "custom-orchestrator": {
                    "mode": "append",
                    "text": custom_text,
                },
            },
        },
    }

    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("kanban_show"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
        patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="custom-orchestrator",
        ),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()

    prompt = agent._build_system_prompt()
    assert KANBAN_GUIDANCE in prompt
    assert custom_text in prompt
