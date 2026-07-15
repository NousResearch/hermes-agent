from unittest.mock import MagicMock, patch

from agent.prompt_builder import KANBAN_GUIDANCE, resolve_kanban_guidance
from hermes_cli.config import DEFAULT_CONFIG
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


def _override_config(text: str, *, mode: str = "append") -> dict:
    return {
        "kanban": {
            "guidance_override": {
                "mode": mode,
                "text": text,
            },
        },
    }


def test_kanban_guidance_default_unchanged():
    assert resolve_kanban_guidance(config={}) == KANBAN_GUIDANCE
    assert DEFAULT_CONFIG["kanban"]["guidance_override"] == {
        "mode": "append",
        "text": "",
    }


def test_kanban_guidance_appends_profile_local_text():
    custom_text = "Always create analyst then coder tasks."

    guidance = resolve_kanban_guidance(config=_override_config(custom_text))

    assert KANBAN_GUIDANCE in guidance
    assert custom_text in guidance


def test_kanban_guidance_replaces_for_profile_local_config():
    custom_text = "Use the exact deterministic decomposition pipeline."

    guidance = resolve_kanban_guidance(
        config=_override_config(custom_text, mode="replace")
    )

    assert guidance == custom_text
    assert KANBAN_GUIDANCE not in guidance


def test_kanban_guidance_old_profile_map_shape_falls_back():
    config = {
        "kanban": {
            "guidance_override": {
                "custom-orchestrator": {
                    "mode": "replace",
                    "text": "This map belongs to a different profile.",
                },
            },
        },
    }

    assert resolve_kanban_guidance(config=config) == KANBAN_GUIDANCE


def test_kanban_guidance_invalid_mode_defaults_to_append():
    custom_text = "Invalid mode should still append this valid text."

    guidance = resolve_kanban_guidance(
        config=_override_config(custom_text, mode="sideways")
    )

    assert KANBAN_GUIDANCE in guidance
    assert custom_text in guidance


def test_kanban_guidance_loads_only_active_profile_home(monkeypatch, tmp_path):
    root_home = tmp_path / ".hermes"
    profile_home = root_home / "profiles" / "custom-orchestrator"
    profile_home.mkdir(parents=True)
    root_home.joinpath("config.yaml").write_text(
        "kanban:\n"
        "  guidance_override:\n"
        "    mode: append\n"
        "    text: Root guidance must not leak.\n",
        encoding="utf-8",
    )
    profile_home.joinpath("config.yaml").write_text(
        "kanban:\n"
        "  guidance_override:\n"
        "    mode: append\n"
        "    text: Profile-local verification pipeline.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    guidance = resolve_kanban_guidance()

    assert "Profile-local verification pipeline." in guidance
    assert "Root guidance must not leak." not in guidance


def test_kanban_worker_agent_uses_profile_local_override():
    custom_text = "Always create analyst then coder tasks."
    config = _override_config(custom_text)

    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("kanban_show"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
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
