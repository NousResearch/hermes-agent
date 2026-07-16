from pathlib import Path
from unittest.mock import patch

import agent.skill_commands as skill_commands_module
from agent.skill_commands import scan_skill_commands
from hermes_cli.moa_config import build_moa_turn_prompt
from run_agent import AIAgent


def _make_triggered_skill(skills_dir: Path) -> None:
    skill_dir = skills_dir / "market-watch"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: market-watch
description: Inspect current market conditions.
triggers: [market conditions]
---

# Market Watch

Inspect the requested market conditions.
""",
        encoding="utf-8",
    )


def test_common_conversation_entrypoint_routes_and_persists_original_message():
    agent = object.__new__(AIAgent)
    setattr(agent, "skills_loading_mode", "routed")
    setattr(agent, "session_id", "session-1")
    setattr(agent, "valid_tool_names", {"skills_list", "terminal"})
    routed = (
        "[expanded skill payload]",
        "What is the market doing?",
        {"name": "market-watch", "trigger": "market"},
    )

    with (
        patch(
            "run_agent.get_toolset_for_tool",
            side_effect=lambda name: {
                "skills_list": "skills",
                "terminal": "terminal",
            }.get(name),
        ),
        patch(
            "agent.skill_commands.expand_triggered_skill_message", return_value=routed
        ) as expand,
        patch(
            "agent.conversation_loop.run_conversation",
            return_value={"final_response": "done"},
        ) as conversation,
    ):
        result = AIAgent.run_conversation(
            agent,
            "What is the market doing?",
            task_id="task-1",
        )

    assert result == {"final_response": "done"}
    expand.assert_called_once_with(
        "What is the market doing?",
        loading_mode="routed",
        task_id="task-1",
        available_tools={"skills_list", "terminal"},
        available_toolsets={"skills", "terminal"},
    )
    args = conversation.call_args.args
    assert args[1] == "[expanded skill payload]"
    assert args[4] == "task-1"
    assert args[6] == "What is the market doing?"


def test_common_entrypoint_preserves_explicit_persistence_override():
    agent = object.__new__(AIAgent)
    setattr(agent, "skills_loading_mode", "routed")
    setattr(agent, "session_id", "session-2")
    routed = (
        "[expanded skill payload]",
        "[voice prefix] original",
        {"name": "voice-helper", "trigger": "original"},
    )

    with (
        patch(
            "agent.skill_commands.expand_triggered_skill_message", return_value=routed
        ) as expand,
        patch(
            "agent.conversation_loop.run_conversation",
            return_value={"final_response": "done"},
        ) as conversation,
    ):
        AIAgent.run_conversation(
            agent,
            "[voice prefix] original",
            persist_user_message="original",
        )

    expand.assert_called_once_with(
        "[voice prefix] original",
        loading_mode="routed",
        task_id="session-2",
        available_tools=set(),
        available_toolsets=set(),
    )
    assert conversation.call_args.args[6] == "original"


def test_encoded_moa_turn_is_decoded_before_routed_trigger_expansion(tmp_path):
    agent = object.__new__(AIAgent)
    setattr(agent, "skills_loading_mode", "routed")
    setattr(agent, "session_id", "session-moa")
    setattr(agent, "valid_tool_names", {"terminal"})
    user_prompt = "Inspect the market conditions before tomorrow."
    encoded_prompt = build_moa_turn_prompt(user_prompt)
    _make_triggered_skill(tmp_path)

    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch("agent.skill_utils.get_external_skills_dirs", return_value=[]),
        patch.object(skill_commands_module, "_skill_commands", {}),
        patch.object(skill_commands_module, "_skill_commands_platform", None),
        patch(
            "agent.conversation_loop.run_conversation",
            return_value={"final_response": "done"},
        ) as conversation,
    ):
        scan_skill_commands()
        result = AIAgent.run_conversation(agent, encoded_prompt, task_id="task-moa")

    assert result == {"final_response": "done"}
    args = conversation.call_args.args
    assert args[1].startswith(
        '[IMPORTANT: The user has invoked the "market-watch" skill'
    )
    assert user_prompt in args[1]
    assert encoded_prompt not in args[1]
    assert args[6] == user_prompt
    assert conversation.call_args.kwargs["moa_config"] is not None
