from unittest.mock import patch

from run_agent import AIAgent


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
