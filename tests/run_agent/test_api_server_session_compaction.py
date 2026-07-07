from unittest.mock import MagicMock

from run_agent import AIAgent


def _agent_for_platform(platform: str) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._persist_disabled = False
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent.session_id = f"{platform}-session"
    agent.platform = platform
    return agent


def test_api_server_session_db_persists_only_chat_text():
    agent = _agent_for_platform("api_server")
    messages = [
        {"role": "user", "content": "run the build"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "content": "large build log",
            "tool_name": "terminal",
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "Build passed.", "finish_reason": "stop"},
    ]

    agent._flush_messages_to_session_db(messages, conversation_history=[])

    appends = agent._session_db.append_message.call_args_list
    assert [call.kwargs["role"] for call in appends] == ["user", "assistant"]
    assert [call.kwargs["content"] for call in appends] == [
        "run the build",
        "Build passed.",
    ]
    assert all(call.kwargs["tool_name"] is None for call in appends)
    assert all(call.kwargs["tool_calls"] is None for call in appends)
    assert all(call.kwargs["tool_call_id"] is None for call in appends)


def test_non_api_server_session_db_keeps_tool_metadata():
    agent = _agent_for_platform("cli")
    messages = [
        {"role": "user", "content": "run the build"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "content": "large build log",
            "tool_name": "terminal",
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "Build passed.", "finish_reason": "stop"},
    ]

    agent._flush_messages_to_session_db(messages, conversation_history=[])

    appends = agent._session_db.append_message.call_args_list
    assert [call.kwargs["role"] for call in appends] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assistant_tool_call = appends[1].kwargs
    tool_result = appends[2].kwargs
    assert assistant_tool_call["tool_calls"] == messages[1]["tool_calls"]
    assert tool_result["tool_name"] == "terminal"
    assert tool_result["tool_call_id"] == "call_1"
