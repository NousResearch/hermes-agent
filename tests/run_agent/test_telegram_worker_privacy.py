import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tool_dispatch_helpers import make_tool_result_message
from run_agent import AIAgent, _trajectory_normalize_msg


SENTINEL = "telegram-worker-prompt-sentinel"


def _agent(session_db):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
        )


def _conversation_agent():
    tool_def = {
        "type": "function",
        "function": {
            "name": "telegram_coding_worker",
            "description": "dispatch coding work",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    with (
        patch("run_agent.get_tool_definitions", return_value=[tool_def]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _response(content, finish_reason, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


def _assistant_call():
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "telegram_coding_worker",
                "arguments": json.dumps({"provider": "codex", "prompt": SENTINEL}),
            },
        }],
    }


def test_session_db_redacts_worker_args_without_mutating_live_message():
    session_db = MagicMock()
    agent = _agent(session_db)
    message = _assistant_call()

    agent._flush_messages_to_session_db([message])

    stored = session_db.append_message.call_args.kwargs["tool_calls"]
    assert SENTINEL not in json.dumps(stored)
    assert SENTINEL in message["tool_calls"][0]["function"]["arguments"]


def test_trajectory_conversion_redacts_worker_args_without_mutation():
    agent = _agent(MagicMock())
    message = _assistant_call()
    messages = [
        {"role": "user", "content": "start worker"},
        message,
        {"role": "tool", "content": "completed", "tool_call_id": "call-1"},
    ]

    normalized = _trajectory_normalize_msg(message)
    trajectory = agent._convert_to_trajectory_format(messages, "start worker", True)

    assert SENTINEL not in json.dumps(normalized)
    assert SENTINEL not in json.dumps(trajectory)
    assert SENTINEL in message["tool_calls"][0]["function"]["arguments"]


def test_live_conversation_redacts_worker_args_after_dispatch_before_all_observers():
    agent = _conversation_agent()
    raw_arguments = json.dumps({"provider": "codex", "prompt": SENTINEL})
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(
            name="telegram_coding_worker",
            arguments=raw_arguments,
        ),
    )
    dispatched = []
    next_request_messages = []
    external_memory_messages = []
    background_snapshots = []
    interim_messages = []

    def _execute(assistant_message, messages, effective_task_id, api_call_count=0):
        dispatched.append(assistant_message.tool_calls[0].function.arguments)
        messages.append(
            make_tool_result_message(
                "telegram_coding_worker", "completed", "call-1"
            )
        )

    request_count = 0

    def _respond(*args, **kwargs):
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return _response("", "tool_calls", [tool_call])
        next_request_messages.append(kwargs["messages"])
        return _response("done", "stop")

    agent.client.chat.completions.create.side_effect = _respond
    agent.valid_tool_names.add("memory")
    agent._memory_store = MagicMock()
    agent._memory_nudge_interval = 1
    agent._turns_since_memory = 0

    with (
        patch.object(agent, "_execute_tool_calls", side_effect=_execute),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(
            agent,
            "_emit_interim_assistant_message",
            side_effect=lambda message: interim_messages.append(message),
        ),
        patch.object(
            agent,
            "_sync_external_memory_for_turn",
            side_effect=lambda **kwargs: external_memory_messages.append(kwargs["messages"]),
        ),
        patch.object(
            agent,
            "_spawn_background_review",
            side_effect=lambda **kwargs: background_snapshots.append(
                kwargs["messages_snapshot"]
            ),
        ),
    ):
        result = agent.run_conversation("start worker")

    assert SENTINEL in dispatched[0]
    assert len(interim_messages) == 1
    assert len(next_request_messages) == 1
    assert len(external_memory_messages) == 1
    assert len(background_snapshots) == 1
    assert SENTINEL not in json.dumps(next_request_messages)
    assert SENTINEL not in json.dumps(interim_messages)
    assert SENTINEL not in json.dumps(result["messages"])
    assert SENTINEL not in json.dumps(external_memory_messages)
    assert SENTINEL not in json.dumps(background_snapshots)
    live_call = next(
        message for message in result["messages"] if message.get("tool_calls")
    )["tool_calls"][0]
    assert live_call["id"] == "call-1"
    assert live_call["function"]["name"] == "telegram_coding_worker"
