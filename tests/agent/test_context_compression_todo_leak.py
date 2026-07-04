"""Regression tests for preserving todo state across context compression."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from tools.todo_tool import TodoStore


def _agent_with_fake_compressor():
    agent = cast(Any, object.__new__(AIAgent))
    agent.session_id = "todo-leak-session"
    agent.model = "test/model"
    agent.tools = []
    agent.platform = "cli"
    agent.log_prefix = ""
    agent.quiet_mode = True
    agent.compression_in_place = False
    agent._compression_feasibility_checked = True
    agent._session_db = None
    agent._memory_manager = None
    agent._cached_system_prompt = None
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._todo_store = TodoStore()
    agent._todo_store.write(
        [
            {
                "id": "stale",
                "content": "STALE TODO SHOULD NOT BE A USER MESSAGE",
                "status": "in_progress",
            },
            {"id": "done", "content": "done item", "status": "completed"},
        ]
    )
    agent._emit_status = lambda *args, **kwargs: None
    agent._emit_warning = lambda *args, **kwargs: None
    agent._vprint = lambda *args, **kwargs: None
    agent._invalidate_system_prompt = lambda *args, **kwargs: None
    agent._build_system_prompt = lambda *args, **kwargs: "rebuilt-system-prompt"
    agent.commit_memory_session = lambda *args, **kwargs: None

    compressor = MagicMock()
    compressor.compress.return_value = [
        {
            "role": "assistant",
            "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\nsummary",
        },
        {"role": "user", "content": "latest real user request"},
    ]
    compressor._last_compress_aborted = False
    compressor._last_summary_error = None
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor.compression_count = 1
    agent.context_compressor = compressor
    return agent


def test_compress_context_does_not_append_todo_snapshot_as_user_message():
    from agent.conversation_compression import compress_context

    agent = _agent_with_fake_compressor()
    compressed, _ = compress_context(
        agent,
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
        ],
        "system",
        approx_tokens=100,
    )

    leaked = [
        msg
        for msg in compressed
        if msg.get("role") == "user"
        and isinstance(msg.get("content"), str)
        and msg["content"].startswith(
            "[Your active task list was preserved across context compression]"
        )
    ]
    assert leaked == []
    user_messages = [msg for msg in compressed if msg.get("role") == "user"]
    assert user_messages[-1].get("content") == "latest real user request"


def test_compress_context_preserves_active_todos_as_paired_tool_state():
    from agent.conversation_compression import compress_context

    agent = _agent_with_fake_compressor()
    compressed, _ = compress_context(
        agent,
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
        ],
        "system",
        approx_tokens=100,
    )

    tool_indexes = [
        idx for idx, msg in enumerate(compressed) if msg.get("role") == "tool"
    ]
    assert len(tool_indexes) == 1
    tool_msg = compressed[tool_indexes[0]]
    assert tool_msg.get("name") == "todo"
    assert tool_msg.get("tool_name") == "todo"
    assert tool_msg.get("tool_call_id")
    assert AIAgent._tool_response_matches_todo_call(compressed, tool_indexes[0]) is True
    assert "STALE TODO SHOULD NOT BE A USER MESSAGE" in tool_msg.get("content", "")
    assert "done item" not in tool_msg.get("content", "")


def test_compress_context_inserts_todo_tool_state_after_latest_real_user_turn():
    from agent.conversation_compression import compress_context

    agent = _agent_with_fake_compressor()
    compressed, _ = compress_context(
        agent,
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
        ],
        "system",
        approx_tokens=100,
    )

    latest_user_index = max(
        idx
        for idx, msg in enumerate(compressed)
        if msg.get("role") == "user" and msg.get("content") == "latest real user request"
    )
    tool_index = next(idx for idx, msg in enumerate(compressed) if msg.get("role") == "tool")
    assistant_index = tool_index - 1

    assert assistant_index > latest_user_index
    assert compressed[assistant_index].get("role") == "assistant"
    assert compressed[assistant_index].get("tool_calls")
    assert compressed[tool_index].get("tool_call_id") == compressed[assistant_index]["tool_calls"][0]["id"]


def test_insert_todo_state_messages_preserves_tail_after_latest_user():
    from agent.conversation_compression import _insert_todo_state_messages

    compressed = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "latest request"},
        {"role": "assistant", "content": "tail reply"},
    ]
    todo_state_messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "todo-1"}]},
        {"role": "tool", "tool_call_id": "todo-1", "content": "{}"},
    ]

    result = _insert_todo_state_messages(compressed, todo_state_messages)

    assert result == [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "latest request"},
        *todo_state_messages,
        {"role": "assistant", "content": "tail reply"},
    ]


def test_insert_todo_state_messages_without_user_turn_leaves_history_unchanged():
    from agent.conversation_compression import _insert_todo_state_messages

    compressed = [{"role": "assistant", "content": "summary only"}]
    todo_state_messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "todo-1"}]},
        {"role": "tool", "tool_call_id": "todo-1", "content": "{}"},
    ]

    assert _insert_todo_state_messages(compressed, todo_state_messages) == compressed


def test_compressed_todo_tool_state_hydrates_fresh_agent_store():
    from agent.conversation_compression import compress_context

    agent = _agent_with_fake_compressor()
    compressed, _ = compress_context(
        agent,
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
        ],
        "system",
        approx_tokens=100,
    )

    fresh = cast(Any, object.__new__(AIAgent))
    fresh.session_id = "todo-leak-session"
    fresh.quiet_mode = True
    fresh.log_prefix = ""
    fresh._todo_store = TodoStore()
    with patch("run_agent._set_interrupt"):
        fresh._hydrate_todo_store(compressed)

    assert fresh._todo_store.read() == [
        {
            "id": "stale",
            "content": "STALE TODO SHOULD NOT BE A USER MESSAGE",
            "status": "in_progress",
        }
    ]
