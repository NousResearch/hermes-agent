"""Integration regressions for proactive pruning after durable tool progress."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from run_agent import AIAgent


def _tool_call(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call_{index}",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query":"x"}'),
    )


def _tool_response(index: int) -> SimpleNamespace:
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[_tool_call(index)],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=SimpleNamespace(
            prompt_tokens=120_000,
            completion_tokens=10,
            total_tokens=120_010,
        ),
    )


def _stop_response() -> SimpleNamespace:
    message = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=SimpleNamespace(
            prompt_tokens=12_000,
            completion_tokens=10,
            total_tokens=12_010,
        ),
    )


def _tool_defs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_incrementally_persisted_tools_do_not_fence_proactive_pruning(
    tmp_path, caplog
) -> None:
    """A durable tool batch must remain eligible for the same-turn prune.

    Production persisted the assistant call and every tool result before the
    prune ran. The old positional fence then compared SQLite's full active row
    count with only the marker-bearing subset of the in-memory list, rejected
    every prune, and let the provider request grow until full compression.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PROACTIVE_PRUNE_DURABLE_TOOL_TURN"
    db.create_session(session_id, source="telegram")
    db.append_messages_batch(
        session_id,
        [
            {
                "role": "session_meta",
                "content": {"platform": "telegram"},
                "timestamp": 1.0,
            },
            {
                "role": "system",
                "content": "durable control checkpoint",
                "timestamp": 1.5,
                "api_content": "durable control checkpoint on wire",
            },
            {"role": "user", "content": "prior task", "timestamp": 2.0},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "old_call",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"old"}',
                        },
                    }
                ],
                "timestamp": 3.0,
            },
            {
                "role": "tool",
                "content": json.dumps({"data": "o" * 20_000}),
                "tool_call_id": "old_call",
                "tool_name": "web_search",
                "timestamp": 4.0,
            },
            {"role": "assistant", "content": "prior done", "timestamp": 5.0},
        ],
    )
    raw_history = db.get_messages_as_conversation(session_id)
    conversation_history = [
        message
        for message in raw_history
        if message.get("role") not in {"session_meta", "system"}
    ]
    assert len(raw_history) == len(conversation_history) + 2
    control_before = db._conn.execute(
        "SELECT id, role, content, timestamp, api_content, active, compacted "
        "FROM messages WHERE session_id = ? "
        "AND role IN ('session_meta', 'system') ORDER BY id",
        (session_id,),
    ).fetchall()

    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
        patch("model_tools.get_tool_definitions", return_value=_tool_defs()),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            platform="telegram",
            skip_context_files=True,
            skip_memory=True,
            max_iterations=6,
        )

    compressor = agent.context_compressor
    compressor.context_length = 1_000_000
    compressor.threshold_tokens = 500_000
    compressor.proactive_prune_tokens = 1_000
    compressor.proactive_prune_min_result_chars = 1_000
    compressor.proactive_prune_min_reclaim_tokens = 1
    compressor.protect_first_n = 1
    compressor.protect_last_n = 1

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _tool_response(1),
        _tool_response(2),
        _stop_response(),
    ]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False

    with (
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "model_tools.handle_function_call",
            side_effect=lambda *args, **kwargs: json.dumps({"data": "x" * 20_000}),
        ),
        caplog.at_level("WARNING", logger="agent.context_compressor"),
    ):
        result = agent.run_conversation(
            "inspect once",
            conversation_history=conversation_history,
            persist_user_platform_id="17183",
        )

    assert result["completed"] is True
    assert "active transcript changed before deterministic prune commit" not in caplog.text

    live = db.get_messages(session_id)
    live_conversation = [
        message
        for message in live
        if message.get("role") not in {"session_meta", "system"}
    ]
    assert sum(message.get("role") == "session_meta" for message in live) == 1
    assert sum(message.get("role") == "system" for message in live) == 1
    assert sum(message.get("platform_message_id") == "17183" for message in live) == 1
    assert len(live_conversation) == len(result["messages"])
    assert all(
        "durable control checkpoint" not in str(message.get("content"))
        for message in result["messages"]
    )
    control_after = db._conn.execute(
        "SELECT id, role, content, timestamp, api_content, active, compacted "
        "FROM messages WHERE session_id = ? "
        "AND role IN ('session_meta', 'system') ORDER BY id",
        (session_id,),
    ).fetchall()
    assert [tuple(row) for row in control_after] == [
        tuple(row) for row in control_before
    ]
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "old_call"
        and len(str(message.get("content"))) < 1_000
        for message in live_conversation
    )
    old_tool_generations = db._conn.execute(
        "SELECT active, compacted, length(content) AS chars "
        "FROM messages WHERE session_id = ? AND tool_call_id = 'old_call' "
        "ORDER BY id",
        (session_id,),
    ).fetchall()
    assert [tuple(row[:2]) for row in old_tool_generations] == [(0, 1), (1, 0)]
    assert old_tool_generations[0]["chars"] > 20_000
    assert old_tool_generations[1]["chars"] < 1_000
    assert db.get_session(session_id)["message_count"] == len(live)

    # A later full compaction still produces one active generation. The
    # metadata-aware prune must not leave a second active copy for the normal
    # compactor to carry forward.
    active_before_full_compaction = db.get_messages_as_conversation(session_id)
    db.archive_and_compact(session_id, active_before_full_compaction)
    active_after_full_compaction = db.get_messages(session_id)
    assert len(active_after_full_compaction) == len(active_before_full_compaction)
    assert sum(
        message.get("platform_message_id") == "17183"
        for message in active_after_full_compaction
    ) == 1
