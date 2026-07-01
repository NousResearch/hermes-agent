import json
from unittest.mock import MagicMock


def _bare_agent():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent._memory_store = MagicMock()
    agent._todo_store = MagicMock()
    agent._session_db = None
    agent.session_id = "incog-test-session"
    agent.valid_tool_names = None
    agent.persist_session = False
    agent._build_memory_write_metadata = lambda **kwargs: {"ok": True}
    return agent


def test_incognito_skips_external_memory_sync():
    agent = _bare_agent()

    agent._sync_external_memory_for_turn(
        original_user_message="remember this secret probe",
        final_response="Understood.",
        interrupted=False,
    )

    agent._memory_manager.sync_all.assert_not_called()
    agent._memory_manager.queue_prefetch_all.assert_not_called()


def test_incognito_blocks_built_in_memory_write(monkeypatch):
    agent = _bare_agent()

    called = {"memory_tool": False}

    def _fake_memory_tool(**kwargs):
        called["memory_tool"] = True
        return json.dumps({"success": True})

    monkeypatch.setattr("tools.memory_tool.memory_tool", _fake_memory_tool)

    result = agent._invoke_tool(
        "memory",
        {"action": "add", "target": "memory", "content": "top secret preference"},
        effective_task_id="task-1",
        tool_call_id="tool-1",
    )
    parsed = json.loads(result)

    assert parsed["success"] is False
    assert "Incognito mode is ON" in parsed["error"]
    assert called["memory_tool"] is False
    agent._memory_manager.on_memory_write.assert_not_called()


def test_incognito_blocks_external_retain_but_allows_recall():
    agent = _bare_agent()
    agent._memory_manager.has_tool.side_effect = lambda name: name in {"hindsight_retain", "hindsight_recall"}
    agent._memory_manager.handle_tool_call.return_value = json.dumps({"success": True, "items": []})

    retain_result = agent._invoke_tool(
        "hindsight_retain",
        {"content": "should not persist"},
        effective_task_id="task-2",
        tool_call_id="tool-2",
    )
    retain_parsed = json.loads(retain_result)
    assert retain_parsed["success"] is False
    assert "external memory retention is disabled" in retain_parsed["error"]

    recall_result = agent._invoke_tool(
        "hindsight_recall",
        {"query": "older private continuity"},
        effective_task_id="task-3",
        tool_call_id="tool-3",
    )
    recall_parsed = json.loads(recall_result)
    assert recall_parsed["success"] is True

    assert agent._memory_manager.handle_tool_call.call_count == 1
    agent._memory_manager.handle_tool_call.assert_called_with(
        "hindsight_recall",
        {"query": "older private continuity"},
    )
