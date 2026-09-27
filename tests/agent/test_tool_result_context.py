"""Mid-turn plugin context rides fresh, durable top-level tool results."""
from copy import deepcopy

import pytest


def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    from hermes_state import SessionDB
    return AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                   model="test-model", provider="openai-compat", api_key="test",
                   base_url="http://127.0.0.1:1/v1", max_iterations=8,
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def tool_tail(content):
    return [{"role": "user", "content": "Continue the commissioned task", "timestamp": 1.0},
            {"role": "assistant", "timestamp": 2.0, "tool_calls": [{"id": "outer", "type": "function",
             "function": {"name": "execute_code", "arguments": "{}"}}]},
            {"role": "tool", "name": "execute_code", "tool_call_id": "outer", "content": content}]


@pytest.mark.parametrize("content", ["result", [{"type": "text", "text": "result"}]])
def test_context_is_in_next_request_and_replay_without_rewriting_history(tmp_path, monkeypatch, content):
    from hermes_cli.plugins import get_plugin_manager
    from agent.tool_executor import _flush_session_db_after_tool_progress
    from agent.turn_iteration_prep import prepare_iteration
    from tools.tool_result_storage import enforce_turn_budget
    from tools.budget_config import BudgetConfig

    agent = make_agent(tmp_path, monkeypatch)
    received, receipts = [], []
    def context(**kwargs):
        received.append(kwargs)
        return {"context": "CHECKPOINT READY: inspect evidence, then explicitly review",
                "on_delivery": receipts.append}
    monkeypatch.setitem(get_plugin_manager()._hooks, "tool_result_context", [context])
    messages = tool_tail(deepcopy(content))
    prefix = deepcopy(messages[:-1])
    try:
        assert _flush_session_db_after_tool_progress(agent, messages, stage="tool result execute_code")
        assert "CHECKPOINT READY" in str(messages[-1]["content"])
        assert receipts == [True]
        assert received[0]["session_id"] == agent.session_id
        assert received[0]["tool_call_id"] == "outer"
        assert [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages[:-1]] == prefix
        persisted = agent._session_db.get_messages(agent.session_id)
        assert "CHECKPOINT READY" in str(persisted[-1]["content"])
        snapshot = deepcopy(messages)
        assert _flush_session_db_after_tool_progress(agent, messages, stage="repeat flush")
        prepare_iteration(agent, messages=messages, api_call_count=1)
        assert messages == snapshot
        assert len(received) == 1
        # Aggregate output limiting must not hide the checkpoint after acknowledging it.
        enforce_turn_budget(messages[-1:], config=BudgetConfig(turn_budget=1))
        assert messages == snapshot
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("mode", ["interrupted", "disabled", "failed_flush"])
def test_no_delivery_claim_when_cancelled_disabled_or_persistence_fails(tmp_path, monkeypatch, mode):
    from hermes_cli.plugins import get_plugin_manager
    from agent.tool_executor import _flush_session_db_after_tool_progress
    agent = make_agent(tmp_path, monkeypatch)
    calls, receipts = [], []
    def context(**kwargs):
        calls.append(kwargs)
        return {"context": "pending checkpoint", "on_delivery": receipts.append}
    monkeypatch.setitem(get_plugin_manager()._hooks, "tool_result_context", [context])
    if mode == "interrupted":
        agent._interrupt_requested = True
    elif mode == "disabled":
        agent._persist_disabled = True
    else:
        monkeypatch.setattr(agent, "_flush_messages_to_session_db", lambda *a, **k: False)
    messages = tool_tail("ordinary result")
    try:
        result = _flush_session_db_after_tool_progress(agent, messages, stage="tool result")
        if mode == "failed_flush":
            assert not result
            assert receipts == [False]
        else:
            assert not calls
            assert receipts == []
            assert messages[-1]["content"] == "ordinary result"
    finally:
        agent._session_db.close()


@pytest.mark.parametrize("mode", ["no_database", "no_write"])
def test_receipt_requires_target_tool_row_to_be_persisted(tmp_path, monkeypatch, mode):
    from hermes_cli.plugins import get_plugin_manager
    from agent.tool_executor import _flush_session_db_after_tool_progress

    agent = make_agent(tmp_path, monkeypatch)
    database, receipts = agent._session_db, []
    monkeypatch.setitem(get_plugin_manager()._hooks, "tool_result_context", [
        lambda **kwargs: {"context": "still pending", "on_delivery": receipts.append}])
    if mode == "no_database":
        monkeypatch.setattr(agent, "_session_db", None)
    else:
        monkeypatch.setattr(agent, "_flush_messages_to_session_db", lambda *a, **k: True)
    try:
        # Preserve the existing no-database flush result, but never claim durability.
        assert _flush_session_db_after_tool_progress(agent, tool_tail("result"), stage="no write")
        assert receipts == [False]
    finally:
        database.close()


def test_failed_spill_keeps_report_pending_until_complete_content_is_retrievable(tmp_path, monkeypatch):
    from hermes_cli.plugins import get_plugin_manager
    from agent.tool_executor import _flush_session_db_after_tool_progress
    import tools.hook_output_spill as spill

    agent = make_agent(tmp_path, monkeypatch)
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied")
    config = {"enabled": True, "max_chars": 1000, "preview_head": 50,
              "preview_tail": 50, "directory": str(blocked)}
    monkeypatch.setattr(spill, "get_spill_config", lambda: config)
    receipts, other_receipts = [], []
    report = "H" * 600 + "MIDDLE_CHECKPOINT" + "T" * 600
    monkeypatch.setitem(get_plugin_manager()._hooks, "tool_result_context", [
        lambda **kwargs: {"context": report, "on_delivery": receipts.append},
        lambda **kwargs: {"context": "other checkpoint", "on_delivery": other_receipts.append}])
    try:
        messages = tool_tail("result")
        assert _flush_session_db_after_tool_progress(agent, messages, stage="failed spill")
        assert receipts == [], "unavailable report was falsely acknowledged"
        assert other_receipts == [True]
        assert "other checkpoint" in messages[-1]["content"]
        config["directory"] = str(tmp_path / "recovered")
        retry = tool_tail("next result")
        assert _flush_session_db_after_tool_progress(agent, retry, stage="retry spill")
        assert receipts == [True]
        saved = list((tmp_path / "recovered").glob("**/*.txt"))
        assert len(saved) == 1 and saved[0].read_text() == report + "\n"
        assert str(saved[0]) in retry[-1]["content"]
        assert retry[-1]["content"] == agent._session_db.get_messages(agent.session_id)[-1]["content"]
    finally:
        agent._session_db.close()
