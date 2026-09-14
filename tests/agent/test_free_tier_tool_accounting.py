"""Real executor batches count completions, never wrappers or skipped slots twice."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def agent(tmp_path, monkeypatch):
    from hermes_cli import auth, auth_nous
    from run_agent import AIAgent
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    guest = {"auth_method": "anonymous", "anon_token": "anon_executor_fixture"}
    auth._save_active_provider_state("nous", guest)
    auth_nous._write_shared_nous_state(guest)
    agent = AIAgent(api_key="fixture", provider="nous", model="nous/welcome",
                    base_url="https://welcome-api.nousresearch.com/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent._cached_system_prompt = "Stable prefix"
    agent.compression_enabled = False
    agent.save_trajectories = False
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    return agent


def tc(id, name="fixture_tool", arguments="{}"):
    return SimpleNamespace(id=id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def test_segmented_dispatch_counts_only_completed_calls_once(agent, monkeypatch):
    from agent.free_tier import admit_turn
    from agent.tool_executor import execute_tool_calls_segmented
    from hermes_cli import free_tier_usage as usage
    from tools.registry import registry
    monkeypatch.setattr(registry, "dispatch", lambda *a, **kw: '{"ok":true}')
    calls = [tc(str(i)) for i in range(3)]
    # Invalid arguments are refused without execution.
    invalid = tc("invalid", arguments="{")
    messages = []
    with admit_turn(agent):
        execute_tool_calls_segmented(agent, SimpleNamespace(tool_calls=calls + [invalid]), messages,
                                     "batch", segments=[("parallel", calls[:2]), ("sequential", calls[2:] + [invalid])])
    assert usage.status()["tool_calls_used"] == 3
    assert len([m for m in messages if m["role"] == "tool"]) == 4


def test_real_loop_persists_crossing_turn_and_refusal_writes_nothing(agent, tmp_path, monkeypatch):
    from hermes_cli import free_tier_usage as usage
    from hermes_state import SessionDB
    from tools.registry import registry
    for _ in range(9):
        usage.record_completed_tool(usage.current_identity())
    db = SessionDB(tmp_path / "state.db")
    agent._session_db = db
    agent._session_db_created = False
    import json
    from tools.file_tools import READ_FILE_SCHEMA
    fixture_file = tmp_path / "readable.txt"
    fixture_file.write_text("real-file-receipt", encoding="utf-8")
    tool = tc("completed", name="read_file", arguments=json.dumps({"path": str(fixture_file)}))
    def response(content, calls=None):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls),
                                                       finish_reason="tool_calls" if calls else "stop")], usage=None)
    create = Mock(side_effect=[response("", [tool]), response("Work completed.")])
    monkeypatch.setattr("agent.turn_api_call._should_stream", lambda _agent: False)
    agent._interruptible_api_call = create
    agent.valid_tool_names = {"read_file"}
    agent.tools = [{"type": "function", "function": READ_FILE_SCHEMA}]
    try:
        result = agent.run_conversation("Do the task")
        assert create.call_count == 2
        assert result["final_response"].endswith(usage.LIMIT_NOTICE)
        assert any("real-file-receipt" in str(message.get("content")) for message in result["messages"] if message["role"] == "tool")
        before = copy.deepcopy(db.get_messages_as_conversation(agent.session_id))
        blocked = agent.run_conversation("MUST NOT BE STORED", conversation_history=result["messages"])
        assert blocked["refusal_reason"] == usage.LIMIT_REASON
        assert db.get_messages_as_conversation(agent.session_id) == before
        assert create.call_count == 2
        assert all(usage.LIMIT_NOTICE not in str(m["content"]) for m in before)
    finally:
        db.close()
