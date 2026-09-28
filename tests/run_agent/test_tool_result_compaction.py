"""Real dispatch, config, spillover and session persistence at the append boundary."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.registry import registry
from tools.tool_result_storage import extract_persisted_path


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    name = "compaction_records"
    source = tmp_path / "records.json"
    raw = json.dumps({"records": [{"id": i, "title": "Meeting", "active": True}
                                  for i in range(30)]}, indent=2)
    source.write_text(raw, encoding="utf-8")
    schema = {"name": name, "description": "Read local test records",
              "parameters": {"type": "object", "properties": {}}}
    registry.register(name, "compaction_test", schema,
                      lambda args, **kw: source.read_text(encoding="utf-8"))
    definitions = [{"type": "function", "function": schema},
                   {"type": "function", "function": {**schema, "name": "todo_list"}}]
    with (
        patch("model_tools.get_tool_definitions", return_value=definitions),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(model="test/model", api_key="test-key",
                        base_url="http://127.0.0.1:9/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._cached_system_prompt = "Stable prompt"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="compaction", source="cli", model="test/model")
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = "compaction"
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    displayed = []
    agent.tool_complete_callback = lambda *args: displayed.append(args[-1])

    def configure(mode):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "tool_output": {"json_compaction": {"mode": mode, "min_savings_ratio": 0.01}},
        }), encoding="utf-8")

    yield SimpleNamespace(agent=agent, db=db, raw=raw, source=source,
                          name=name, configure=configure, displayed=displayed)
    db.close()
    registry._tools.pop(name, None)


def _run(runtime, executor, name=None, args=None, messages=None, call_id="result-1"):
    call = SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name=name or runtime.name, arguments=json.dumps(args or {})))
    if messages is None:
        messages = [{"role": "system", "content": "Stable prompt"},
                    {"role": "user", "content": "Read records"}]
    messages.append({"role": "assistant", "content": "", "tool_calls": [{
        "id": call.id, "type": "function", "function": {
            "name": call.function.name, "arguments": call.function.arguments},
    }]})
    executor(runtime.agent, SimpleNamespace(tool_calls=[call]), messages, "compaction-task")
    return messages


@pytest.mark.parametrize("executor", [execute_tool_calls_sequential, execute_tool_calls_concurrent])
@pytest.mark.parametrize("mode", ["off", "observe", "compact"])
def test_dispatch_persists_projection_and_preserves_prefix(runtime, executor, mode):
    runtime.configure(mode)
    messages = _run(runtime, executor)
    result = messages[-1]
    assert result["role"] == "tool" and result["tool_call_id"] == "result-1"
    assert json.loads(result["content"]) == json.loads(runtime.raw)
    assert (len(result["content"]) < len(runtime.raw)) == (mode == "compact")
    assert runtime.displayed == [runtime.raw]
    persisted = runtime.db.get_messages_as_conversation("compaction")
    assert persisted[-1]["content"] == result["content"]

    prefix = copy.deepcopy(messages)
    runtime.configure("compact" if mode != "compact" else "off")
    _run(runtime, executor, messages=messages, call_id="result-2")
    assert messages[:len(prefix)] == prefix
    assert runtime.agent._cached_system_prompt == "Stable prompt"
    assert runtime.db.get_messages_as_conversation("compaction")[-3]["content"] == prefix[-1]["content"]


@pytest.mark.parametrize("executor", [execute_tool_calls_sequential, execute_tool_calls_concurrent])
def test_inline_todo_result_uses_same_policy(runtime, executor):
    runtime.configure("compact")
    todos = [{"id": str(i), "content": "Task", "status": "pending"} for i in range(40)]
    messages = _run(runtime, executor, name="todo_list", args={"todos": todos})
    result = messages[-1]["content"]
    assert json.loads(result)["todos"] == todos
    assert len(result) < len(runtime.displayed[-1])


def test_spillover_keeps_original_file_and_recovery_path(runtime):
    runtime.configure("compact")
    raw = json.dumps({"records": [{"id": i, "body": "value"} for i in range(5000)]}, indent=2)
    runtime.source.write_text(raw, encoding="utf-8")
    messages = _run(runtime, execute_tool_calls_sequential)
    path = extract_persisted_path(messages[-1]["content"])
    assert path is not None
    assert Path(path).read_text(encoding="utf-8") == raw
    assert runtime.displayed == [raw]
