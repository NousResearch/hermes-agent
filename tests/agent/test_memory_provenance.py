"""A built-in memory write is traceable without certifying model assertions."""

import json
from types import SimpleNamespace

from agent.agent_runtime_helpers import execute_memory_tool
from agent.background_review import build_memory_write_metadata
from hermes_state import SessionDB
from tools.memory_tool import MemoryStore


def test_memory_write_links_real_source_context_and_preserves_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="s1", source="cli", model="test")
        user_id = db.append_message("s1", "user", "I prefer concise answers.")
        rejected_id = db.append_message(
            "s1", "tool", json.dumps({"status": "blocked", "executed": False}),
            tool_name="terminal", tool_call_id="curl-refused",
        )
        store = MemoryStore()
        store.load_from_disk()
        snapshot = dict(store._system_prompt_snapshot)
        agent = SimpleNamespace(
            session_id="s1", _parent_session_id=None, platform="cli",
            _session_db=db, _memory_store=store, _memory_manager=None,
        )
        agent._build_memory_write_metadata = lambda **kw: build_memory_write_metadata(agent, **kw)
        result = execute_memory_tool(agent, {
            "action": "add", "content": "User prefers concise answers.",
        }, "task-1", "memory-1")
        payload = json.loads(result)
        assert payload["success"] is True
        provenance = payload["provenance"]
        assert provenance["tool_call_id"] == "memory-1"
        assert provenance["evidence_status"] == "unverified"
        assert user_id in provenance["context_message_ids"]
        assert rejected_id not in provenance["context_message_ids"]
        db.append_message("s1", "tool", result, tool_name="memory", tool_call_id="memory-1")
        durable = db.get_messages("s1")[-1]
        assert json.loads(durable["content"])["provenance"] == provenance
        assert store._system_prompt_snapshot == snapshot
        reloaded = MemoryStore()
        reloaded.load_from_disk()
        assert reloaded.memory_entries == ["User prefers concise answers."]
    finally:
        db.close()
