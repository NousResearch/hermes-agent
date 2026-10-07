"""The inline ``session_search`` executor is the production path for every model tool call: it maps an
explicit argument list onto ``tools.session_search_tool.session_search`` before registry dispatch. A schema
property missing from that map is silently dropped, so the tool shape it selects is dead in production
while direct and registry calls stay green.
"""

import json
from types import SimpleNamespace

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from hermes_state import SessionDB
from tools.session_search_tool import SESSION_SEARCH_SCHEMA


def _ctx():
    return InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")


def test_a_ref_call_through_the_inline_executor_returns_the_original(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="the original words")
        row_id, uid = db._conn.execute("SELECT id, message_uid FROM messages").fetchone()
        # Take the original out of live context, the way compaction archives it.
        db._conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", (row_id,))
        db._conn.commit()
        agent = SimpleNamespace(_get_session_db_for_recall=lambda: db, session_id="s1")
        result = json.loads(INLINE_TOOL_EXECUTORS["session_search"](agent, {"ref": f"m:{uid[:12]}"}, _ctx()))
    finally:
        db.close()
    assert result["success"] is True and result["mode"] == "ref"
    assert [(m["id"], m["content"]) for m in result["messages"]] == [(row_id, "the original words")]


def test_every_schema_property_reaches_the_tool_through_the_inline_executor(monkeypatch):
    received = {}

    def fake_session_search(**kwargs):
        received.update(kwargs)
        return "{}"

    import tools.session_search_tool as tool_module
    monkeypatch.setattr(tool_module, "session_search", fake_session_search)
    properties = SESSION_SEARCH_SCHEMA["parameters"]["properties"]
    sentinels = {name: object() for name in properties}
    agent = SimpleNamespace(_get_session_db_for_recall=lambda: object(), session_id="s1")
    INLINE_TOOL_EXECUTORS["session_search"](agent, dict(sentinels), _ctx())
    dropped = sorted(name for name, sentinel in sentinels.items() if received.get(name) is not sentinel)
    assert dropped == [], f"schema properties the inline executor does not forward: {dropped}"
