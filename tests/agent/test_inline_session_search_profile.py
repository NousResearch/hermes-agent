"""The inline ``session_search`` executor forwards ``profile`` so a routed gateway agent can read a
named profile's store instead of silently searching the injected (default) DB (#82903).

The gateway hard-injects ``db=agent._get_session_db_for_recall()``; without ``profile`` reaching
``tools.session_search_tool.session_search`` the tool's ``_resolve_profile_db`` never runs and
``profile="llm-wiki"`` returns the same rows as no profile at all.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from hermes_state import SessionDB
import tools.session_search_tool  # noqa: F401  (register built-in before override)
from tools.registry import registry


def test_session_search_honours_requested_profile_db(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    profile_home = hermes_home / "profiles" / "llm-wiki"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    current_db = SessionDB(hermes_home / "state.db")
    current_db.create_session("default-session", source="gateway")
    current_db.append_message("default-session", role="user", content="weekly report default")
    current_db._conn.commit()
    profile_db = SessionDB(profile_home / "state.db")
    profile_db.create_session("profile-session", source="cli")
    profile_db.append_message("profile-session", role="user", content="weekly report llm wiki")
    profile_db._conn.commit()
    profile_db.close()

    agent = SimpleNamespace(_get_session_db_for_recall=lambda: current_db, session_id="gw-1")
    ctx = InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")
    try:
        routed = json.loads(INLINE_TOOL_EXECUTORS["session_search"](
            agent, {"query": "weekly report", "profile": "llm-wiki"}, ctx))
        missing = json.loads(INLINE_TOOL_EXECUTORS["session_search"](
            agent, {"query": "weekly report", "profile": "missing-profile"}, ctx))
    finally:
        current_db.close()

    assert [r["session_id"] for r in routed["results"]] == ["profile-session"]
    assert missing["success"] is False and "default-session" not in json.dumps(missing)


def test_inline_session_search_uses_scoped_registry_override_with_live_db(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(home / "state.db")
    db.create_session("history", source="cli")
    db.append_message("history", role="user", content="orchid retrospective")
    assert db._conn is not None
    db._conn.commit()
    agent = SimpleNamespace(_get_session_db_for_recall=lambda: db, session_id="current")
    ctx = InlineToolContext(effective_task_id="task")
    builtin = registry.get_entry("session_search")
    assert builtin is not None
    builtin_handler = builtin.handler
    observed = {}

    def enriched_handler(args, **kwargs):
        observed.update(kwargs)
        result = json.loads(builtin_handler(args, **kwargs))
        result["override_ran"] = True
        return json.dumps(result)

    registry.register(
        name="session_search", toolset="history-plugin", schema=builtin.schema,
        handler=enriched_handler, override=True, scope=str(home),
    )
    try:
        assert registry.get_entry("session_search").handler is enriched_handler
        result = json.loads(INLINE_TOOL_EXECUTORS["session_search"](
            agent, {"query": "orchid"}, ctx))
        assert result["override_ran"] is True
        assert [hit["session_id"] for hit in result["results"]] == ["history"]
        assert observed == {"db": db, "current_session_id": "current"}
    finally:
        registry.deregister("session_search", scope=str(home))
        db.close()
    assert registry.get_entry("session_search") is builtin
