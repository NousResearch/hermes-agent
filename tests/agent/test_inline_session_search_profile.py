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
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_session_search_honours_requested_profile_db(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    profile_home = hermes_home / "profiles" / "llm-wiki"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    current_db = open_persistence(profile_scope(home=hermes_home))
    seed_session(current_db, "default-session", "gateway")
    seed_messages(current_db, "default-session", {"role": "user", "content": "weekly report default"})
    profile_db = open_persistence(profile_scope(home=profile_home))
    seed_session(profile_db, "profile-session", "cli")
    seed_messages(profile_db, "profile-session", {"role": "user", "content": "weekly report llm wiki"})
    profile_db.close()

    agent = SimpleNamespace(_get_session_db_for_recall=lambda: current_db, persistence=current_db, session_id="gw-1")
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
