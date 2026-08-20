"""Regression tests for HermesCLI._reload_mcp session persistence.

`_reload_mcp` appends a "[IMPORTANT: MCP servers have been reloaded ...]" note
and persists immediately so the session log reflects the new tool list even if
the user quits right after reloading (see commit 3ead3401e). The call used to
pass ``self.conversation_history`` as BOTH arguments to ``_persist_session``,
which made ``_flush_messages_to_session_db`` treat every message — including the
brand-new note — as already-durable and skip it (stamping a false
``_DB_PERSISTED_MARKER``). The note therefore never reached state.db and no
later flush could recover it.

These drive the REAL ``_reload_mcp`` against a REAL ``SessionDB`` with the MCP
transport calls mocked, asserting the note is actually persisted and that a
resumed (not-yet-marked) history prefix is not duplicated.
"""
import threading
from pathlib import Path
from unittest.mock import patch

import pytest


def _real_agent(db, session_id, session_messages):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.platform = "cli"
    agent.model = "test-model"
    agent.tools = []
    agent._session_messages = session_messages
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    agent._cached_system_prompt = None
    agent._session_init_model_config = None
    agent._parent_session_id = None
    agent._session_json_enabled = False
    agent.quiet_mode = True
    agent.commit_memory_session = lambda *a, **k: None
    return agent


def _make_cli(agent, conversation_history):
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj._command_running = True
    obj.agent = agent
    obj.enabled_toolsets = {"all"}  # skip the enabled-override merge branch
    obj.conversation_history = conversation_history
    return obj


def _drive_reload(cli):
    """Run _reload_mcp with the MCP transport layer stubbed out."""
    with patch("tools.mcp_tool.shutdown_mcp_servers"), \
         patch("tools.mcp_tool.discover_mcp_tools", return_value=["tool_a", "tool_b"]), \
         patch("tools.mcp_tool.refresh_agent_mcp_tools"), \
         patch("tools.mcp_tool._servers", {"srv": object()}), \
         patch("tools.mcp_tool._lock", threading.RLock()):
        cli._reload_mcp()


def _note_rows(db, session_id):
    rows = db.get_messages_as_conversation(session_id)
    return [m.get("content") or "" for m in rows]


def test_reload_note_persisted_after_a_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess-reload-turn"
    db.create_session(session_id=session_id, source="cli")

    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    agent = _real_agent(db, session_id, history)
    # A prior turn durably persisted the prefix (stamps _DB_PERSISTED_MARKER).
    agent._persist_session(history)

    cli = _make_cli(agent, history)
    _drive_reload(cli)

    contents = _note_rows(db, session_id)
    assert any("MCP servers have been reloaded" in c for c in contents), contents
    # Prefix not duplicated.
    assert contents.count("hi") == 1 and contents.count("hello") == 1, contents


def test_reload_note_persisted_on_resumed_session_without_duplicating(tmp_path, monkeypatch):
    """Resumed session, /reload-mcp before any turn: the loaded prefix is durable
    in the DB but its in-memory dicts carry no marker yet. The note must persist
    and the prefix must not be re-appended."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess-reload-resumed"
    db.create_session(session_id=session_id, source="cli")
    for m in [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]:
        db.append_message(session_id=session_id, role=m["role"], content=m["content"])

    # Fresh, unmarked dicts, as hydrated by /resume.
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    agent = _real_agent(db, session_id, history)
    agent._last_flushed_db_idx = len(history)  # cli resume sets this

    cli = _make_cli(agent, history)
    _drive_reload(cli)

    contents = _note_rows(db, session_id)
    assert any("MCP servers have been reloaded" in c for c in contents), contents
    assert contents.count("hi") == 1 and contents.count("hello") == 1, contents
