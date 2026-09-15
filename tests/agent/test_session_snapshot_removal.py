"""Automatic persistence stays in SQLite; explicit exports remain available."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent_init import _init_session_state
from agent.session_persistence import SessionPersistenceMixin
from hermes_state import SessionDB


@pytest.mark.parametrize("legacy_enabled", [False, True])
def test_persistence_never_snapshots_but_explicit_save_works(tmp_path, monkeypatch, legacy_enabled):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"sessions:\n  write_json_snapshots: {str(legacy_enabled).lower()}\n", encoding="utf-8")
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = SessionPersistenceMixin()
    agent.max_iterations = 1
    _init_session_state(agent, "snapshot-removal", db, None, None, None, False, 1, 1, 1)
    agent.model = "fixture"
    agent.base_url = "http://127.0.0.1:1/v1"
    agent.platform = "cli"
    agent.tools = []
    agent.verbose_logging = True
    db.create_session(agent.session_id, source="cli", model=agent.model)
    agent._session_db_created = True
    messages = [{"role": "user", "content": "keep this conversation"},
                {"role": "assistant", "content": "durable answer"}]
    try:
        agent._persist_session(messages)
        assert len(db.get_messages_as_conversation(agent.session_id)) == 2
        assert not list(agent.logs_dir.glob("session_*.json"))
        from hermes_cli.cli_session_mixin import CLISessionMixin
        cli = SimpleNamespace(_session_db=db, session_id=agent.session_id)
        output = tmp_path / "explicit.json"
        CLISessionMixin.save_conversation(cli, f"/save json {output}")
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert [m["content"] for m in saved["messages"]] == [m["content"] for m in messages]
    finally:
        db.close()
