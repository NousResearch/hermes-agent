"""Real config → SQLite persistence → explicit /save A/B, without provider calls.

Run with a clean HOME/HERMES_HOME and PYTHONPATH pointing at the tree under test.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

home = Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True, exist_ok=True)
(home / "config.yaml").write_text("sessions:\n  write_json_snapshots: true\n", encoding="utf-8")
from agent.agent_init import _init_session_state
from agent.session_persistence import SessionPersistenceMixin
from hermes_cli.cli_session_mixin import CLISessionMixin
from hermes_state import SessionDB

with SessionDB(db_path=home / "state.db") as db:
    agent = SessionPersistenceMixin()
    agent.max_iterations = 1
    _init_session_state(agent, "snapshot-probe", db, None, None, None, False, 1, 1, 1)
    agent.model, agent.base_url, agent.platform = "fixture", "http://127.0.0.1:1/v1", "cli"
    agent.tools, agent.verbose_logging = [], True
    db.create_session(agent.session_id, source="cli", model=agent.model)
    agent._session_db_created = True
    messages = [{"role": "user", "content": "retained question"},
                {"role": "assistant", "content": "retained answer"}]
    agent._persist_session(messages)
    snapshots = [p.name for p in agent.logs_dir.glob("session_*.json")]
    cli = SimpleNamespace(_session_db=db, session_id=agent.session_id)
    explicit = home / "explicit.json"
    CLISessionMixin.save_conversation(cli, f"/save json {explicit}")
    saved = json.loads(explicit.read_text(encoding="utf-8"))
    print(json.dumps({"automatic_snapshots": snapshots,
                      "sqlite_messages": len(db.get_messages_as_conversation(agent.session_id)),
                      "explicit_saved_messages": [m["content"] for m in saved["messages"]]}))
