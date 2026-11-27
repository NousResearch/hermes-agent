"""Real config → SQLite persistence → explicit /save A/B, without provider calls.

Run with a clean HOME/HERMES_HOME and PYTHONPATH pointing at the tree under test.
"""
import json
import os
from pathlib import Path

home = Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True, exist_ok=True)
(home / "config.yaml").write_text("sessions:\n  write_json_snapshots: true\n", encoding="utf-8")
from agent.agent_init import _init_session_state
from agent.session_persistence import SessionPersistenceMixin
from hermes_cli.cli_session_mixin import _write_save
from hermes_cli.session_export import load_save_snapshot, render_session_for_save
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, profile_scope

with open_persistence(profile_scope(home=home), eager=True) as store:
    agent = SessionPersistenceMixin()
    agent.max_iterations = 1
    _init_session_state(agent, "snapshot-probe", store, None, None, None, False, 1, 1, 1, persistence=store)
    agent.model, agent.base_url, agent.platform = "fixture", "http://127.0.0.1:1/v1", "cli"
    agent.tools, agent.verbose_logging = [], True
    store.sessions.create(NewSession(session_id=agent.session_id, source="cli", model=agent.model))
    agent._session_db_created = True
    messages = [{"role": "user", "content": "retained question"},
                {"role": "assistant", "content": "retained answer"}]
    agent._persist_session(messages)
    snapshots = [p.name for p in agent.logs_dir.glob("session_*.json")]
    explicit = home / "explicit.json"
    # The session owner's /save: the stored row rendered and written by the CLI's save writer.
    _write_save(agent.session_id, "json", str(explicit), lambda: render_session_for_save(
        load_save_snapshot(None, agent.session_id, "json", persistence=store), "json"))
    saved = json.loads(explicit.read_text(encoding="utf-8"))
    print(json.dumps({"automatic_snapshots": snapshots,
                      "sqlite_messages": len(store.messages.read_conversation(agent.session_id)),
                      "explicit_saved_messages": [m["content"] for m in saved["messages"]]}))
