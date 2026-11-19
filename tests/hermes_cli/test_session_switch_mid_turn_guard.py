"""/branch, /resume and /sessions <id> refuse to switch sessions while a turn is in flight.

The classic CLI shares ONE agent object across sessions: a switch ends the current session row and
repoints ``agent.session_id`` (``_sync_agent_to_session``), so a still-running turn would flush its
remaining messages onto the switched-to session. ``/handoff`` already refuses mid-turn for the same
reason; these commands must match it. See #112137.

``/new`` is intentionally NOT guarded: ``new_session`` flushes the in-flight turn to the old session
before rotating (flush-then-rotate, #47202), so nothing lands on the wrong row.

The CLI's store is its ``GatewaySessionStore`` on a connection to a real session authority in this
process: the owner writes the rows the assertions read.
"""

import pytest

from cli import HermesCLI
from hermes_cli import cli_commands_mixin
from tests.fixtures.gateway_owner import GatewayOwner
from tests.fixtures.storage_seed import seed_messages, seed_session


class _Agent:
    def __init__(self, sid):
        self.session_id = sid
        self._last_flushed_db_idx = 0

    def _flush_messages_to_session_db(self, history, conversation_history=None):
        pass

    def reset_session_state(self):
        pass


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    owner = GatewayOwner(tmp_path, monkeypatch)
    obj = object.__new__(HermesCLI)
    obj._session_db, obj.model, obj.max_turns = owner.client(), "m", 5
    obj.reasoning_config, obj._pending_title, obj._resumed, obj._pending_resume_sessions = {}, None, False, None
    obj.session_id = obj.parent_id = obj._session_db.open_session(obj)
    obj._gateway_session_open = True
    obj.agent = _Agent(obj.parent_id)
    obj.conversation_history = [{"role": "user", "content": "hi"}]
    for name in ("_restore_session_cwd", "_restore_session_yolo",
                 "_restore_session_model", "_display_resumed_history"):
        setattr(obj, name, lambda *a, **k: None)
    seed_messages(owner.persistence, obj.parent_id, {"role": "user", "content": "hi"})
    seed_session(owner.persistence, "other", "cli", model="m")
    seed_messages(owner.persistence, "other", {"role": "user", "content": "x"})
    obj.printed = []
    monkeypatch.setattr(cli_commands_mixin, "_cp", lambda *lines: obj.printed.extend(lines))
    yield obj
    owner.close()


@pytest.mark.parametrize("command", ["/branch explore", "/resume other", "/sessions other"])
def test_session_switch_refused_mid_turn(cli, command):
    cli._agent_running = True
    handler = {"/branch": cli._handle_branch_command, "/resume": cli._handle_resume_command,
               "/sessions": cli._handle_sessions_command}[command.split()[0]]
    handler(command)
    assert "busy" in " ".join(cli.printed).lower()
    assert cli.session_id == cli.parent_id and cli.agent.session_id == cli.parent_id
    assert cli._session_db.get_session(cli.parent_id)["end_reason"] is None


def test_session_switch_proceeds_when_idle(cli):
    cli._agent_running = False
    cli._handle_branch_command("/branch explore")
    assert cli.session_id != cli.parent_id and cli.agent.session_id == cli.session_id
    assert cli._session_db.get_session(cli.parent_id)["end_reason"] == "branched"
