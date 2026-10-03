"""CLI interactive text/control source carry against real SessionDB and journals."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

import hermes_constants
from agent.session_persistence import _db_flush_row
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
from hermes_cli.cli_loops_mixin import CLILoopsMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin
from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin
from hermes_cli.maintenance_input import AuthoredInput, authored_source
from hermes_maintenance_source import (
    CLI_CONTROL_CONTRACT, committed_source_for_receipt, current_admitted_source,
    delivery_sources, mint_local_input,
)
from hermes_state import SessionDB
from _maintenance_intake import MaintenanceIntake


MODES = ("new", "resume")
TEXT_PATHS = ("multiline_paste", "skill_or_reference_expansion")
CONTROL_PATHS = (
    "busy_queue", "busy_steer", "busy_interrupt", "slash_queue", "slash_steer",
    "retry_previous_input", "queue_edit",
)


class LocalCLI(CLILoopsMixin, CLITuiRuntimeMixin, CLITuiMixin, CLIChatTurnMixin):
    def __init__(self, home, db, session_id):
        self.home = home
        self.db = db
        self.session_id = session_id
        self._session_db = db
        self._pending_input = queue.Queue()
        self._interrupt_queue = queue.Queue()
        self._pending_resume_sessions = None
        self._agent_running = False
        self.busy_input_mode = "interrupt"
        self._app = SimpleNamespace(invalidate=lambda: None)
        self.conversation_history = []
        self._retry_text = None
        self.agent = SimpleNamespace(_session_messages=[], _session_persist_lock=None)

    def _tui_unwrap_input(self, value):
        return value, False, False

    def _typed_voice_stop(self, value):
        return False

    def handle_bang_shell(self, value):
        return False

    def _print_user_message_preview(self, value):
        return None

    def _turn_summary_begin(self):
        return None

    def _tui_after_turn(self):
        return None

    def _expand_paste_references(self, text):
        return text or ""

    def _record_model_friction(self, *args, **kwargs):
        return None

    def retry_last(self):
        return self._retry_text

    def chat(self, message, **kwargs):
        self.last_source = current_admitted_source()
        assert self.last_source is not None
        self._chat_stage_user_message(self.agent, message)
        row = _db_flush_row(self.agent, self.agent._pending_cli_user_message, True)
        self.db.append_messages_batch(self.session_id, [row])


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "mem0.json").write_text(
        json.dumps({"governed": {"maintenance_intake": True}}), encoding="utf-8")
    db = SessionDB(db_path=home / "state.db")
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    try:
        yield home, db
    finally:
        db.close()


def _cli(cli_env, mode):
    home, db = cli_env
    session_id = f"{mode}-cli-session"
    db.create_session(session_id, "cli", session_key=f"cli:{session_id}")
    return LocalCLI(home, db, session_id)


def _accept(home, source):
    source_id = committed_source_for_receipt(home, source.intended_session_id, source)["source_id"]
    return MaintenanceIntake(
        home, "cli-owner", {("cli", "local-user"): "cli-owner"}, enabled=True).accept(source_id)


def _source_for(cli, source):
    return committed_source_for_receipt(cli.home, cli.session_id, source)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("path", TEXT_PATHS)
def test_cli_text_family_preserves_original_and_expansion(cli_env, mode, path):
    cli = _cli(cli_env, mode)
    raw = "line one\nline two"
    message = raw
    if path == "skill_or_reference_expansion":
        raw = "/demo do the task"
        message = AuthoredInput("expanded skill body\ninstruction: do the task", raw=raw)

    cli._tui_process_one_input(message)

    source = cli.last_source
    proof = _source_for(cli, source)
    assert proof is not None
    assert proof["schema"] == "maintenance-source-v1"
    assert proof["authority"] == "cli-local"
    assert proof["raw_text"] == raw
    assert proof["event_id"] is None
    assert proof["message_row_id"] > 0
    if path == "skill_or_reference_expansion":
        assert "expanded skill body" in proof["extraction_text"]
        assert proof["raw_text"] != proof["extraction_text"]
    task = _accept(cli.home, source)
    assert task["source_id"] == proof["source_id"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("path", CONTROL_PATHS)
def test_cli_control_matrix_mints_or_reuses_one_local_source(cli_env, mode, path):
    cli = _cli(cli_env, mode)
    text = f"{path} fact"
    source = None

    if path == "busy_queue":
        cli.busy_input_mode = "queue"
        cli._agent_running = True
        cli._tui_enter_while_busy(text, [], text)
        source = authored_source(cli._pending_input.get_nowait())
    elif path == "busy_steer":
        cli.busy_input_mode = "steer"
        cli._agent_running = True
        cli.agent.steer = lambda value: True
        cli._tui_enter_while_busy(text, [], text)
        source = cli._maintenance_steer_inputs[0][1]
    elif path == "busy_interrupt":
        cli.busy_input_mode = "interrupt"
        cli._agent_running = True
        cli.agent._supports_active_turn_redirect = True
        cli.agent.redirect = lambda value: True
        cli._tui_enter_while_busy(text, [], text)
        source = cli._maintenance_steer_inputs[0][1]
    elif path == "slash_queue":
        cli._cmd_queue(f"/queue {text}")
        source = authored_source(cli._pending_input.get_nowait())
    elif path == "slash_steer":
        cli._agent_running = True
        cli.agent.steer = lambda value: True
        cli._cmd_steer(f"/steer {text}")
        source = cli._maintenance_steer_inputs[0][1]
    elif path == "queue_edit":
        cli._cmd_queue("/queue first fact")
        original = authored_source(cli._pending_input.get_nowait())
        _accept(cli.home, original)
        cli._pending_input.put(AuthoredInput("first fact", raw="first fact", source=original))
        cli._queue_edit(f"1 {text}")
        source = authored_source(cli._pending_input.get_nowait())
    else:
        original = mint_local_input(text, home=cli.home, session_id=cli.session_id)
        cli.db.append_message(cli.session_id, "user", text, _maintenance_source=original)
        original_task = _accept(cli.home, original)
        cli._retry_text = text
        cli._cmd_retry("/retry")
        source = authored_source(cli._pending_input.get_nowait())

    proof = _source_for(cli, source)
    assert proof is not None
    assert proof["authority"] == "cli-local"
    assert proof["event_id"] is None
    if path == "retry_previous_input":
        assert source.occurrence == original.occurrence
        assert proof["source_id"] == original_task["source_id"]
        assert _accept(cli.home, source)["task_id"] == original_task["task_id"]
        return

    assert proof["schema"] == CLI_CONTROL_CONTRACT
    assert proof["message_row_id"] == 0
    assert proof["source_id"] in delivery_sources(cli.home)
    assert text in proof["raw_text"]
    assert "/queue" not in proof["raw_text"] and "/steer" not in proof["raw_text"]
    if path == "queue_edit":
        assert proof["revision"] == "2"
        assert proof["control_kind"] == "queue_edit"
    else:
        assert proof["revision"] == "1"
    assert _accept(cli.home, source)["source_id"] == proof["source_id"]


def test_cli_concurrent_busy_inputs_keep_distinct_sources(cli_env):
    cli = _cli(cli_env, "new")
    cli.busy_input_mode = "queue"
    cli._agent_running = True
    cli._tui_enter_while_busy("first busy fact", [], "first busy fact")
    cli._tui_enter_while_busy("second busy fact", [], "second busy fact")
    first = authored_source(cli._pending_input.get_nowait())
    second = authored_source(cli._pending_input.get_nowait())
    assert first.occurrence != second.occurrence
    assert _source_for(cli, first)["source_id"] != _source_for(cli, second)["source_id"]


def test_cli_internal_notification_does_not_mint_local_source(cli_env):
    from tools.process_registry_notifications import TimelineNotification
    cli = _cli(cli_env, "new")
    seen = []

    def chat(message, **kwargs):
        seen.append(current_admitted_source())

    cli.chat = chat
    cli._tui_process_one_input(TimelineNotification("internal", "internal", "internal"))
    assert seen == [None]
    assert cli._pending_input.empty()


def test_cli_gateway_clarify_multiple_candidates_stays_unresolved():
    from tools import clarify_gateway
    key = "cli:multi-candidate"
    clarify_gateway.clear_session(key)
    clarify_gateway.register(
        "c1", key, "one?", None,
        control_binding={"task_id": "t1", "item_id": "i1", "proposal_revision": "1"})
    clarify_gateway.register(
        "c2", key, "two?", None,
        control_binding={"task_id": "t2", "item_id": "i2", "proposal_revision": "1"})
    selected, candidates = clarify_gateway.get_pending_control_for_session(key)
    assert selected is None and len(candidates) == 2
    clarify_gateway.clear_session(key)
