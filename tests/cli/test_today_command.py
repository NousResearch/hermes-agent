"""Tests for CLI /today command behavior."""
from datetime import datetime
from unittest.mock import MagicMock

from cli import HermesCLI
from hermes_cli.commands import resolve_command


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "session-123"
    cli_obj._pending_input = MagicMock()
    cli_obj._status_bar_visible = True
    cli_obj.model = "openai/gpt-5.4"
    cli_obj.provider = "openai"
    cli_obj.session_start = datetime(2026, 4, 9, 19, 24)
    cli_obj._agent_running = False
    cli_obj._session_db = MagicMock()
    cli_obj._session_db.get_session.return_value = None
    cli_obj._pending_model_switch_note = None
    return cli_obj


def _write_todo_state(base_dir):
    todo_dir = base_dir / "hermes-daily-state"
    todo_dir.mkdir(parents=True, exist_ok=True)
    todo_path = todo_dir / "todo-state.json"
    todo_path.write_text(
        """{
  \"version\": 1,
  \"updated_at\": \"2026-04-19T20:08:00+08:00\",
  \"pending\": [
    {\"id\": \"p1\", \"title\": \"Continue benchmark from checkpoint 482\", \"status\": \"pending\"},
    {\"id\": \"p2\", \"title\": \"Check roleplay harness cleanup\", \"status\": \"pending\"}
  ],
  \"active\": [
    {\"id\": \"a1\", \"title\": \"Stabilize today's todo bridge\", \"status\": \"active\"}
  ],
  \"resolved_recent\": [
    {\"id\": \"r1\", \"title\": \"Fix None.strip crash\", \"status\": \"resolved\"}
  ],
  \"archive\": []
}
""",
        encoding="utf-8",
    )
    return todo_path


def test_today_command_is_available_in_cli_registry():
    cmd = resolve_command("today")
    assert cmd is not None
    assert cmd.gateway_only is False


def test_process_command_today_prints_snapshot_and_stashes_next_turn_note(monkeypatch, tmp_path):
    cli_obj = _make_cli()
    hermes_home = tmp_path / ".hermes"
    _write_todo_state(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert cli_obj.process_command("/today") is True

    printed = "\n".join(str(call.args[0]) for call in cli_obj.console.print.call_args_list)
    assert "Today Todo" in printed
    assert "Active (1)" in printed
    assert "Stabilize today's todo bridge" in printed
    assert "Pending (2)" in printed
    assert "Continue benchmark from checkpoint 482" in printed
    assert cli_obj._pending_model_switch_note is not None
    assert "today todo snapshot" in cli_obj._pending_model_switch_note.lower()
    assert "Stabilize today's todo bridge" in cli_obj._pending_model_switch_note
    assert "Continue benchmark from checkpoint 482" in cli_obj._pending_model_switch_note
