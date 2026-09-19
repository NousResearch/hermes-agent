"""Regression: /branch refuses mid-turn (cross-project proposal pi-8937, upstream pi#8937).

An in-flight agent run flushes through the rotating session identity: /branch ends
the parent session row and repoints ``agent.session_id`` via ``_sync_agent_to_session``,
so the turn's remaining messages land on the freshly created branch instead of the
session they belong to. ``/handoff`` already refuses this race mid-turn via
``_handoff_prepare_session``; ``/branch`` must match that behavior.
"""
import sys
from types import SimpleNamespace

from hermes_cli import cli_commands_mixin


def _make_cli(agent_running: bool):
    cli = object.__new__(cli_commands_mixin.CLICommandsMixin)
    cli._agent_running = agent_running
    cli.conversation_history = [{"role": "user", "content": "hello"}]
    cli._session_db = None
    return cli


def test_branch_refuses_mid_turn(monkeypatch):
    printed = []
    monkeypatch.setattr(cli_commands_mixin, "_cp", lambda *lines: printed.extend(lines))
    monkeypatch.setitem(sys.modules, "cli", SimpleNamespace(_sync_process_session_id=lambda *a, **k: None))
    cli = _make_cli(agent_running=True)

    cli._handle_branch_command("/branch explore")

    assert any("busy" in line for line in printed), printed


def test_branch_still_proceeds_when_idle(monkeypatch):
    printed = []
    monkeypatch.setattr(cli_commands_mixin, "_cp", lambda *lines: printed.extend(lines))
    monkeypatch.setitem(sys.modules, "cli", SimpleNamespace(_sync_process_session_id=lambda *a, **k: None))
    cli = _make_cli(agent_running=False)

    cli._handle_branch_command("/branch explore")

    assert printed and "busy" not in " ".join(printed), printed
