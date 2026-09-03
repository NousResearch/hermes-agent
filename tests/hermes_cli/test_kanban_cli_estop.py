"""CLI dispatch must honor the global emergency stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from agent import estop
from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def dispatch_environment(tmp_path, monkeypatch):
    """Use a profile home and its canonical fleet root under one temp tree."""
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "fixture"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    estop._reset_log_state_for_tests()
    return root, profile_home


def _dispatch_args() -> argparse.Namespace:
    return argparse.Namespace(
        kanban_action="dispatch",
        board=None,
        dry_run=False,
        max=None,
        failure_limit=2,
        json=True,
    )


@pytest.mark.parametrize(
    ("sentinel", "reason"),
    (("profile", "profile pause"), ("canonical", "canonical pause")),
)
def test_cli_dispatch_blocks_estop(
    dispatch_environment, monkeypatch, capsys, sentinel, reason
):
    """Either supported sentinel location stops dispatch before dispatch_once."""
    root, profile_home = dispatch_environment
    target = profile_home / "ESTOP" if sentinel == "profile" else root / "ESTOP"
    target.write_text(json.dumps({"reason": reason}), encoding="utf-8")
    dispatch_calls = []
    monkeypatch.setattr(
        kb,
        "dispatch_once",
        lambda *args, **kwargs: dispatch_calls.append((args, kwargs)),
    )

    rc = kb_cli.kanban_command(_dispatch_args())

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["paused"] is True
    assert payload["reason"] == reason
    assert payload["spawned"] == []
    assert dispatch_calls == []


def test_cli_dispatch_runs_when_estop_is_not_engaged(
    dispatch_environment, monkeypatch, capsys
):
    """Without either sentinel, the normal dispatch result is returned."""
    expected = [("fixture-task", "fixture", "/tmp/fixture-workspace")]
    monkeypatch.setattr(
        kb,
        "dispatch_once",
        lambda *args, **kwargs: kb.DispatchResult(spawned=expected),
    )

    rc = kb_cli.kanban_command(_dispatch_args())

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["paused"] is False
    assert payload["spawned"] == [
        {
            "task_id": "fixture-task",
            "assignee": "fixture",
            "workspace": "/tmp/fixture-workspace",
        }
    ]


def test_dispatch_help_documents_estop_behavior(capsys):
    """The subcommand help must expose the pause safety contract."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    kb_cli.build_parser(subparsers)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["kanban", "dispatch", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out.lower()
    assert "estop" in help_text
    assert "no workers are spawned" in help_text
