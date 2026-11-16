"""Operator CLI coverage for standalone Kanban admission control."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _args(*argv):
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["kanban", *argv])


def _isolated_home(tmp_path, monkeypatch, *, extra_board=False):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="default")
    if extra_board:
        kb.create_board("alpha")


def test_pause_status_resume_emit_multi_board_readback(tmp_path, monkeypatch, capsys):
    _isolated_home(tmp_path, monkeypatch, extra_board=True)

    assert kc.kanban_command(_args("pause", "--reason", "maintenance", "--json")) == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["state"] == "paused"
    assert [row["board"] for row in paused["boards"]] == ["default", "alpha"]
    assert all(row["state"] == "paused" and row["revision"] == 1 for row in paused["boards"])
    assert all(row["reason"] == "maintenance" for row in paused["boards"])

    assert kc.kanban_command(_args("pause-status", "--json")) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["state"] == "paused"
    assert status["boards"] == paused["boards"]

    assert kc.kanban_command(_args("dispatch", "--json")) == 0
    held = json.loads(capsys.readouterr().out)
    assert held["control_state"] == "paused"
    assert held["spawned"] == []

    assert kc.kanban_command(
        _args("--board", "alpha", "resume", "--reason", "complete", "--json"),
    ) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["state"] == "running"
    assert resumed["boards"][0]["board"] == "alpha"
    assert resumed["boards"][0]["revision"] == 2

    assert kc.kanban_command(_args("pause-status", "--json")) == 1
    mixed = json.loads(capsys.readouterr().out)
    assert mixed["state"] == "partial"
    assert {row["board"]: row["state"] for row in mixed["boards"]} == {
        "default": "paused", "alpha": "running",
    }


def test_claim_reports_explicit_pause_reason(tmp_path, monkeypatch, capsys):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect_closing(board="default") as conn:
        conn.execute(
            "INSERT INTO tasks(id,title,status,created_at,assignee) "
            "VALUES('queued','Queued','ready',1,'worker')",
        )
        conn.commit()
    assert kc.kanban_command(_args("pause")) == 0
    capsys.readouterr()

    assert kc.kanban_command(_args("claim", "queued")) == 1

    output = capsys.readouterr()
    assert "kanban_paused" in output.err
    with kbc.connect_closing(board="default") as conn:
        assert conn.execute("SELECT status FROM tasks WHERE id='queued'").fetchone()[0] == "ready"


def test_multi_board_partial_failure_is_nonzero_and_retry_converges(
    tmp_path, monkeypatch, capsys,
):
    _isolated_home(tmp_path, monkeypatch, extra_board=True)
    original = kbc.connect_closing

    @contextmanager
    def fail_alpha(*args, **kwargs):
        if kwargs.get("board") == "alpha":
            raise OSError("simulated alpha failure")
        with original(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(kbc, "connect_closing", fail_alpha)
    assert kc.kanban_command(_args("pause", "--json")) == 1
    partial = json.loads(capsys.readouterr().out)
    assert partial["state"] == "partial"
    assert partial["boards"][0]["state"] == "paused"
    assert partial["boards"][1]["state"] == "unknown"

    monkeypatch.setattr(kbc, "connect_closing", original)
    assert kc.kanban_command(_args("pause", "--json")) == 0
    converged = json.loads(capsys.readouterr().out)
    assert converged["state"] == "paused"
    assert converged["boards"][0]["revision"] == 1
    assert converged["boards"][1]["revision"] == 1
