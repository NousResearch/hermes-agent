"""Tests for hermes_cli/delegation_visualizer.py.

The delegation_visualizer module is a thin wrapper around the standalone
``hermes_delegation`` package (M1+M2). These tests mirror the conventions in
tests/agent/test_curator.py: a per-test fixture sets up an isolated ledger
base dir, the daemon socket is monkeypatched so no real Unix socket is needed,
and the three subcommand handlers are exercised through ``cli_main``.

No real verifier daemon is ever started — ``_daemon_is_listening`` is
monkeypatched. Ledger fixtures are built with the real ``Ledger`` + event
classes so ``compute_snapshot`` / ``render_report`` run against authentic data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

# Skip the whole module cleanly if the M1+M2 package isn't installed in this
# environment — the wrapper itself degrades gracefully, but these tests assert
# real behaviour and need the package.
hd = pytest.importorskip("hermes_delegation")

from hermes_cli import delegation_visualizer as dv  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger_dir(tmp_path):
    """A base dir containing one valid ledger for task ``task-id``."""
    from hermes_delegation.ledger import Ledger

    base = tmp_path / "ledgers"
    base.mkdir(parents=True)
    led = Ledger(task_id="task-id", base_dir=base)
    led.append(
        hd.TaskStartedEvent(task_id="task-id", user_request="do x", agents_planned=1)
    )
    led.append(
        hd.TaskCompletedEvent(task_id="task-id", status="success", elapsed_ms=100)
    )
    led.close()
    return base


@pytest.fixture
def daemon_up(monkeypatch):
    """Pretend the verifier daemon is listening."""
    import hermes_delegation.cli as hdcli

    monkeypatch.setattr(hdcli, "_daemon_is_listening", lambda *a, **k: True)


@pytest.fixture
def daemon_down(monkeypatch):
    """Pretend the verifier daemon is NOT listening."""
    import hermes_delegation.cli as hdcli

    monkeypatch.setattr(hdcli, "_daemon_is_listening", lambda *a, **k: False)


# ---------------------------------------------------------------------------
# register_cli
# ---------------------------------------------------------------------------

def test_register_cli_attaches_subcommands():
    parser = argparse.ArgumentParser(prog="hermes delegation")
    dv.register_cli(parser)
    # Each subcommand should parse without error and set a `func`.
    for sub in ("status", "verify", "report"):
        if sub == "status":
            args = parser.parse_args([sub])
        else:
            args = parser.parse_args([sub, "some-task"])
        assert getattr(args, "func", None) is not None, f"{sub} has no func"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_cli_main_status_runs_when_daemon_up(daemon_up, capsys):
    rc = dv.cli_main(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "running" in out


def test_cli_main_status_returns_1_when_daemon_down(daemon_down, capsys):
    rc = dv.cli_main(["status"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not running" in out


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def test_cli_main_verify_delegates_to_compute_snapshot(ledger_dir, capsys):
    rc = dv.cli_main(["verify", "task-id", "--base-dir", str(ledger_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["task_id"] == "task-id"
    assert payload["status"] == "success"


def test_cli_main_verify_missing_task_returns_1(tmp_path, capsys):
    rc = dv.cli_main(["verify", "nope", "--base-dir", str(tmp_path)])
    capsys.readouterr()
    assert rc == 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def test_cli_main_report_writes_markdown(ledger_dir, tmp_path, capsys):
    out_path = tmp_path / "r.md"
    rc = dv.cli_main(
        ["report", "task-id", "--base-dir", str(ledger_dir), "--output", str(out_path)]
    )
    capsys.readouterr()
    assert rc == 0
    assert out_path.exists()
    assert out_path.read_text().strip(), "report file is empty"


def test_cli_main_report_to_stdout(ledger_dir, capsys):
    rc = dv.cli_main(["report", "task-id", "--base-dir", str(ledger_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip(), "expected non-empty markdown on stdout"
    assert "task-id" in out


# ---------------------------------------------------------------------------
# no args / passthrough
# ---------------------------------------------------------------------------

def test_cli_main_no_args_prints_help(capsys):
    rc = dv.cli_main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage" in out.lower() or "status" in out


def test_argv_passthrough(daemon_up, capsys):
    rc1 = dv.cli_main(["status"])
    out1 = capsys.readouterr().out
    rc2 = dv.cli_main(argv=["status"])
    out2 = capsys.readouterr().out
    assert rc1 == rc2 == 0
    assert out1 == out2


# ---------------------------------------------------------------------------
# slash command  (HermesCLI._handle_delegation_command)
# ---------------------------------------------------------------------------
#
# The slash handler is the interactive-session mirror of the standalone
# `hermes delegation` subcommand. It must:
#   * default to `status` when no subcommand is given,
#   * swallow argparse's SystemExit (e.g. on --help) so the REPL survives,
#   * catch generic exceptions and print `(._.) delegation: <msg>` instead of
#     bubbling them up and killing the session.
#
# HermesCLI.__init__ takes ~60 args, so we build a bare instance with
# __new__ to skip it — the handler is self-contained and touches no instance
# state beyond `self`.


@pytest.fixture
def hermes_cli():
    """A bare HermesCLI instance (skips the ~60-arg __init__)."""
    from cli import HermesCLI

    return HermesCLI.__new__(HermesCLI)


def test_handle_delegation_command_status_runs(hermes_cli, daemon_up, capsys):
    hermes_cli._handle_delegation_command("/delegation status")
    out = capsys.readouterr().out
    assert "daemon" in out


def test_handle_delegation_command_no_args_defaults_to_status(
    hermes_cli, daemon_up, capsys
):
    hermes_cli._handle_delegation_command("/delegation")
    out = capsys.readouterr().out
    assert "daemon" in out


def test_handle_delegation_command_swallows_argparse_exit(hermes_cli, capsys):
    # argparse calls sys.exit() on --help; the handler must not let the
    # resulting SystemExit propagate and kill the interactive session.
    hermes_cli._handle_delegation_command("/delegation --help")
    # no SystemExit raised => we get here; --help printed usage to stdout.
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_handle_delegation_command_error_is_caught(hermes_cli, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(dv, "cli_main", boom)

    # Must not raise — the error is caught and printed.
    hermes_cli._handle_delegation_command("/delegation status")
    out = capsys.readouterr().out
    assert "(._.) delegation:" in out
    assert "kaboom" in out
