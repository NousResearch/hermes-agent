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


# ---------------------------------------------------------------------------
# slash dispatch  (HermesCLI.process_command + commands.py registry)
# ---------------------------------------------------------------------------
#
# Task 3 wires `/delegation` into the central slash dispatcher. Two pieces:
#   1. hermes_cli/commands.py registers a CommandDef named "delegation" so
#      `resolve_command("delegation")` resolves it to the canonical name.
#   2. cli.py's process_command() switch has an
#      `elif canonical == "delegation": self._handle_delegation_command(...)`
#      case so typing `/delegation ...` in the REPL actually fires the handler.
#
# These are integration-style tests: they go through the real process_command
# entry point (the same code path the interactive REPL uses), not the handler
# directly.


def test_commands_registry_knows_delegation():
    """The CommandDef must exist so process_command resolves the canonical name."""
    from hermes_cli.commands import resolve_command

    cd = resolve_command("delegation")
    assert cd is not None, "delegation not registered in hermes_cli/commands.py"
    assert cd.name == "delegation"
    # The M3 module exposes exactly these three subcommands.
    for sub in ("status", "verify", "report"):
        assert sub in cd.subcommands, f"missing subcommand {sub!r}"


def test_process_command_dispatches_delegation(hermes_cli, monkeypatch):
    """`/delegation status` must route to _handle_delegation_command."""
    calls = []
    monkeypatch.setattr(
        type(hermes_cli),
        "_handle_delegation_command",
        lambda self, cmd: calls.append(cmd),
    )
    hermes_cli.process_command("/delegation status")
    assert calls == ["/delegation status"], "handler not invoked via dispatch"


def test_process_command_delegation_integration(hermes_cli, daemon_up, capsys):
    """Full path: /delegation status through the dispatcher emits status output."""
    hermes_cli.process_command("/delegation status")
    out = capsys.readouterr().out
    assert "daemon" in out


def test_process_command_delegation_defaults_to_status(hermes_cli, daemon_up, capsys):
    """Bare /delegation (no subcommand) defaults to status via the dispatcher."""
    hermes_cli.process_command("/delegation")
    out = capsys.readouterr().out
    assert "daemon" in out


def test_process_command_delegation_help_does_not_crash(hermes_cli, capsys):
    """/delegation --help must not propagate SystemExit out of the dispatcher."""
    # Returns truthy (continue REPL); no SystemExit escapes.
    hermes_cli.process_command("/delegation --help")
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_process_command_curator_still_dispatches(hermes_cli, monkeypatch):
    """Regression: adding the delegation case must not break /curator."""
    calls = []
    monkeypatch.setattr(
        type(hermes_cli),
        "_handle_curator_command",
        lambda self, cmd: calls.append(cmd),
    )
    hermes_cli.process_command("/curator status")
    assert calls == ["/curator status"]


def test_process_command_kanban_still_dispatches(hermes_cli, monkeypatch):
    """Regression: adding the delegation case must not break /kanban."""
    calls = []
    monkeypatch.setattr(
        type(hermes_cli),
        "_handle_kanban_command",
        lambda self, cmd: calls.append(cmd),
    )
    hermes_cli.process_command("/kanban")
    assert calls == ["/kanban"]


# ---------------------------------------------------------------------------
# packaging: optional `delegation` extra version constraint (M3.5 Task 1)
# ---------------------------------------------------------------------------

def test_hermes_delegation_version_in_supported_range():
    """The installed ``hermes-delegation`` must satisfy the pyproject extra.

    The ``delegation`` optional-dependency in hermes-agent's pyproject.toml
    pins ``hermes-delegation>=0.1.0,<0.2.0``. This test documents and enforces
    that the editable/installed package actually falls in that range, so a
    drift (e.g. a 0.2.x release with breaking changes) is caught here rather
    than at runtime via the /delegation slash command.

    Skips cleanly when the package isn't installed — the module-level
    ``importorskip`` already guarantees it is by the time we get here, but the
    explicit metadata lookup is guarded too for robustness.
    """
    import importlib.metadata

    from packaging.version import Version

    try:
        raw = importlib.metadata.version("hermes-delegation")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        pytest.skip("hermes-delegation distribution metadata not found")

    version = Version(raw)
    assert Version("0.1.0") <= version < Version("0.2.0"), (
        f"installed hermes-delegation {version} is outside the supported "
        "range [0.1.0, 0.2.0) declared by the `delegation` extra"
    )


# ---------------------------------------------------------------------------
# install hint: runtime auto-detect (M3.5 Task 2)
# ---------------------------------------------------------------------------
#
# When ``hermes_delegation`` is missing, the wrapper must print an install hint
# pointing at the *new* optional-extra install path (``pip install -e
# ".[delegation]"`` from the hermes-agent root, or ``pip install
# "hermes-agent[delegation]"`` for PyPI installs), NOT the old M1+M2-only path
# under ``~/.hermes/delegation-visualizer/``.
#
# The hint is computed at call time via ``_install_hint()`` so it can adapt to
# wherever hermes-agent actually lives (located with
# ``importlib.util.find_spec("hermes_cli")``). We exercise the missing-package
# branch by monkeypatching ``dv.HD_AVAILABLE`` to False.


def test_install_hint_mentions_optional_extra(monkeypatch):
    """The hint must mention the `pip install -e ".[delegation]"` extra path."""
    monkeypatch.setattr(dv, "HD_AVAILABLE", False)
    hint = dv._install_hint()
    assert 'pip install -e ".[delegation]"' in hint, hint


def test_install_hint_mentions_hermes_delegation_package():
    """The hint must name the package so users know what to install."""
    hint = dv._install_hint()
    assert "hermes-delegation" in hint or "hermes_delegation" in hint, hint


def test_install_hint_does_not_mention_old_install_path():
    """The old M1+M2-only path must be gone — it no longer applies."""
    hint = dv._install_hint()
    assert "~/.hermes/delegation-visualizer/" not in hint, hint


def test_install_hint_uses_runtime_detect(monkeypatch, tmp_path):
    """The hint is constructed at call time so it adapts to the agent location.

    We swap out the hermes_cli spec's origin (what find_spec returns) and the
    hint must reflect the new location — proving it is not frozen at import
    time as a module-level constant.
    """
    import importlib.util

    real_find_spec = importlib.util.find_spec

    fake_root = tmp_path / "some" / "other" / "hermes-agent"
    fake_pkg = fake_root / "hermes_cli"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")

    def fake_find_spec(name, *a, **k):
        if name == "hermes_cli":
            return importlib.util.spec_from_file_location(
                "hermes_cli", str(fake_pkg / "__init__.py")
            )
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(dv.importlib.util, "find_spec", fake_find_spec)

    hint = dv._install_hint()
    # The hint should reference the detected hermes-agent root (parent of the
    # hermes_cli package), proving runtime detection drives the message.
    assert str(fake_root) in hint, hint
