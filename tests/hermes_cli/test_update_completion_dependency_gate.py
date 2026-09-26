"""Update completion must verify startup dependencies before reporting success (#122697).

An interrupted update can leave the committed dependency environment missing
startup packages (ruamel and friends) while every input stamp still matches,
and the completion report otherwise probes only the SQLite runtime. These
tests simulate that interrupted state at the Python level: the probe reports
the incomplete environment, one repair runs, and completion is reported only
after importability is restored (or withheld when repair cannot restore it).
The real macOS update lifecycle is never exercised.
"""
from __future__ import annotations

from pathlib import Path

from hermes_cli import sqlite_runtime, update_cmd, update_cmd_maint
from pm.package import InstallError

#: The real completion messages carry a check-mark prefix; built from chr()
#: so this file stays ASCII-only.
COMPLETION = chr(0x2713) + " Update complete!"


class _SimulatedEnvironment:
    """The committed environment as an interrupted update leaves it.

    ``probe`` selects the initial verdict: ``"broken"`` (imports missing) or
    ``"unavailable"`` (no committed environment to probe at all). ``repair``
    controls whether one repair restores the startup imports. Every seam call
    is recorded in ``events`` so the tests can pin the order: probe -> repair
    -> probe -> completion reported.
    """

    def __init__(self, *, probe: str = "broken", repairable: bool = True):
        self.probe = probe
        self.repairable = repairable
        self.healthy = False
        self.events: list[str] = []

    def validate(self, python, *, env, cwd):
        self.events.append("probe")
        if self.probe == "unavailable":
            raise RuntimeError("no committed environment to probe")
        if not self.healthy:
            raise InstallError("venv", "startup validation failed: No module named 'ruamel'")

    def repair(self, project_root):
        self.events.append("repair")
        if not self.repairable:
            raise InstallError("venv", "startup validation failed: No module named 'ruamel'")
        self.healthy = True


def _wire(monkeypatch, environment: _SimulatedEnvironment) -> None:
    """Point the completion gate at the simulated environment's seams."""
    import pm.environments
    import pm.recovery

    monkeypatch.setattr(pm.recovery, "validate_environment", environment.validate)
    monkeypatch.setattr(pm.recovery, "repair_dependencies", environment.repair)
    monkeypatch.setattr(pm.environments, "project_python", lambda root: Path("<selected-python>"))
    monkeypatch.setattr(pm.environments, "activation_environment", lambda root: {})
    monkeypatch.setattr(sqlite_runtime, "probe_sqlite_runtime", lambda python: None)
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda: "")
    monkeypatch.setenv("HERMES_ACTION_ID", "invalid")

    original = update_cmd_maint._print_update_completion

    def report(message):
        environment.events.append("report")
        original(message)

    monkeypatch.setattr(update_cmd_maint, "_print_update_completion", report)


def test_completion_repairs_missing_startup_dependency_before_reporting(monkeypatch, capsys):
    environment = _SimulatedEnvironment()
    _wire(monkeypatch, environment)

    complete = update_cmd_maint._print_verified_update_completion(COMPLETION)

    assert complete is True
    # Repair ran and restored importability BEFORE completion was reported.
    assert environment.events == ["probe", "repair", "probe", "report"]
    assert environment.healthy
    output = capsys.readouterr().out
    assert "dependency environment repaired" in output
    assert "Update complete!" in output


def test_completion_withheld_when_dependency_repair_fails(monkeypatch, capsys):
    environment = _SimulatedEnvironment(repairable=False)
    _wire(monkeypatch, environment)

    complete = update_cmd_maint._print_verified_update_completion(COMPLETION)

    assert complete is False
    assert environment.events == ["probe", "repair"]
    output = capsys.readouterr().out
    assert "Update complete!" not in output
    assert "completion withheld" in output


def test_healthy_environment_completes_without_repair(monkeypatch, capsys):
    environment = _SimulatedEnvironment()
    environment.healthy = True
    _wire(monkeypatch, environment)

    complete = update_cmd_maint._print_verified_update_completion(COMPLETION)

    assert complete is True
    assert environment.events == ["probe", "report"]
    assert "Update complete!" in capsys.readouterr().out


def test_unavailable_probe_does_not_block_completion(monkeypatch, capsys):
    # Grace, same philosophy as the SQLite probe: only a positive import-level
    # failure repairs or withholds; an unprobeable install must not block.
    environment = _SimulatedEnvironment(probe="unavailable")
    _wire(monkeypatch, environment)

    complete = update_cmd_maint._print_verified_update_completion(COMPLETION)

    assert complete is True
    assert "repair" not in environment.events
    assert "Update complete!" in capsys.readouterr().out
