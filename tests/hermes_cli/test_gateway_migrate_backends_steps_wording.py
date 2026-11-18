"""The printed by-hand steps quote what ``hermes gateway status`` and ``hermes webapp --status`` really print.

A step that says "if it prints X" is only followable when X is a line the command prints. Each quoted phrase
is checked against the source of the status commands, so a wording change there fails here instead of in a
user's terminal. The closing ``webapp --status`` runs while a just-started gateway may still be binding, so
the step before it says to wait and re-run.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_migrate_backends_plan as plan
from hermes_cli import gateway_migrate_backends_windows as win

_ROOT = Path(plan.__file__).resolve().parent
_H = "hermes -p work"


def _source(*names: str) -> str:
    return "\n".join((_ROOT / n).read_text(encoding="utf-8") for n in names)


def _quoted(line: str) -> list[str]:
    return re.findall(r"'([^']+)'", line)


def _branch(steps: list[str]) -> str:
    (line,) = [s for s in steps if s.startswith("# then by what it prints")]
    return line


def _system_unit_steps():
    item = SimpleNamespace(supervisor="systemd", scope="system", unit="hermes-dashboard.service", pid=None)
    return plan.manual_steps("127.0.0.1", 9119, item, env_keys=(), home="/home/u/.hermes/profiles/work")


def _windows_steps():
    item = SimpleNamespace(supervisor="scheduled-task", unit="Hermes Dashboard", unit_path=None, pid=11)
    return win.manual_steps("127.0.0.1", 9119, item, home="/home/u/.hermes/profiles/work")


def test_the_posix_branch_quotes_what_gateway_status_prints():
    line = _branch(_system_unit_steps())
    phrases = _quoted(line)

    assert "not installed ->" not in line  # `hermes gateway status` never prints "not installed" there
    source = _source("gateway.py", "gateway_launchd.py")
    assert phrases and all(p in source for p in phrases), [p for p in phrases if p not in source]
    for verb in ("install", "start", "restart"):
        assert f"-> {_H} gateway {verb}" in line


def test_the_windows_branch_quotes_what_gateway_status_prints_there():
    line = _branch(_windows_steps())
    phrases = _quoted(line)

    source = _source("gateway_windows.py")
    assert phrases and all(p in source for p in phrases), [p for p in phrases if p not in source]
    for verb in ("install", "start", "restart"):
        assert f"-> {_H} gateway {verb}" in line


@pytest.mark.parametrize("steps", [_system_unit_steps, _windows_steps,
                                   lambda: plan.manual_steps("127.0.0.1", 9119, None, env_keys=(),
                                                             home="/home/u/.hermes/profiles/work", no_service=True)])
def test_the_closing_check_says_to_wait_and_re_run_while_the_gateway_binds(steps):
    got = steps()

    assert got[-1] == f"{_H} webapp --status"
    wait = got[-2]
    assert wait.startswith("# ") and "wait" in wait and "again" in wait
    phrases = _quoted(wait)
    source = _source("webapp_launcher.py")
    assert phrases and all(p in source for p in phrases), [p for p in phrases if p not in source]
