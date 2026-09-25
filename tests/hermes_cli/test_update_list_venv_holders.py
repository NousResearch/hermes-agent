"""``hermes update --list-venv-holders``: read-only JSON twin of the Windows venv-holder refusal.

Drives ``cmd_update`` (the production entry point) so the wiring in ``_update_preflight_handled``
is what is proven, not the helper alone.
"""

from __future__ import annotations

import argparse
import json

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd_windows


def _args(**overrides):
    base = dict(list_venv_holders=True, plan=False, check=False, gateway=False, branch=None)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def _quiet_preflight(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    # No live psutil lookups: the fixture tuple IS the process table.
    monkeypatch.setattr(update_cmd_windows, "_psutil", lambda: None)


def test_list_venv_holders_json_and_exit_3_when_holders_present(monkeypatch, capsys, _quiet_preflight):
    holders = [
        (4242, "python.exe", r"C:\hermes\venv\Scripts\python.exe -m hermes_cli.main serve --port 8642"),
        (4343, "python.exe", r"C:\hermes\venv\Scripts\python.exe -m hermes_cli.main gateway run"),
        (4444, "python.exe", r"C:\hermes\venv\Scripts\python.exe -m hermes_cli.main -p work kanban list"),
        (4545, "python.exe", r"C:\hermes\venv\Scripts\python.exe some_script.py"),
    ]
    monkeypatch.setattr(update_cmd_windows, "_detect_venv_python_processes", lambda: holders)

    with pytest.raises(SystemExit) as exc:
        cli_main.cmd_update(_args())

    assert exc.value.code == update_cmd_windows.VENV_HOLDERS_EXIT == 3
    payload = json.loads(capsys.readouterr().out)
    assert [(h["pid"], h["kind"]) for h in payload] == [
        (4242, "backend"), (4343, "gateway"), (4444, "hermes:kanban"), (4545, "python")]
    assert set(payload[0]) == {"pid", "exe", "argv", "kind"}
    assert payload[0]["argv"].endswith("serve --port 8642")


def test_list_venv_holders_empty_list_exits_zero_without_updating(monkeypatch, capsys, _quiet_preflight):
    monkeypatch.setattr(update_cmd_windows, "_detect_venv_python_processes", lambda: [])

    def _boom(*_a, **_k):  # the mutating update body must never run behind the read-only flag
        raise AssertionError("update body ran")

    monkeypatch.setattr("hermes_cli.update_cmd._cmd_update_impl", _boom)

    assert cli_main.cmd_update(_args()) is None
    assert json.loads(capsys.readouterr().out) == []


def test_list_venv_holders_reads_the_live_scan_not_the_retired_main_alias(monkeypatch, _quiet_preflight):
    # The holder scan must use this module's live detector; the ``hermes_cli.main`` alias is the
    # retired no-work shim that always returns [] (#123050). The reverse patching also shows why
    # the old tests hid the bug: patching the main alias used to satisfy the call chain while
    # production stayed on the unconditional [].
    rows = [(5050, "python.exe", r"C:\hermes\venv\Scripts\python.exe -m hermes_cli.main gateway run")]
    monkeypatch.setattr(cli_main, "_detect_venv_python_processes", lambda: pytest.fail("retired main alias used"))
    monkeypatch.setattr(update_cmd_windows, "_detect_venv_python_processes", lambda: rows)

    holders = update_cmd_windows.list_venv_holders()

    assert [(h["pid"], h["kind"]) for h in holders] == [(5050, "gateway")]
