"""CLI diagnostics must report the same target error without masking it as a missing install (#115033)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from hermes_cli.config import save_config
from hermes_cli.subcommands.computer_use import build_computer_use_parser
from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd


@pytest.mark.parametrize("target", ["mac", "unsupported-host"])
@pytest.mark.parametrize("command", [
    ["status"], ["permissions", "status"], ["permissions", "status", "--json"],
    ["doctor"], ["doctor", "--json"],
])
def test_cli_target_errors_keep_diagnostics_and_failure_exit(tmp_path, monkeypatch, capsys, target, command):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(tmp_path / "missing-driver"))
    monkeypatch.setattr("hermes_constants.is_wsl", lambda: False)
    if target == "unsupported-host":
        target = "windows" if sys.platform == "linux" else "linux"
    save_config({"computer_use": {"target": target}})
    with pytest.raises(ValueError) as invalid:
        resolve_cua_driver_cmd()
    diagnostic = str(invalid.value)
    parser = argparse.ArgumentParser()
    build_computer_use_parser(parser.add_subparsers())
    args = parser.parse_args(["computer-use", *command])
    try:
        code = args.func(args)
    except SystemExit as exc:
        code = exc.code
    assert code == (2 if command[0] == "doctor" else 1)
    out, err = capsys.readouterr()
    assert diagnostic in out + err
    assert "not installed" not in out + err
    assert "Traceback" not in out + err
    if command == ["permissions", "status", "--json"]:
        status = json.loads(out)
        assert status["error"] == diagnostic
        assert status["installed"] is False and status["ready"] is None
