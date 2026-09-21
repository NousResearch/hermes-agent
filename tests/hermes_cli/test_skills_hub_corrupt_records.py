"""Corrupt provenance is reported at command boundaries without replacing it."""

import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
from rich.console import Console

from hermes_constants import get_hermes_home
from hermes_cli.skills_hub import handle_skills_slash


@pytest.fixture
def corrupt_record():
    path = get_hermes_home() / "skills" / ".hub" / "lock.json"
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    return path


@pytest.mark.parametrize("args", [("list",), ("snapshot", "export", "-")])
def test_cli_reports_corrupt_provenance_and_recovers(corrupt_record, args):
    command = [sys.executable, "-m", "hermes_cli.main", "skills", *args]
    root = Path(__file__).resolve().parents[2]
    failed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    output = failed.stdout + failed.stderr
    assert failed.returncode == 1
    assert "Invalid skills hub lock file" in output
    assert "Traceback" not in output
    assert corrupt_record.read_text(encoding="utf-8") == "{"

    corrupt_record.write_text(
        json.dumps({"version": 1, "installed": {}}), encoding="utf-8"
    )
    recovered = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert "Invalid skills hub lock file" not in recovered.stdout + recovered.stderr


def test_chat_command_reports_corrupt_provenance_without_exiting(corrupt_record):
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=200)
    handle_skills_slash("/skills list", console=console)
    assert "Invalid skills hub lock file" in output.getvalue()
    assert corrupt_record.read_text(encoding="utf-8") == "{"

    corrupt_record.write_text(
        json.dumps({"version": 1, "installed": {}}), encoding="utf-8"
    )
    handle_skills_slash("/skills list", console=console)
    assert "Installed Skills" in output.getvalue()