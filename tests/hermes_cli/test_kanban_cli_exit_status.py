"""Regression coverage for Kanban CLI process exit status propagation."""

from __future__ import annotations

import pytest

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _run_hermes(home: Path, *args: str, marker: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_HOME"] = str(home)
    for name in (
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if marker:
        env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"
    else:
        env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("marker", [False, True])
def test_removed_kanban_cli_rejects_normal_and_delegated_calls(tmp_path, marker):
    home = tmp_path / "hermes"
    home.mkdir()
    refused = _run_hermes(home, "kanban", "create", "must be refused", marker=marker)
    assert refused.returncode != 0
    assert "kanban" in refused.stderr
    assert not (home / "kanban.db").exists()
