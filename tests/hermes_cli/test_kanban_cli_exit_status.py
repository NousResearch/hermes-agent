"""Regression coverage for Kanban CLI process exit status propagation."""

from __future__ import annotations

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


def test_delegated_child_kanban_cli_refusal_returns_nonzero_exit_status(tmp_path):
    """A printed Kanban mutation refusal must not look like CLI success."""
    home = tmp_path / "hermes"
    home.mkdir()

    created = _run_hermes(home, "kanban", "create", "exit status probe", "--json")
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]

    refused = _run_hermes(
        home,
        "kanban",
        "comment",
        task_id,
        "must be refused",
        marker=True,
    )

    assert refused.returncode == 1
    assert "delegate_task child contexts cannot mutate Kanban tasks via the CLI" in refused.stderr


def test_delegated_child_can_still_read_kanban(tmp_path):
    """A delegated child shelling out to the CLI must be able to read.

    Regression for the board-list "(empty)" + "could not initialize
    database" bug: `init_db()` runs an idempotent migration within
    `connect()`'s init path, and `recompute_ready` runs inside `list` — both
    are harmless writes that a child MUST be allowed to perform so read
    commands return real data instead of failing on the mutation guard.
    """
    home = tmp_path / "hermes"
    home.mkdir()

    seeded = _run_hermes(home, "kanban", "create", "seed for read", "--json")
    assert seeded.returncode == 0, seeded.stderr

    listed = _run_hermes(home, "kanban", "list", marker=True)
    assert listed.returncode == 0, listed.stderr
    assert "seed" in listed.stdout and "for" in listed.stdout

    boards = _run_hermes(home, "kanban", "boards", "list", marker=True)
    assert boards.returncode == 0, boards.stderr
    assert "(empty)" not in boards.stdout
    assert "ready=1" in boards.stdout
