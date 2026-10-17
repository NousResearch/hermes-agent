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
    assert "delegate_task" in refused.stderr


def test_delegated_child_kanban_list_degrades_instead_of_refusing(tmp_path):
    """`kanban list` is a read; a fenced child must list, not fail on the ready refresh (#123733)."""
    home = tmp_path / "hermes"
    home.mkdir()

    created = _run_hermes(home, "kanban", "create", "listable in a fenced lane", "--json")
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]

    listed = _run_hermes(home, "kanban", "list", "--json", marker=True)

    assert listed.returncode == 0, listed.stderr
    assert task_id in listed.stdout
    assert "delegate_task" not in listed.stderr


def test_fenced_child_lists_when_only_pinned_db_is_fenced(tmp_path):
    """`kanban list` must fence on the DB path the connection opens, not just kanban_home().

    A grandchild that moved ``HERMES_KANBAN_HOME`` to its own scratch root but still
    inherits the owner's board via a path-valued marker plus a dispatcher-pinned
    ``HERMES_KANBAN_DB`` leaves ``kanban_home()`` unfenced while ``kanban_db_path()``
    — what ``connect``/``write_txn`` fence on — is read-only. The legacy ``"1"`` marker
    short-circuits ``kanban_path_is_fenced`` and so cannot exercise this divergence;
    here the two paths genuinely disagree, so a guard that checks only ``kanban_home()``
    runs ``recompute_ready`` into a refused write txn (#123733).
    """
    owner_root = tmp_path / "owner"
    owner_root.mkdir()
    db_file = owner_root / "board" / "kanban.db"
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def _run(*args: str, env_extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        for name in (
            "HERMES_KANBAN_BOARD",
            "HERMES_KANBAN_DB",
            "HERMES_KANBAN_WORKSPACES_ROOT",
            "HERMES_DELEGATED_CHILD_CONTEXT",
        ):
            env.pop(name, None)
        env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    # Owner (unfenced) creates and populates the board pinned at db_file.
    created = _run(
        "kanban",
        "create",
        "cross-home fenced board",
        "--json",
        env_extra={
            "HERMES_HOME": str(owner_root),
            "HERMES_KANBAN_HOME": str(owner_root),
            "HERMES_KANBAN_DB": str(db_file),
        },
    )
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]

    # Grandchild: HERMES_KANBAN_HOME points at an unfenced scratch root, but the
    # inherited path marker + pinned HERMES_KANBAN_DB fence the board it reads.
    listed = _run(
        "kanban",
        "list",
        "--json",
        env_extra={
            "HERMES_HOME": str(scratch),
            "HERMES_KANBAN_HOME": str(scratch),
            "HERMES_KANBAN_DB": str(db_file),
            "HERMES_DELEGATED_CHILD_CONTEXT": str(owner_root),
        },
    )
    assert listed.returncode == 0, listed.stderr
    assert task_id in listed.stdout
    assert "delegate_task" not in listed.stderr
