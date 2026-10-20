"""A dispatcher worker must boot from the dispatcher's tree even when cwd is a worktree.

The dispatcher spawns ``python -P -m hermes_cli.kanban_worker_client`` with
``cwd = <task workspace>``. ``python -m`` prepends cwd to ``sys.path`` unless
``-P`` is passed, so a worktree workspace used to shadow the install tree's
``hermes_bootstrap`` with the worktree's own copy: that copy resolves its root to
the worktree, which has no PM install state, and the client exited
``NoCommittedEnvironment`` before its first turn. Six consecutive worktree-cwd
runs died that way while every scratch-workspace run in the same window booted.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


# Prints the origin of `hermes_bootstrap` without importing it: the module's own
# import is the bootstrap side effect under test, so only its resolution matters.
PROBE_SOURCE = """import importlib.util

spec = importlib.util.find_spec("hermes_bootstrap")
print(spec.origin if spec else "")
"""


def _task() -> kb.Task:
    return kb.Task(
        id="t_boot_cwd",
        title="boot cwd",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="worktree",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def test_worker_boot_ignores_workspace_copy_of_hermes_bootstrap(tmp_path):
    """The interpreter flags the dispatcher spawns the worker with must keep cwd
    off ``sys.path``; the PYTHONPATH-pinned install tree is the only Hermes tree
    the worker may import."""
    install_root = Path(kbd.__file__).resolve().parents[1]
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    (workspace / "hermes_bootstrap.py").write_text(
        'raise SystemExit("imported the workspace copy of hermes_bootstrap")\n',
        encoding="utf-8",
    )
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "bootprobe.py").write_text(PROBE_SOURCE, encoding="utf-8")

    worker_argv = kbd._worker_argv(_task(), "worker", None)
    # The payload module is swapped for the probe; every interpreter flag the
    # dispatcher passes is kept, so this asserts the real spawn contract.
    assert worker_argv[-2:] == ["-m", "hermes_cli.kanban_worker_client"]
    probe_argv = [*worker_argv[:-2], "-m", "bootprobe"]

    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(probe_dir), str(install_root)])}
    # Prove `-P` does the work rather than an inherited interpreter policy.
    env.pop("PYTHONSAFEPATH", None)

    result = subprocess.run(
        probe_argv, cwd=workspace, env=env, capture_output=True, text=True, timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    resolved = Path(result.stdout.strip()).resolve()
    assert resolved == (install_root / "hermes_bootstrap.py").resolve(), (
        f"worker boot resolved hermes_bootstrap from {resolved}, not the install tree"
    )
