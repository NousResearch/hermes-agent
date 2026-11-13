"""A Kanban worker's imports never come from its task workspace.

The worker is spawned with ``cwd=workspace``; under a plain ``-m`` Python prepends that cwd to
``sys.path``, so a workspace that is a git worktree of the Hermes repo shadowed the pinned tree and
the worker died on "no dependency environment is committed for this install".
"""

from __future__ import annotations

import os
import subprocess

from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.kanban_db import Task


def _task() -> Task:
    return Task(
        id="t_argv1", title="t", body="", assignee="worker", status="running", priority=0,
        created_by=None, created_at=1, started_at=None, completed_at=None,
        workspace_kind="worktree", workspace_path=None, claim_lock=None, claim_expires=None,
        tenant=None,
    )


def test_worker_interpreter_does_not_import_from_its_workspace_cwd(tmp_path):
    argv = dispatch._worker_argv(_task(), "worker", None)
    interpreter = argv[:argv.index("-m")]
    assert "-P" in interpreter
    (tmp_path / "shadow_probe.py").write_text("print('SHADOWED')\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONSAFEPATH", "PYTHONPATH")}

    result = subprocess.run([*interpreter, "-m", "shadow_probe"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60, check=False)

    assert "SHADOWED" not in result.stdout
    assert "No module named shadow_probe" in result.stderr
