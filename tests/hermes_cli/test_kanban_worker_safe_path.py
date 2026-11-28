"""A Kanban worker launch must not import task-workspace files in place of the stdlib.

The dispatcher spawns ``_worker_argv`` with the task workspace as cwd. Plain ``-m`` puts that cwd
first on ``sys.path``, so a workspace ``inspect.py`` (left by an earlier run) replaced the stdlib
module and the worker died at start with ``module 'inspect' has no attribute 'signature'``.
"""

from __future__ import annotations

import json
import subprocess

from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.kanban_db import Task


def _task() -> Task:
    return Task(
        id="t_shadow", title="t", body=None, assignee="worker", status="running", priority=0,
        created_by=None, created_at=1, started_at=None, completed_at=None, workspace_kind="scratch",
        workspace_path=None, claim_lock=None, claim_expires=None, tenant=None,
    )


def test_worker_launch_ignores_stdlib_named_workspace_files(tmp_path):
    for name in ("inspect", "json"):
        (tmp_path / f"{name}.py").write_text(f"raise RuntimeError('workspace {name}.py shadowed the stdlib')\n")
    argv = dispatch._worker_argv(_task(), "worker", None)
    assert argv[-2:] == ["-m", "hermes_cli.kanban_worker_client"]
    # Same interpreter flags, probe in place of the module: what would the worker import?
    probe = "import inspect, json; print(json.dumps([inspect.__file__, hasattr(inspect, 'signature')]))"
    out = subprocess.run([*argv[:-2], "-c", probe], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    inspect_file, has_signature = json.loads(out.stdout)
    assert not inspect_file.startswith(str(tmp_path))
    assert has_signature
