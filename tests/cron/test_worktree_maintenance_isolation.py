"""Background worktree maintenance must not initialize the interactive CLI."""

import os
from pathlib import Path
import subprocess
import sys


def test_cold_worktree_maintenance_preserves_terminal_cwd(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--initial-branch=main", str(repo)], check=True,
                   capture_output=True, timeout=15)
    (repo / ".worktrees").mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               TERMINAL_CWD=str(workspace))
    script = """
import os
import sys
import threading
from cron import scheduler
from hermes_cli import worktree_ops

assert 'cli' not in sys.modules, 'requires a cold CLI import'
before = os.environ['TERMINAL_CWD']
finished = threading.Event()
errors = []
prune = worktree_ops._prune_stale_worktrees

def observed_prune(repo):
    try:
        prune(repo)
    except Exception as exc:
        errors.append(exc)
    finally:
        finished.set()

worktree_ops._prune_stale_worktrees = observed_prune
scheduler._worktree_maintenance_repos = lambda: [sys.argv[1]]
scheduler._maybe_run_worktree_maintenance()
assert finished.wait(30), 'maintenance did not finish'
assert not errors, errors
assert os.environ['TERMINAL_CWD'] == before
"""
    result = subprocess.run([sys.executable, "-c", script, str(repo)], env=env,
                            cwd=tmp_path, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
