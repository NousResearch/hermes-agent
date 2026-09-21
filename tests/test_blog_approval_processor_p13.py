"""P13 isolation proof for scripts/blog-approval-processor.sh.

Verifies the env override:
- BLOG_APPROVAL_DRY_RUN=1 prints the dry-run line and exits 0 before any
  Python invocation that flips MDX, builds, or pushes.
- The content_engine / blog.blog_approval import is never reached.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "blog-approval-processor.sh"


def _run(env_extra=None):
    env = dict(os.environ)
    env.pop("BLOG_APPROVAL_DRY_RUN", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, env=env,
        cwd=str(REPO_ROOT), timeout=15,
    )


def test_dry_run_exits_zero_and_prints_plan(tmp_path):
    """BLOG_APPROVAL_DRY_RUN=1 prints the plan and exits 0."""
    env = {"BLOG_APPROVAL_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
    assert "dry-run: would process approvals" in r.stdout


def test_dry_run_never_invokes_python(tmp_path):
    """Dry-run must exit before the cd content_engine + python3 heredoc.
    Pointing HOME to an empty dir: if the python ran it would fail on the
    missing content_engine, but dry-run exits 0 cleanly."""
    env = {"BLOG_APPROVAL_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
    assert "No such file" not in r.stderr
    assert "Traceback" not in r.stderr
    assert "ModuleNotFoundError" not in r.stderr
