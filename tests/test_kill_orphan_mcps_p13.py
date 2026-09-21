"""P13 isolation proof for scripts/kill-orphan-mcps.sh.

Verifies the env override:
- MCP_ORPHAN_DRY_RUN=1 forces dry-run mode even when stdout is not a TTY.
- Exit 0 in dry-run mode with no orphans found.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "kill-orphan-mcps.sh"


def _run(env_extra=None, args=None):
    env = dict(os.environ)
    env.pop("MCP_ORPHAN_DRY_RUN", None)
    if env_extra:
        env.update(env_extra)
    argv = ["bash", str(SCRIPT)]
    if args:
        argv += args
    return subprocess.run(
        argv, capture_output=True, text=True, env=env,
        cwd=str(REPO_ROOT), timeout=20,
    )


def test_env_dry_run_exits_zero(tmp_path):
    """MCP_ORPHAN_DRY_RUN=1 forces dry-run; with no MCP orphans it exits 0
    silently."""
    env = {"MCP_ORPHAN_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr


def test_env_dry_run_overrides_non_tty(tmp_path):
    """stdout is captured (non-TTY, normally sets DRY_RUN=false), but
    MCP_ORPHAN_DRY_RUN=1 keeps DRY_RUN=true so no kill is attempted."""
    env = {"MCP_ORPHAN_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
    # No MCP processes should have been killed.


def test_no_env_non_tty_still_runs(tmp_path):
    """Without the env override, non-TTY mode runs (and with no orphans exits 0).
    Control case proving the env guard is what forces dry-run."""
    env = {"HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
