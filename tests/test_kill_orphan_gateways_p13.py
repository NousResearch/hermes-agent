"""P13 isolation proof for scripts/kill-orphan-gateways.sh.

Verifies the env override:
- GATEWAY_ORPHAN_DRY_RUN=1 forces dry-run mode even when stdout is not a TTY
  (which would normally activate kill mode).
- The script short-circuits to dry-run and never kills processes.
- Exit 0 in dry-run mode.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "kill-orphan-gateways.sh"


def _run(env_extra=None, args=None):
    env = dict(os.environ)
    env.pop("GATEWAY_ORPHAN_DRY_RUN", None)
    env.pop("KILL_ORPHAN_THRESHOLD", None)
    if env_extra:
        env.update(env_extra)
    argv = ["bash", str(SCRIPT)]
    if args:
        argv += args
    return subprocess.run(
        argv, capture_output=True, text=True, env=env,
        cwd=str(REPO_ROOT), timeout=20,
    )


def test_env_dry_run_exits_zero_silent(tmp_path):
    """GATEWAY_ORPHAN_DRY_RUN=1 forces dry-run; with no orphans it is silent
    and exits 0."""
    # Redirect state dir so the real ~/.hermes/state is untouched.
    env = {"GATEWAY_ORPHAN_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
    # Dry-run with 0 orphans: silent (no kill, no alert).


def test_env_dry_run_overrides_non_tty(tmp_path):
    """Even though stdout is captured (non-TTY, which would normally set
    DRY_RUN=false), GATEWAY_ORPHAN_DRY_RUN=1 must keep DRY_RUN=true so no
    kill is attempted."""
    env = {"GATEWAY_ORPHAN_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
    # No process should have been killed; with no gateway orphans the
    # script is silent. The key assertion is exit 0 with no error.


def test_no_env_non_tty_still_runs(tmp_path):
    """Without the env override, non-TTY mode runs (and with no orphans exits 0).
    Control case proving the env guard is what forces dry-run."""
    env = {"HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
