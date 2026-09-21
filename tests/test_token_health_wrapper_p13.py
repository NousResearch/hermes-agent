"""P13 dry-run contract test for scripts/token_health_wrapper.sh.

TOKEN_HEALTH_DRY_RUN=1 must short-circuit before any Python invocation.
The runbook output path uses HERMES_HOME, not a hardcoded ~/.hermes.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "token_health_wrapper.sh"


def _run(env_extra=None):
    env = dict(os.environ)
    env.pop("TOKEN_HEALTH_DRY_RUN", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True,
        env=env, cwd=str(REPO_ROOT), timeout=15,
    )


def test_dry_run_exits_zero_and_prints_plan(tmp_path):
    """TOKEN_HEALTH_DRY_RUN=1 prints the dry-run line and exits 0."""
    env = {"TOKEN_HEALTH_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
    assert "dry-run: would run token_health" in r.stdout


def test_dry_run_never_invokes_python(tmp_path):
    """Dry-run must exit before the python token_health.py invocation."""
    env = {"TOKEN_HEALTH_DRY_RUN": "1", "HOME": str(tmp_path)}
    r = _run(env)
    assert r.returncode == 0, r.stderr
    assert "No such file" not in r.stderr
    assert "Traceback" not in r.stderr


def test_respects_hermes_home():
    """Runbook path must use HERMES_HOME, not hardcoded /home/kensei/.hermes."""
    text = SCRIPT.read_text()
    assert "/home/kensei/.hermes" not in text, (
        "token_health_wrapper.sh: hardcoded /home/kensei/.hermes path"
    )
    assert "HERMES_HOME" in text, (
        "token_health_wrapper.sh: does not reference HERMES_HOME"
    )
