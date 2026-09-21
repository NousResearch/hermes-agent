"""P13 isolation proof for scripts/reap-workspace-mcp.sh.

Verifies the dry-run flag:
- REAP_DRY_RUN=1 lists candidates only, never kills, never writes the log
- REAP_DRY_RUN=1 with no candidates prints a dry-run summary line
- REAP_LOG_PATH can be redirected to a temp file so the production log is
  never touched
- exit 0 in both dry-run and (synthetic) live runs
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "reap-workspace-mcp.sh"


def _run(env_extra: dict | None = None):
    env = dict(os.environ)
    env.pop("REAP_DRY_RUN", None)
    env.pop("REAP_LOG_PATH", None)
    env.pop("REAP_MAX_AGE_SECONDS", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=15,
    )


def test_dry_run_does_not_touch_log(tmp_path):
    """REAP_DRY_RUN=1 must not create or write the redirected log file."""
    log = tmp_path / "reap.log"
    r = _run({"REAP_DRY_RUN": "1", "REAP_LOG_PATH": str(log)})
    assert r.returncode == 0, r.stderr
    assert "dry-run reap complete" in r.stdout
    assert not log.exists(), "dry-run wrote to the log file"


def test_dry_run_emits_summary_with_no_candidates(tmp_path):
    """With no workspace-mcp processes running the dry-run summary is
    printed with would_kill=0 kept=0 and exits 0."""
    r = _run({"REAP_DRY_RUN": "1", "REAP_LOG_PATH": str(tmp_path / "x.log")})
    assert r.returncode == 0, r.stderr
    assert "dry-run reap complete: would_kill=0 kept=0" in r.stdout


def test_default_run_writes_log_when_redirected(tmp_path):
    """Without REAP_DRY_RUN the live mode still exits 0 with no candidates
    and writes the log line (the log path is redirected so the production
    log is untouched)."""
    log = tmp_path / "reap.log"
    r = _run({"REAP_LOG_PATH": str(log)})
    assert r.returncode == 0, r.stderr
    assert log.exists(), "live mode wrote no log line"
    assert "reap complete: killed=0 kept=0" in log.read_text()
