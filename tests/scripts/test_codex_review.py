"""End-to-end coverage for the Codex review wrapper watchdog."""

import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_SCRIPT = REPO_ROOT / ".codex" / "scripts" / "codex-review.sh"


def test_three_unchanged_heartbeats_terminate_review_as_failed(tmp_path):
    fake_codex = tmp_path / "codex"
    terminated_marker = tmp_path / "terminated"
    child_terminated_marker = tmp_path / "child-terminated"
    fake_codex.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  printf 'codex-test 1.0\\n'
  exit 0
fi
(
  trap 'touch "$CODEX_TEST_CHILD_TERMINATED_MARKER"; exit 0' TERM
  while :; do
    sleep 1
  done
) &
child_pid=$!
trap 'touch "$CODEX_TEST_TERMINATED_MARKER"; wait "$child_pid"; exit 0' TERM
printf 'review started but made no further progress\\n'
while :; do
  sleep 1
done
"""
    )
    fake_codex.chmod(0o755)
    env = {
        **os.environ,
        "CODEX_HOME": str(tmp_path / "source-codex-home"),
        "CODEX_REVIEW_CODEX_BIN": str(fake_codex),
        "CODEX_REVIEW_HEARTBEAT_SECONDS": "1",
        "CODEX_TEST_TERMINATED_MARKER": str(terminated_marker),
        "CODEX_TEST_CHILD_TERMINATED_MARKER": str(child_terminated_marker),
        "TMPDIR": str(tmp_path),
    }

    started_at = time.monotonic()
    proc = subprocess.run(
        [str(REVIEW_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert proc.returncode == 124
    assert time.monotonic() - started_at < 8
    assert terminated_marker.exists()
    assert child_terminated_marker.exists()
    assert proc.stderr.count("log unchanged") == 3
    assert "heartbeat 1/3" in proc.stderr
    assert "heartbeat 2/3" in proc.stderr
    assert "heartbeat 3/3" in proc.stderr
    assert "terminating its process group so it can be rerun" in proc.stderr
    assert "Full Codex review log:" in proc.stderr
