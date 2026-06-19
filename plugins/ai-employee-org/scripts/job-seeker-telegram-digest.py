#!/usr/bin/env python3
"""Cron no_agent script: run job-seeker scan and print Telegram-ready digest to stdout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

OPS_SCAN = Path(os.environ.get("JOB_SEEKER_OPS_DIR", r"C:/Users/downl/Documents/ops/job-seeker/run_scan.py"))
HERMES_REPO = os.environ.get(
    "HERMES_REPO",
    r"C:\Users\downl\Documents\New project\hermes-agent",
)
DEFAULT_HERMES_HOME = Path.home() / ".hermes"


def main() -> int:
    if not OPS_SCAN.is_file():
        print(f"ERROR: run_scan.py not found at {OPS_SCAN}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    # Gmail OAuth token lives on the default profile, not job-seeker.
    env["HERMES_HOME"] = str(DEFAULT_HERMES_HOME)
    if HERMES_REPO and Path(HERMES_REPO).is_dir():
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = HERMES_REPO if not prev else f"{HERMES_REPO}{os.pathsep}{prev}"

    result = subprocess.run(
        [sys.executable, str(OPS_SCAN)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        cwd=str(OPS_SCAN.parent),
        env=env,
    )
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(result.stdout or result.stderr or "run_scan.py failed")
        return result.returncode

    digest = result.stdout.strip()
    if not digest or digest.upper() == "[SILENT]":
        return 0

    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
