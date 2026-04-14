#!/usr/bin/env python3
"""Canonical readiness gate for autonomy foundations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.autonomy_guard import load_autonomy_policy


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode


def main() -> int:
    policy = load_autonomy_policy()
    readiness = policy.get("readiness", {})
    pytest_targets = [str(target) for target in readiness.get("pytest_targets", [])]
    smoke_script = readiness.get("smoke_script", "scripts/smoke_autonomy.py")

    if pytest_targets:
        rc = _run([sys.executable, "-m", "pytest", *pytest_targets, "-q"])
        if rc != 0:
            return rc

    return _run([sys.executable, smoke_script])


if __name__ == "__main__":
    raise SystemExit(main())
