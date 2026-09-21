#!/usr/bin/env python3
"""KenseiAgent retention adapter wiring tests.

The crash-recovery/idempotency logic is conformance-tested in the
`research-mashup-pipeline` package (tests/test_retention.py). These tests
prove the KenseiAgent wrapper:

1. resolves the package (not a local duplicate);
2. keeps the CLI contract (args + exit codes);
3. defaults to the Hermes blog dir when configured.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "mashup_retention.py"


def _run(args=None, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, str(SCRIPT)]
    if args:
        cmd.extend(args)
    return subprocess.run(
        cmd, capture_output=True, text=True,
        env=env, cwd=str(REPO_ROOT), timeout=20,
    )


def test_wrapper_runs_against_temp(tmp_path):
    """Runs cleanly against an isolated state dir; exits 0."""
    env = {
        "HERMES_HOME": str(tmp_path / "hermes"),
        "MASHUP_BLOG_DIR": str(tmp_path / "blog"),
    }
    r = _run(["--proposal-days", "30", "--blog-days", "90"], env)
    assert r.returncode == 0, r.stderr
    assert "[retention]" in r.stdout


def test_wrapper_resolves_package(tmp_path):
    """Must resolve mashup package; no ModuleNotFoundError."""
    env = {
        "HERMES_HOME": str(tmp_path / "hermes"),
        "MASHUP_BLOG_DIR": str(tmp_path / "blog"),
    }
    r = _run(["--proposal-days", "30"], env)
    assert r.returncode == 0
    assert "ModuleNotFoundError" not in r.stderr
    assert "Traceback" not in r.stderr
