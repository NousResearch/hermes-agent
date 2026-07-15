"""Documentation gate: the real `hermes oauth-broker` CLI must expose every
verb documented in docs/user-guide/features/oauth-broker.md.

Runs the actual entry point in a subprocess; conftest's hermetic
HERMES_HOME env is inherited, so no live profile is ever read.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTED_VERBS = (
    "run",
    "status",
    "doctor",
    "install",
    "uninstall",
    "auth",
    "migrate",
    "rollback",
)
DOCUMENTED_AUTH_VERBS = ("login", "status", "logout")


def _help(argv):
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *argv, "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_top_level_help_lists_every_documented_verb():
    out = _help(["oauth-broker"])
    for verb in DOCUMENTED_VERBS:
        assert verb in out, f"documented verb {verb!r} missing from --help"


def test_auth_help_lists_every_documented_auth_verb():
    out = _help(["oauth-broker", "auth"])
    for verb in DOCUMENTED_AUTH_VERBS:
        assert verb in out, f"documented auth verb {verb!r} missing from --help"
