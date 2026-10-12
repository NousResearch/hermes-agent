"""``activate.fish``: fish takes on the environment and gives it back, as ``activate`` does for bash.

Runs the real script under a real fish against the shared isolated checkout
(tests/pm/activation_support.py). Skipped where fish is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.pm.activation_support import fake_store, posix, sync_checkout

# Spawns children with a home it builds itself; the parent's must stay real.
pytestmark = [
    pytest.mark.real_machine_home,
    pytest.mark.platforms("posix"),
    pytest.mark.skipif(shutil.which("fish") is None, reason="fish is not installed"),
]


def _fish(root: Path, env: dict, script: str, tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["fish", "--no-config", "-c", script.replace("@ACTIVATE@", posix(root / "activate.fish"))],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )


def test_activate_applies_the_environment_and_deactivate_restores_the_shell(tmp_path):
    root, env = sync_checkout(tmp_path)
    fake_store(tmp_path)
    run = _fish(root, env, """
        set -gx PYTHONPATH caller-original
        set -gx VIRTUAL_ENV caller-venv
        set -l original_path (string join : $PATH)

        source "@ACTIVATE@"
        or exit 10
        # The sentinel and the tool variables must reach child processes.
        bash -c 'test -n "$__HERMES_ACTIVATED"'; or exit 11
        test (bash -c 'printf %s "$HERMES_PM_ACTIVATE_CANARY"') = env-ok; or exit 12
        # PYTHONPATH stays a list: the checkout first, then the selected environment.
        printf '%s\\n' $PYTHONPATH

        # Re-activating deactivates first, so one deactivate undoes everything.
        source "@ACTIVATE@"
        or exit 13
        deactivate

        test (string join : $PATH) = $original_path; or exit 20
        test "$PYTHONPATH" = caller-original; or exit 21
        test "$VIRTUAL_ENV" = caller-venv; or exit 22
        set -q __HERMES_ACTIVATED; and exit 23
        set -q HERMES_PM_ACTIVATE_CANARY; and exit 24
        functions -q deactivate; and exit 25
        functions -q hermes; and exit 26
        functions -q __hermes_saved_fish_prompt; and exit 27
        exit 0
    """, tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.splitlines()[0] == str(root)
    assert "/first/venv/" in run.stdout.splitlines()[1]


def test_setup_failure_leaves_the_caller_unchanged(tmp_path):
    root, env = sync_checkout(tmp_path)
    fake_store(tmp_path)
    (root / "fail").touch()
    run = _fish(root, env, """
        set -gx PYTHONPATH caller-path
        set -l before (env | sort | string collect)

        source "@ACTIVATE@"
        set -l code $status
        test $code -ne 0; or exit 10
        test (env | sort | string collect) = "$before"; or exit 11
        set -q __HERMES_ACTIVATED; and exit 12
        functions -q deactivate; and exit 13
        exit 0
    """, tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "setup failed" in run.stderr
