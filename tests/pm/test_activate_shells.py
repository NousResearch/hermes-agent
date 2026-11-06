"""``activate`` under bash and zsh: the shell takes on the environment and gives it back.

zsh does not word-split unquoted parameters, has no ``type -t`` and leaves
``BASH_SOURCE`` empty, so each of those bash habits broke activation there.
Runs the real script under each real shell against the shared isolated checkout
(tests/pm/activation_support.py). A shell that is not installed is skipped.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.pm.activation_support import fake_store, posix, sync_checkout

# Spawns children with a home it builds itself; the parent's must stay real.
pytestmark = [pytest.mark.real_machine_home, pytest.mark.platforms("posix")]

# No startup files, so the developer's rc files can neither mask nor cause a failure.
SHELLS = {"bash": ["bash", "--norc", "--noprofile", "-c"], "zsh": ["zsh", "-f", "-c"]}


@pytest.fixture(params=sorted(SHELLS))
def shell(request) -> list[str]:
    if shutil.which(request.param) is None:
        pytest.skip(f"{request.param} is not installed")
    return SHELLS[request.param]


def _run(shell: list[str], root: Path, env: dict, script: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*shell, script.replace("@ACTIVATE@", posix(root / "activate"))],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
        check=False,
    )


def test_activate_applies_the_environment_and_deactivate_restores_the_shell(tmp_path, shell):
    root, env = sync_checkout(tmp_path)
    fake_store(tmp_path)
    # Sourced by path from outside the checkout: the script must find its own repo.
    run = _run(shell, root, env, """
        export PYTHONPATH=caller-original VIRTUAL_ENV=caller-venv
        original_path="$PATH"

        source "@ACTIVATE@" || exit 10
        # The sentinel and the tool variables must reach child processes.
        bash -c 'test -n "$__HERMES_ACTIVATED"' || exit 11
        test "$(bash -c 'printf %s "$HERMES_PM_ACTIVATE_CANARY"')" = env-ok || exit 12
        test "${PYTHONPATH%%:*}" = "$__HERMES_WORKTREE" || exit 13

        # Re-activating deactivates first, so one deactivate undoes everything.
        source "@ACTIVATE@" || exit 14
        deactivate || exit 15

        test "$PATH" = "$original_path" || exit 20
        test "$PYTHONPATH" = caller-original || exit 21
        test "$VIRTUAL_ENV" = caller-venv || exit 22
        test -z "${__HERMES_ACTIVATED+set}" || exit 23
        test -z "${HERMES_PM_ACTIVATE_CANARY+set}" || exit 24
        typeset -f deactivate >/dev/null && exit 25
        typeset -f hermes >/dev/null && exit 26
        exit 0
    """, tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stderr.count("setup progress") == 2, run.stderr


def test_setup_failure_leaves_the_caller_unchanged(tmp_path, shell):
    root, env = sync_checkout(tmp_path)
    fake_store(tmp_path)
    (root / "fail").touch()
    run = _run(shell, root, env, """
        export PYTHONPATH=caller-path
        before="$(env | sort)"

        source "@ACTIVATE@"
        code=$?
        test $code -ne 0 || exit 10
        test "$(env | sort)" = "$before" || exit 11
        test -z "${__HERMES_ACTIVATED+set}" || exit 12
        typeset -f deactivate >/dev/null && exit 13
        exit 0
    """, tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "setup failed" in run.stderr
