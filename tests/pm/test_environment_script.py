"""``pm.environments.shell_exports``: the composed environment as a script a shell can evaluate.

Every shell that takes on the environment (``activate``, ``activate.fish``,
``scripts/run-in-hermes-env``) evaluates this output, so the contract is what a
real shell reads back: any value survives its dialect's quoting, and names the
shell cannot assign are left out instead of failing the whole script.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from pm.environments import shell_exports
from tests.pm.activation_support import bash

NASTY = {
    "PLAIN": "value",
    "SPACES": "a b  c",
    "QUOTES": "it's a \"test\"",
    "SHELL_SYNTAX": "$HOME `id` $(id) ; & | > < * ? [x] ~ #",
    "BACKSLASHES": "C:\\path\\to\\ \\n \\'",
    "EMPTY": "",
    "MULTILINE": "first\nsecond",
}

# Prints each value NUL-terminated, in the order of NASTY; `$NAME` reads the same in both shells.
PROBE = "printf '%s\\0' " + " ".join(f'"${name}"' for name in NASTY)


def _shell(dialect: str) -> list[str]:
    if dialect == "sh":
        return [bash(), "-c"]
    found = shutil.which("fish")
    if found is None:
        pytest.skip("fish is not installed")
    return [found, "--no-config", "-c"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("dialect", ["sh", "fish"])
def test_a_real_shell_reads_every_value_back_unchanged(dialect):
    script = shell_exports(NASTY, dialect) + "\n" + PROBE
    run = subprocess.run([*_shell(dialect), script], capture_output=True, text=True, timeout=30, check=False)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split("\0")[:-1] == list(NASTY.values())


@pytest.mark.platforms("posix")
def test_fish_keeps_path_variables_as_lists():
    script = shell_exports({"PATH": "/a b:/c", "PYTHONPATH": "/x:/y"}, "fish")
    run = subprocess.run(
        [*_shell("fish"), script + "\nprintf '%s ' (count $PATH) (count $PYTHONPATH)"],
        capture_output=True, text=True, timeout=30,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["2", "2"]


@pytest.mark.platforms("posix")
def test_fish_read_only_variables_do_not_abort_the_script():
    script = shell_exports({"PWD": "/elsewhere", "SHLVL": "9", "AFTER": "reached"}, "fish")
    run = subprocess.run([*_shell("fish"), script + "\nprintf %s $AFTER"],
                         capture_output=True, text=True, timeout=30, check=False)
    assert run.returncode == 0 and run.stderr == "", run.stderr
    assert run.stdout == "reached"


@pytest.mark.parametrize("dialect", ["sh", "fish"])
def test_names_no_shell_can_assign_are_left_out(dialect):
    script = shell_exports({"ProgramFiles(ARM)": "x", "1BAD": "x", "GOOD": "y"}, dialect)
    assert "GOOD" in script
    assert "ProgramFiles" not in script and "1BAD" not in script
