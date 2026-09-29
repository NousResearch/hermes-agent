"""``pm.environments.shell_exports``: the composed environment as a script a shell can evaluate.

Both ``activate`` and ``scripts/run-in-hermes-env`` evaluate this output, so the
contract is what a real shell reads back: any value survives the quoting, and
names the shell cannot assign are left out instead of failing the whole script.
"""

from __future__ import annotations

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

# Prints each value NUL-terminated, in the order of NASTY.
PROBE = "printf '%s\\0' " + " ".join(f'"${name}"' for name in NASTY)


@pytest.mark.platforms("posix")
def test_a_real_shell_reads_every_value_back_unchanged():
    script = shell_exports(NASTY) + "\n" + PROBE
    run = subprocess.run([bash(), "-c", script], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split("\0")[:-1] == list(NASTY.values())


def test_names_no_shell_can_assign_are_left_out():
    script = shell_exports({"ProgramFiles(ARM)": "x", "1BAD": "x", "GOOD": "y"})
    assert "GOOD" in script
    assert "ProgramFiles" not in script and "1BAD" not in script
