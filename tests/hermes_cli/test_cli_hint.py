"""Owner suite for `hermes_cli.cli_hint`: a printed remedy must PASTE.

`hint_arg` is the single place that knows the escaping rules, so it is the
single place those rules have to be pinned. A call-site test cannot do it: a
test that asserts `<site>(p) == hint_arg("--flag", p)` pins DELEGATION -- both
sides move together when the predicate changes -- so the predicate's own
correctness is unguarded by construction.

THE PASTE SIMULATOR IS A REAL SHELL, and that is load-bearing. Deciding safety
with `shlex.split` and then SIMULATING the paste with `shlex.split` is the same
function on both sides: `$HOME`, `` `id` `` and `star*glob` are one clean word
to `shlex` and are expanded, EXECUTED or globbed by a real `/bin/bash`. Here the
printed string is handed to `/bin/bash`, the words bash actually produces are
fed to a real `argparse` parser, and the bound value must equal the token that
was printed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import pytest

from hermes_cli.cli_hint import hint_arg

BASH = shutil.which("bash")

requires_bash = pytest.mark.skipif(
    BASH is None or sys.platform.startswith("win"),
    reason="the paste oracle needs a real POSIX shell",
)

# Tokens a real filesystem path or repository directory name can hold. Each of
# these creates fine as a directory on macOS/Linux, so each is reachable.
HOSTILE = [
    # argparse layer: a value beginning with '-'
    "-leading-dash", "--double-dash", "-",
    # shell layer: word splitting
    "qa output", "two  spaces", "trailing ", "\ttab",
    # shell layer: expansion / substitution -- the printed remedy would EXECUTE
    "$HOME", "`id`", "$(id)", "a$b", "${x}",
    # shell layer: control operators
    "semi;colon", "amp&sand", "pipe|line", "paren(th)", "a&&b",
    # shell layer: brace expansion and globbing (the SILENT, CWD-dependent half)
    "brace{a,b}", "star*glob", "q?mark", "brack[et]",
    # shell layer: quoting
    "quo'te", 'dou"ble', "back\\slash",
    # shell layer: tilde expansion
    "~scratch", "~",
]
PLAIN = ["plain", "with/slash", "repo@v2", "a+b", "under_score", "repo.name", "CAPS"]


def _bash_words(printed, cwd):
    """The argv a REAL bash produces for `printed`, or None if bash refuses.

    NUL-delimited so a word containing a newline survives the round trip.
    """
    proc = subprocess.run(
        [str(BASH), "-c", 'printf "%s\\0" ' + printed],
        capture_output=True, cwd=cwd,
    )
    if proc.returncode != 0:
        return None
    out = proc.stdout.decode("utf-8", errors="replace")
    return out.split("\0")[:-1] if out else []


@pytest.fixture
def hostile_cwd(tmp_path):
    """A CWD holding names that MATCH the glob tokens above.

    Globbing is the silent half of the bug: bash leaves `star*glob` literal
    only when nothing matches. With `starXglob` on disk the same printed hint
    binds a different value and nothing errors. The oracle must run somewhere
    the glob can actually hit, or it cannot see that failure mode.
    """
    for name in ("starXglob", "qAmark", "bracke", "brace"):
        (tmp_path / name).mkdir()
    return str(tmp_path)


@requires_bash
@pytest.mark.parametrize("token", HOSTILE + PLAIN)
def test_hint_arg_survives_a_real_shell_then_argparse(token, hostile_cwd):
    """The helper's output round-trips through /bin/bash and a real parser."""
    parser = argparse.ArgumentParser(prog="prog", add_help=False)
    parser.add_argument("--thing")

    printed = hint_arg("--thing", token)
    words = _bash_words(printed, hostile_cwd)

    assert words is not None, f"bash refused the printed hint {printed!r}"
    args = parser.parse_args(words)
    assert args.thing == token, f"printed {printed!r} bound {args.thing!r}"


@requires_bash
def test_the_bare_form_really_is_dangerous(hostile_cwd):
    """The oracle's premise, measured rather than assumed.

    Without this, the round-trip test above could be describing whatever the
    helper happens to emit. This asserts what a REAL bash does with the BARE
    spelling `hint_arg` exists to avoid -- so a predicate that calls these
    tokens safe-to-print-bare is known to be wrong, not merely different.
    """
    assert _bash_words("--thing $HOME", hostile_cwd) == ["--thing", os.environ["HOME"]]
    # backticks EXECUTE: the printed remedy would run `id`.
    executed = _bash_words("--thing `id`", hostile_cwd)
    assert executed is not None and executed[1].startswith("uid="), executed
    # a control operator makes bash run a second command, not pass a word.
    assert _bash_words("--thing paren(th)", hostile_cwd) is None


@pytest.mark.parametrize("token", PLAIN)
def test_the_readable_form_is_kept_when_no_escaping_is_needed(token):
    """Quoting everything would be safe but unreadable; plain tokens stay plain.

    This is the over-fix guard: a helper that always returned the quoted
    `'--flag=value'` form would pass every round-trip test above while making
    the common message worse.
    """
    assert hint_arg("--thing", token) == f"--thing {token}"
