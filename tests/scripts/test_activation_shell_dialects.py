"""``scripts/_activation.sh`` is sourced by bash AND zsh, so it must answer the same in both.

``activate`` documents ``source ./activate`` for both shells, and
``hermes_activation_current`` decides with two globs over the stamp directory pm
writes beside the installed-state file. Neither glob matches anything when the
install recorded no input stamps, and the shells disagree about that: bash keeps
the pattern as a literal word that ``[ -f ]`` then rejects, while zsh's NOMATCH
-- on by default -- makes it a fatal error. Because the file is sourced rather
than executed, that error abandons the caller instead of letting it act on the
answer, so ``scripts/run-in-hermes-env`` never reaches its re-sync.

These tests pin the agreement rather than the implementation: the real library
is driven from each shell and expected to give the same verdict, for an install
with no input stamps and for a fully stamped one. The shells run with their rc
files disabled, since a developer's ``.zshrc`` may already turn NOMATCH off and
would hide exactly the defect under test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pm.environments import ACTIVATION_INPUTS, activation_input_mtimes, record_activation_inputs
from tests.pm.activation_support import bash, posix

pytestmark = pytest.mark.platforms("posix")

LIBRARY = Path(__file__).resolve().parents[2] / "scripts" / "_activation.sh"

# zsh ships with macOS and is a packaged extra on Linux; name the skip rather
# than dropping the parameter, so a host that cannot run it says so.
ZSH_MISSING = pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed")
SHELLS = [pytest.param("bash", id="bash"), pytest.param("zsh", id="zsh", marks=ZSH_MISSING)]


def _run(shell: str, script: str, root: Path, sentinel: Path) -> subprocess.CompletedProcess:
    """Source the real library from *shell* and run *script*, rc files disabled."""
    probe = root / "probe.sh"
    probe.write_text(script, encoding="utf-8")
    interpreter = bash() if shell == "bash" else str(shutil.which("zsh"))
    flags = ["--noprofile", "--norc"] if shell == "bash" else ["--no-rcs"]
    return subprocess.run(
        [interpreter, *flags, posix(probe)],
        capture_output=True, text=True, cwd=posix(root),
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": posix(root / "home"),
            "LIBRARY": posix(root / "scripts" / "_activation.sh"),
            "REPO": posix(root / "checkout"),
            "__HERMES_ACTIVATED": posix(sentinel),
        },
        timeout=60,
    )


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A checkout and the installed state whose ``inputs/`` the library globs."""
    (tmp_path / "scripts").mkdir()
    shutil.copy2(LIBRARY, tmp_path / "scripts" / "_activation.sh")
    (tmp_path / "checkout").mkdir()
    (tmp_path / "home").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "facts.json").touch()
    return tmp_path


def _markers_only(checkout: Path) -> Path:
    """What a successful install leaves when the checkout has none of pm's
    dependency inputs: the two dotfile markers and no input stamps at all
    (``activation_input_mtimes`` skips an input that is not a file)."""
    stamps = checkout / "state" / "inputs"
    record_activation_inputs(stamps, {}, checkout / "checkout", test_environment=True)
    assert not [entry for entry in stamps.iterdir() if not entry.name.startswith(".")]
    return stamps


def _fully_stamped(checkout: Path) -> Path:
    """What it leaves for a checkout that has them, one of them nested."""
    root = checkout / "checkout"
    for name in ACTIVATION_INPUTS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()
    stamps = checkout / "state" / "inputs"
    record_activation_inputs(stamps, activation_input_mtimes(root), root, test_environment=True)
    assert any(entry.is_dir() for entry in stamps.iterdir()), "one stamp must be nested"
    return stamps


@pytest.mark.parametrize("shell", SHELLS)
def test_an_install_with_no_input_stamps_reads_as_not_current(shell: str, checkout: Path):
    """Both globs match nothing, and the honest answer is "not current" -- a
    status the caller can act on by re-syncing, never a shell error that stops
    it from acting at all."""
    _markers_only(checkout)
    result = _run(
        shell,
        '. "$LIBRARY" || exit 90\n'
        'hermes_activation_current "$REPO" && exit 91\n'
        'echo CALLER_CONTINUED\n',
        checkout, checkout / "state" / "facts.json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "CALLER_CONTINUED"
    assert "no matches found" not in result.stderr


@pytest.mark.parametrize("shell", SHELLS)
def test_a_fully_stamped_install_still_reads_as_current(shell: str, checkout: Path):
    """The guard must not buy zsh's agreement by losing the answer: with every
    input stamped at its own mtime, including the nested one that only the
    second glob reaches, the environment is current in both shells."""
    _fully_stamped(checkout)
    result = _run(
        shell,
        '. "$LIBRARY" || exit 90\n'
        'hermes_activation_current "$REPO" || exit 91\n'
        'echo CURRENT\n',
        checkout, checkout / "state" / "facts.json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "CURRENT"


@pytest.mark.parametrize("shell", SHELLS)
def test_a_stamp_that_moved_reads_as_not_current(shell: str, checkout: Path):
    """The verdict the guard exists to preserve: an input whose mtime no longer
    equals its stamp means the install was never verified against what is on
    disk now, whichever direction a branch switch moved it."""
    _fully_stamped(checkout)
    moved = checkout / "checkout" / ACTIVATION_INPUTS[0]
    os.utime(moved, (0, 0))
    result = _run(
        shell,
        '. "$LIBRARY" || exit 90\n'
        'hermes_activation_current "$REPO" && exit 91\n'
        'echo CALLER_CONTINUED\n',
        checkout, checkout / "state" / "facts.json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "CALLER_CONTINUED"


@ZSH_MISSING
def test_the_zsh_glob_guard_is_not_left_behind_in_the_callers_shell(checkout: Path):
    """``activate`` sources this library into an interactive shell, so the guard
    has to be scoped: a user who sources it keeps the NOMATCH behaviour they
    had, rather than inheriting silent globs for the rest of the session."""
    _markers_only(checkout)
    result = _run(
        "zsh",
        'state() { printf "%s nomatch=%s nullglob=%s\\n" "$1" \\\n'
        '    "$([[ -o nomatch ]] && echo on || echo off)" \\\n'
        '    "$([[ -o nullglob ]] && echo on || echo off)"; }\n'
        'state before\n'
        '. "$LIBRARY" || exit 90\n'
        'hermes_activation_current "$REPO" && exit 91\n'
        'state after\n',
        checkout, checkout / "state" / "facts.json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == [
        "before", "nomatch=on", "nullglob=off",
        "after", "nomatch=on", "nullglob=off",
    ], result.stdout
