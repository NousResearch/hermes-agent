"""Behavioral tests for the install.sh launcher surviving a home directory change.

``setup_path()`` resolves ``$HERMES_BIN`` at install time. Baking that absolute
path into the shim breaks ``hermes`` outright when the user's home later moves —
a username change, a profile restored onto a new machine, or a migration — because
the shim keeps exec'ing a home that no longer exists.

The fix emits a ``$HOME``-relative exec target *only* for installs that live under
the installing user's home, so the path is expanded at launch time. Installs
outside ``$HOME`` (explicit ``--dir``/``$HERMES_INSTALL_DIR``, the root FHS layout
under ``/usr/local/lib``, and ``USE_VENV=false`` PATH installs) must keep their
absolute resolved path.

These tests execute the generated launcher rather than inspecting install.sh's
source text — see AGENTS.md "Never read source code in tests".
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

FAKE_BIN_BODY = """#!/usr/bin/env bash
echo "HERMES_LAUNCHED args=$*"
"""


def _extract_setup_path_shim_block() -> str:
    """Return the install.sh shim-write block used by setup_path()."""
    text = INSTALL_SH.read_text()
    match = re.search(
        r"(?P<block>mkdir -p \"\$command_link_dir\".*?chmod \+x \"\$command_link_dir/hermes\")",
        text,
        re.DOTALL,
    )
    assert match is not None, (
        "Could not locate the setup_path shim-write block in scripts/install.sh"
    )
    return match["block"]


def _write_fake_hermes(path: Path) -> None:
    """Create an executable stand-in for the pip-generated venv entry point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_BIN_BODY)
    path.chmod(0o755)


def _run_shim_block(*, home: Path, hermes_bin: Path, command_link_dir: Path) -> None:
    """Drive the real setup_path() shim-write block with a controlled env."""
    block = _extract_setup_path_shim_block()
    script = (
        "set -e\n"
        f"HOME={home!s}\n"
        f"HERMES_BIN={hermes_bin!s}\n"
        f"command_link_dir={command_link_dir!s}\n"
        f"{block}\n"
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"shim-write block failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def _run_shim(shim: Path, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Execute the generated launcher under the given HOME."""
    return subprocess.run(
        [str(shim), *args],
        capture_output=True,
        text=True,
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
    )


def test_launcher_still_runs_after_the_home_directory_moves(tmp_path: Path) -> None:
    """The regression: install under one home, then relocate it and launch.

    Mirrors a username change / profile restore — the whole home directory is
    installed at one path and later exists at another. The launcher must still
    exec the interpreter inside the *current* home.
    """
    old_home = tmp_path / "homes" / "alice"
    hermes_bin = old_home / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes"
    command_link_dir = old_home / ".local" / "bin"
    _write_fake_hermes(hermes_bin)

    _run_shim_block(home=old_home, hermes_bin=hermes_bin, command_link_dir=command_link_dir)

    # Sanity: the launcher works in its original home.
    shim = command_link_dir / "hermes"
    before = _run_shim(shim, old_home)
    assert before.returncode == 0, f"launcher broken pre-move: {before.stderr}"
    assert "HERMES_LAUNCHED" in before.stdout

    # The home directory moves (new username / restored profile / new machine).
    new_home = tmp_path / "homes" / "bob"
    old_home.rename(new_home)

    after = _run_shim(new_home / ".local" / "bin" / "hermes", new_home, "--version")
    assert after.returncode == 0, (
        "launcher must survive the home directory moving, but it failed:\n"
        f"stdout={after.stdout}\nstderr={after.stderr}"
    )
    assert "HERMES_LAUNCHED" in after.stdout
    assert "args=--version" in after.stdout, "launcher must forward args untouched"


def test_launcher_preserves_absolute_path_for_installs_outside_home(tmp_path: Path) -> None:
    """Non-default layouts must keep the install-time resolved path.

    Covers explicit --dir/$HERMES_INSTALL_DIR, the root FHS layout
    (/usr/local/lib/hermes-agent), and USE_VENV=false PATH installs: none of
    those live under $HOME, so rewriting them to a $HOME-relative path would
    point the launcher at a nonexistent executable.
    """
    home = tmp_path / "homes" / "alice"
    home.mkdir(parents=True)
    # Stands in for /usr/local/lib/hermes-agent — deliberately outside $HOME.
    hermes_bin = tmp_path / "opt" / "hermes-agent" / "venv" / "bin" / "hermes"
    command_link_dir = tmp_path / "usr_local_bin"
    _write_fake_hermes(hermes_bin)

    _run_shim_block(home=home, hermes_bin=hermes_bin, command_link_dir=command_link_dir)

    shim = command_link_dir / "hermes"
    assert str(hermes_bin) in shim.read_text(), (
        "an install outside $HOME must keep its absolute resolved path in the shim"
    )

    result = _run_shim(shim, home)
    assert result.returncode == 0, f"launcher broken: {result.stderr}"
    assert "HERMES_LAUNCHED" in result.stdout

    # A home change must not disturb an install that never lived under home.
    other_home = tmp_path / "homes" / "carol"
    other_home.mkdir(parents=True)
    moved = _run_shim(shim, other_home)
    assert moved.returncode == 0, (
        f"launcher outside $HOME must be unaffected by HOME changing: {moved.stderr}"
    )
    assert "HERMES_LAUNCHED" in moved.stdout
