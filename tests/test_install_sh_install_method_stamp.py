"""Contract test: install.sh stamps the install method next to the code tree
($INSTALL_DIR), not into the shared $HERMES_HOME.

Background (shared-$HERMES_HOME bug)
------------------------------------
$HERMES_HOME is a data directory users frequently bind-mount into a Docker
gateway as well (``~/.hermes:/opt/data``). The published image stamps 'docker'
there on boot, so if install.sh had written its 'git' marker into the same
$HERMES_HOME the two installs would fight over one slot — and the container,
booting last, would win and wrongly make the host install look like 'docker'
(blocking ``hermes update``).

The fix: detect_install_method() reads a CODE-scoped stamp first, and the
installer writes ``git`` into $INSTALL_DIR (the git checkout, e.g.
``~/.hermes/hermes-agent``), which is unique to this install and immune to the
shared data dir.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_sh_stamps_code_tree_not_home() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    # Stamps the code tree.
    assert text.count('echo "git" > "$INSTALL_DIR/.install_method"') >= 1, (
        "install.sh must stamp $INSTALL_DIR/.install_method (code-scoped)"
    )

    # Never stamps the shared data dir.
    assert not re.search(r'>\s*"\$HERMES_HOME/\.install_method"', text), (
        "install.sh must not stamp $HERMES_HOME/.install_method — that data "
        "dir may be shared with a Docker gateway whose 'docker' stamp would "
        "clobber it and block host-side `hermes update`"
    )


def test_install_method_marker_is_ignored_by_git(tmp_path: Path) -> None:
    """The installer-created marker should not dirty the checkout."""
    if shutil.which("git") is None:
        pytest.skip("git executable is required for gitignore contract test")

    (tmp_path / ".gitignore").write_text(
        (REPO_ROOT / ".gitignore").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / ".install_method").write_text("git\n", encoding="utf-8")

    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", ".install_method"],
        cwd=tmp_path,
        check=False,
    )

    assert ignored.returncode == 0, (
        ".install_method must be ignored by Git so hermes update does not "
        "treat the installer-created marker as checkout dirt (#66189)"
    )
