"""Behavioral coverage for Bash PATH setup on macOS."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.macos_only
@pytest.mark.parametrize("existing_bashrc", [False, True])
def test_path_stage_creates_login_bash_profile(
    tmp_path: Path, existing_bashrc: bool
) -> None:
    """A macOS Bash login must find Hermes with or without an existing bashrc."""
    home = tmp_path / "home"
    install_dir = tmp_path / "hermes-agent"
    venv_bin = install_dir / "venv" / "bin"
    home.mkdir()
    venv_bin.mkdir(parents=True)
    if existing_bashrc:
        (home / ".bashrc").write_text("# existing config\n", encoding="utf-8")

    _make_executable(venv_bin / "python", "#!/bin/sh\nexit 0\n")
    (install_dir / "hermes").write_text("# launcher target\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "HERMES_HOME": str(home / ".hermes"),
            "HERMES_INSTALL_DIR": str(install_dir),
            "SHELL": "/bin/bash",
        }
    )
    local_bin = str(home / ".local" / "bin")
    env["PATH"] = os.pathsep.join(
        entry for entry in env.get("PATH", "").split(os.pathsep) if entry != local_bin
    )

    completed = subprocess.run(
        [
            "/bin/bash",
            str(INSTALL_SH),
            "--stage",
            "path",
            "--non-interactive",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    bash_profile = home / ".bash_profile"
    assert bash_profile.exists()
    assert 'export PATH="$HOME/.local/bin:$PATH"' in bash_profile.read_text(
        encoding="utf-8"
    )

    resolved = subprocess.run(
        ["/bin/bash", "-lc", "command -v hermes"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert resolved.returncode == 0, resolved.stderr
    assert resolved.stdout.strip() == str(home / ".local" / "bin" / "hermes")
