"""Windows Desktop hand-off resolves both supported project venv layouts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.windows_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VENV_PATH_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "venv-path.ps1"


def _resolve(root: Path) -> str:
    command = f'. "{VENV_PATH_PS1}"; Resolve-HermesVenvDir "{root}"'
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_resolver_falls_back_to_dot_venv_and_preserves_plain_venv_precedence(tmp_path):
    dot_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    dot_python.parent.mkdir(parents=True)
    dot_python.touch()
    assert Path(_resolve(tmp_path)) == tmp_path / ".venv"

    plain_python = tmp_path / "venv" / "Scripts" / "python.exe"
    plain_python.parent.mkdir(parents=True)
    plain_python.touch()
    assert Path(_resolve(tmp_path)) == tmp_path / "venv"