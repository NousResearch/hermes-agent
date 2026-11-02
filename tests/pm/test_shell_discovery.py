"""The local transport can use Git Bash without a PATH entry."""
import os
from pathlib import Path
import subprocess

import pytest


_GIT_BASH = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe"


@pytest.mark.platforms("windows")
@pytest.mark.skipif(not _GIT_BASH.is_file(), reason="a conventional Git for Windows installation is required")
def test_local_transport_finds_conventional_bash_without_path(monkeypatch, tmp_path):
    from tools.environments.local import _find_bash

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    conventional = _GIT_BASH
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    empty_bin = tmp_path / "empty-bin"
    alias_bin = tmp_path / "WindowsApps"
    empty_bin.mkdir()
    alias_bin.mkdir()
    (alias_bin / "bash.exe").write_bytes(b"unusable execution alias")
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")

    monkeypatch.setenv("PATH", str(empty_bin))
    selected = _find_bash()
    assert Path(selected) == conventional
    child = subprocess.run(
        [selected, "--noprofile", "--norc", "-c", "printf shell-ready"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10, check=False,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout == "shell-ready"

    monkeypatch.setenv("PATH", str(alias_bin))
    selected = _find_bash()
    assert Path(selected) == conventional
    child = subprocess.run(
        [selected, "--noprofile", "--norc", "-c", "printf shell-ready"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10, check=False,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout == "shell-ready"
