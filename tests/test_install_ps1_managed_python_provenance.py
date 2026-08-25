"""Behavioral regression for Hermes-managed Python provenance on Windows."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.install_ps1_fake_uv import compile_fake_uv


pytestmark = pytest.mark.windows_only

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


def test_venv_stage_rejects_third_party_python_and_uses_managed_path(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")

    hermes_home = tmp_path / "hermes-home"
    install_dir = tmp_path / "install"
    managed_root = install_dir / ".hermes-runtime" / "python"
    managed_python = managed_root / "cpython-3.11" / "python.exe"
    third_party = tmp_path / "KiCad" / "bin" / "python.exe"
    log = tmp_path / "uv.log"
    uv = hermes_home / "bin" / "uv.exe"
    uv.parent.mkdir(parents=True)
    install_dir.mkdir()
    managed_python.parent.mkdir(parents=True)
    managed_python.write_text("fake", encoding="ascii")
    third_party.parent.mkdir(parents=True)
    third_party.write_text("fake", encoding="ascii")
    compile_fake_uv(powershell, uv)

    env = {
        **dict(__import__("os").environ),
        "FAKE_UV_LOG": str(log),
        "FAKE_MANAGED_PYTHON": str(managed_python),
        "FAKE_THIRD_PARTY_PYTHON": str(third_party),
        "UV_PYTHON": str(third_party),
        "UV_NO_MANAGED_PYTHON": "1",
        "UV_SYSTEM_PYTHON": "1",
    }
    stdout_path = tmp_path / "installer.stdout"
    stderr_path = tmp_path / "installer.stderr"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        run = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(_INSTALL_PS1),
                "-Stage",
                "venv",
                "-HermesHome",
                str(hermes_home),
                "-InstallDir",
                str(install_dir),
            ],
            cwd=tmp_path,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )

    installer_stdout = stdout_path.read_text(encoding="utf-8")
    installer_stderr = stderr_path.read_text(encoding="utf-8")
    frames = [
        json.loads(line) for line in installer_stdout.splitlines() if line.startswith("{")
    ]
    assert run.returncode == 0, installer_stdout + installer_stderr
    assert frames[-1]["ok"] is True
    commands = log.read_text(encoding="utf-8").splitlines()
    assert any(
        command.startswith("python find 3.11") and "--managed-python" in command
        for command in commands
    )
    venv_command = next(command for command in commands if command.startswith("venv venv"))
    assert f"--python {managed_python}" in venv_command
    assert "--managed-python" in venv_command
    assert "--no-python-downloads" in venv_command
