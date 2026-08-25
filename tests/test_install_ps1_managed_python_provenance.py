"""Behavioral regression for Hermes-managed Python provenance on Windows."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.install_ps1_fake_uv import compile_fake_uv


pytestmark = pytest.mark.windows_only

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


def _run_venv_stage(
    powershell: str,
    tmp_path: Path,
    hermes_home: Path,
    install_dir: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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
        capture_output=True,
        text=True,
        check=False,
    )


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
    third_party.parent.mkdir(parents=True)
    third_party.write_text("fake", encoding="ascii")
    compile_fake_uv(powershell, uv)
    shutil.copy2(uv, managed_python)

    env = {
        **os.environ,
        "FAKE_UV_LOG": str(log),
        "FAKE_MANAGED_PYTHON": str(managed_python),
        "FAKE_THIRD_PARTY_PYTHON": str(third_party),
        "UV_PYTHON": str(third_party),
        "UV_NO_MANAGED_PYTHON": "1",
        "UV_SYSTEM_PYTHON": "1",
    }
    run = _run_venv_stage(powershell, tmp_path, hermes_home, install_dir, env)
    installer_stdout = run.stdout
    installer_stderr = run.stderr
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


def test_fallback_minor_is_reported_from_resolved_managed_interpreter(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")

    hermes_home = tmp_path / "hermes-home"
    install_dir = tmp_path / "install"
    managed_python = (
        install_dir / ".hermes-runtime" / "python" / "cpython-3.12" / "python.exe"
    )
    uv = hermes_home / "bin" / "uv.exe"
    uv.parent.mkdir(parents=True)
    managed_python.parent.mkdir(parents=True)
    install_dir.mkdir(exist_ok=True)
    compile_fake_uv(powershell, uv)
    shutil.copy2(uv, managed_python)
    env = {
        **os.environ,
        "FAKE_UV_LOG": str(tmp_path / "uv.log"),
        "FAKE_MANAGED_PYTHON": str(managed_python),
        "FAKE_MANAGED_PYTHON_VERSION": "3.12",
        "FAKE_PYTHON_VERSION": "Python 3.12.13",
    }

    run = _run_venv_stage(powershell, tmp_path, hermes_home, install_dir, env)

    assert run.returncode == 0, run.stdout + run.stderr
    assert "Creating virtual environment with Python 3.12" in run.stdout
    assert "Virtual environment ready (Python 3.12)" in run.stdout
    created_python = install_dir / "venv" / "Scripts" / "python.exe"
    version = subprocess.run(
        [str(created_python), "--version"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert version.stdout.strip() == "Python 3.12.13"


def test_venv_failure_fails_stage_and_restores_existing_environment(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")

    hermes_home = tmp_path / "hermes-home"
    install_dir = tmp_path / "install"
    managed_python = (
        install_dir / ".hermes-runtime" / "python" / "cpython-3.11" / "python.exe"
    )
    old_python = install_dir / "venv" / "Scripts" / "python.exe"
    uv = hermes_home / "bin" / "uv.exe"
    for directory in (uv.parent, managed_python.parent, old_python.parent):
        directory.mkdir(parents=True, exist_ok=True)
    old_python.write_text("previous environment", encoding="ascii")
    compile_fake_uv(powershell, uv)
    shutil.copy2(uv, managed_python)
    env = {
        **os.environ,
        "FAKE_UV_LOG": str(tmp_path / "uv.log"),
        "FAKE_MANAGED_PYTHON": str(managed_python),
        "FAKE_UV_VENV_EXIT": "37",
    }

    run = _run_venv_stage(powershell, tmp_path, hermes_home, install_dir, env)

    assert run.returncode != 0
    assert "Failed to create virtual environment (uv venv exited with 37)" in (
        run.stdout + run.stderr
    )
    assert old_python.read_text(encoding="ascii") == "previous environment"
    frames = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    assert frames[-1]["ok"] is False
