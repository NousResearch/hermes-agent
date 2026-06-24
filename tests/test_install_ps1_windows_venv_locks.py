"""Regression coverage for Windows venv lock handling in install.ps1.

The Windows installer previously tried to recreate ``venv`` by killing only
``hermes.exe`` and then running ``Remove-Item -Recurse -Force venv``. That does
not cover gateway/dashboard children running as ``python.exe`` or
``pythonw.exe`` from the venv, and Windows refuses to delete loaded ``.exe`` /
``.pyd`` files.

These tests are source-level because CI may not execute install.ps1 on Windows,
but they lock the important contracts around the PowerShell implementation.
"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_INSTALL_PS1 = _ROOT / "scripts" / "install.ps1"
_GITIGNORE = _ROOT / ".gitignore"


@pytest.fixture(scope="module")
def source() -> str:
    return _INSTALL_PS1.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    start = source.index(f"function {name}")
    brace = source.index("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : i + 1]
    raise AssertionError(f"unterminated function body for {name}")


def test_install_venv_moves_existing_venv_aside(source: str):
    body = _function_body(source, "Install-Venv")
    assert 'Move-WindowsVenvAside -VenvPath "venv"' in body
    assert 'Remove-Item -Recurse -Force "venv"' not in body


def test_windows_venv_stop_targets_processes_by_venv_path_and_descendants(
    source: str,
):
    body = _function_body(source, "Stop-WindowsHermesVenvProcesses")
    assert "Get-CimInstance Win32_Process" in body
    assert "Test-PathStartsWith -Path $exe -BasePath $venvFull" in body
    assert "$cmd.IndexOf($venvFull" in body
    assert "$targetPids.ContainsKey($parentPid)" in body
    assert "Stop-Process -Id $targetPid" in body


def test_windows_venv_move_uses_ignored_timestamped_directory(source: str):
    body = _function_body(source, "Move-WindowsVenvAside")
    assert "venv.pre-recreate-$stamp" in body
    assert "Move-Item -LiteralPath $VenvPath" in body
    assert "/venv.pre-recreate-*/" in _GITIGNORE.read_text(encoding="utf-8")
