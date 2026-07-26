"""Regression tests for safe managed SQLite runtime handling on Windows.

The Desktop stage driver invokes ``install.ps1`` one stage per PowerShell
process. A compatible repaired venv must survive the venv stage, and the
subsequent dependencies stage must fail closed unless the managed runtime is
verified safe.
"""

from pathlib import Path

import pytest

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


@pytest.fixture(scope="module")
def source() -> str:
    return _INSTALL_PS1.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    """Return a PowerShell ``function <name> { ... }`` block."""
    start = source.index(f"function {name}")
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : index + 1]
    raise AssertionError(f"unterminated function body for {name}")


def test_install_venv_preserves_compatible_runtime_before_teardown(source: str) -> None:
    body = _function_body(source, "Install-Venv")
    probe_at = body.find("$existingPythonVersion")
    teardown_at = body.find('if (Test-Path "venv")')

    assert probe_at != -1, "Install-Venv must probe the existing venv interpreter"
    assert teardown_at != -1, "expected the existing venv teardown branch"
    assert probe_at < teardown_at, "the compatibility probe must run before teardown"
    assert '"$existingPythonVersion"' in body
    assert ").Trim() -eq $PythonVersion" in body
    assert '$preserveExistingVenv = $true' in body
    assert '$env:UV_PYTHON = $venvPythonExe' in body
    assert "if (-not $preserveExistingVenv)" in body


def test_windows_preserve_requires_verified_holder_sweep(source: str) -> None:
    """In-place dependency sync must not race an unverifiable venv holder."""
    body = _function_body(source, "Install-Venv")
    stop_at = body.index("Stop-Process")
    verify_at = body.index("$holderSweepVerified = $true", stop_at)
    preserve_gate_at = body.index(
        "$preserveExistingVenv -and -not $holderSweepVerified"
    )
    teardown_at = body.index("if (-not $preserveExistingVenv)")

    assert "$holderSweepVerified = $false" in body
    assert "-ErrorAction Stop" in body[stop_at:verify_at]
    assert stop_at < verify_at < preserve_gate_at < teardown_at
    assert "Cannot safely preserve virtual environment" in body


def test_dependencies_stage_repairs_managed_runtime(source: str) -> None:
    repair = _function_body(source, "Repair-ManagedRuntime")
    stage = _function_body(source, "Stage-Dependencies")

    assert "repair_vulnerable_runtime" in repair
    assert '$repairExitCode -ne 0' in repair
    assert "throw" in repair
    assert '$env:UV_PYTHON = $venvPythonExe' in repair
    assert stage.find("Install-Dependencies") < stage.find("Repair-ManagedRuntime")
