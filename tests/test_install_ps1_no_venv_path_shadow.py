"""Regression: Windows install must not persist the whole venv Scripts dir.

A follow-up on issue #22054 showed the packaged Windows/Desktop install was
writing ``%LOCALAPPDATA%\\hermes\\hermes-agent\\venv\\Scripts`` into the user
PATH. That made bare ``python`` / ``pip`` resolve to Hermes's bundled 3.11
instead of the user's system Python.

The installer still needs a global ``hermes`` command, but that must come from
a dedicated shim directory (``$HermesHome\\bin``), not from the venv itself.
These tests lock that source-level contract.
"""

from pathlib import Path

import pytest

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


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


def test_set_path_variable_uses_hermes_bin_shim_dir(source: str):
    body = _function_body(source, "Set-PathVariable")
    assert 'Join-Path $HermesHome "bin"' in body, (
        "Set-PathVariable must publish a Hermes-managed shim directory, not the "
        "venv Scripts directory itself"
    )
    assert 'Join-Path $pathEntry "hermes.cmd"' in body, (
        "expected a dedicated hermes.cmd launcher under the Hermes-managed bin dir"
    )


def test_set_path_variable_removes_old_venv_scripts_entry(source: str):
    body = _function_body(source, "Set-PathVariable")
    assert "$entry -eq $venvScriptsDir" in body, (
        "Set-PathVariable must filter any persisted venv Scripts entry out of "
        "the user PATH during install/update"
    )
    assert 'Where-Object { $_ -and $_ -ne $venvScriptsDir }' in body, (
        "the current installer session must also drop the venv Scripts entry so "
        "bare python/pip stop resolving to Hermes immediately"
    )


def test_hermes_cmd_clears_python_env_before_exec(source: str):
    body = _function_body(source, "Set-PathVariable")
    assert 'set `"PYTHONPATH=`"' in body, (
        "the Windows hermes.cmd shim must clear PYTHONPATH before execing the venv launcher"
    )
    assert 'set `"PYTHONHOME=`"' in body, (
        "the Windows hermes.cmd shim must clear PYTHONHOME before execing the venv launcher"
    )
    assert '`"$InstallDir\\venv\\Scripts\\hermes.exe`" %*' in body, (
        "the shim should delegate directly to the venv hermes.exe without publishing "
        "the rest of venv Scripts on PATH"
    )
