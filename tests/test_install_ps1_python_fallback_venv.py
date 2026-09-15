"""Regression: the Windows installer must honor its Python fallback at venv time.

A user on Windows 11 reported (#50769) that the installer correctly detected a
Python 3.12 fallback when 3.11 was absent::

    [OK] Found fallback: Python 3.12.8
    ...
    -> Creating virtual environment with Python 3.11...

and then failed with ``Failed to create virtual environment (uv venv exited
with 2)``.

Root cause: ``Test-Python`` records the fallback via an in-memory
``$script:PythonVersion = $fallbackVer`` mutation, but under Hermes-Setup.exe
each ``-Stage NAME`` runs in its *own* fresh ``powershell.exe`` process.  The
``venv`` stage therefore starts with ``$PythonVersion`` back at its ``"3.11"``
default, so ``uv venv venv --python 3.11`` runs on a machine that has no 3.11.

The fix re-resolves the interpreter inside ``Install-Venv`` (via the
cross-process-safe ``Resolve-AvailablePythonVersion`` helper) before creating
the venv, so the venv stage uses whatever interpreter is actually present.
These tests lock that contract at the source level (the script only runs on
Windows, so there's no runner to execute it on Linux CI).
"""

import re
from pathlib import Path

import pytest

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


@pytest.fixture(scope="module")
def source() -> str:
    return _INSTALL_PS1.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    """Return the text of a PowerShell ``function <name> { ... }`` block."""
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


def test_resolver_helper_is_defined(source: str):
    """A cross-process-safe Python-version resolver must exist."""
    assert "function Resolve-AvailablePythonVersion" in source, (
        "expected a Resolve-AvailablePythonVersion helper that re-resolves the "
        "interpreter independently of the in-memory $script:PythonVersion mutation"
    )


def test_install_venv_reresolves_before_creating_venv(source: str):
    """Install-Venv must re-resolve the interpreter BEFORE `uv venv`.

    Otherwise the venv stage's fresh process trusts the stale "3.11" default
    and fails on machines where only the fallback (e.g. 3.12) is installed.
    """
    body = _function_body(source, "Install-Venv")
    resolve_at = body.find("Resolve-AvailablePythonVersion")
    assert resolve_at != -1, (
        "Install-Venv must call Resolve-AvailablePythonVersion so the venv "
        "stage doesn't trust the stale $PythonVersion default across processes"
    )
    create_at = body.find("Creating virtual environment with Python")
    assert create_at != -1, "expected the venv-creation log line in Install-Venv"
    assert resolve_at < create_at, (
        "the interpreter must be re-resolved BEFORE the 'Creating virtual "
        "environment' step (and before `uv venv` runs)"
    )


def test_fallback_list_is_single_source_of_truth(source: str):
    """The fallback versions live in one shared constant, used by both paths.

    A drifting second copy of the list is how detection and venv creation
    disagree in the first place.
    """
    assert re.search(r"\$PythonFallbackVersions\s*=", source), (
        "expected a shared $PythonFallbackVersions constant"
    )
    # Test-Python's fallback loop must iterate the shared constant, not an
    # inline literal list.
    test_python = _function_body(source, "Test-Python")
    assert "foreach ($fallbackVer in $PythonFallbackVersions)" in test_python, (
        "Test-Python must iterate the shared $PythonFallbackVersions constant"
    )
    # The resolver must seed its candidate list from the same constant.
    resolver = _function_body(source, "Resolve-AvailablePythonVersion")
    assert "$PythonFallbackVersions" in resolver, (
        "Resolve-AvailablePythonVersion must reuse the shared fallback constant"
    )


def test_resolver_prefers_requested_version_then_fallbacks(source: str):
    """The resolver tries the requested version first, then the fallbacks."""
    resolver = _function_body(source, "Resolve-AvailablePythonVersion")
    assert "@($PythonVersion) + $PythonFallbackVersions" in resolver, (
        "resolver candidate order must be: requested $PythonVersion first, "
        "then the shared fallbacks"
    )
    assert "uv" in resolver and "python find" in resolver, (
        "resolver must probe availability via `uv python find`"
    )
