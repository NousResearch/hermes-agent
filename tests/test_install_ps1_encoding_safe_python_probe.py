"""Regression tests for #60129: corp-managed PowerShell encoding restriction.

On corp-managed Windows hosts (group policy constraining console encoding),
any ``2>&1`` / ``2>$null`` stderr redirection on a python.exe call can abort
with ``StandardErrorEncoding is only supported when standard error is
redirected`` before python even starts.  That turned a fully successful
dependency install into a false-negative at the baseline-import gate (and
silently broke the pyproject.toml extras parse), so the installer never
reached the path / config-templates / platform-sdks stages.

``scripts/install.ps1`` therefore routes every python probe through the
``Invoke-PythonEncodingSafe`` helper, which uses ``Start-Process`` with
explicit ``-RedirectStandardOutput`` / ``-RedirectStandardError`` file
redirects — PowerShell's native-command stream plumbing (the thing that sets
``StandardErrorEncoding``) is never involved.

These tests are source-level because Linux CI cannot execute the PowerShell
installer.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_encoding_safe_helper_uses_start_process_file_redirects() -> None:
    text = _install_ps1()
    m = re.search(
        r"function Invoke-PythonEncodingSafe \{(?P<body>[\s\S]*?)^\}",
        text,
        re.MULTILINE,
    )
    assert m is not None, (
        "install.ps1 must define Invoke-PythonEncodingSafe so python probes "
        "avoid PowerShell's stderr stream plumbing (#60129)"
    )
    body = m.group("body")
    assert "Start-Process" in body
    assert "-RedirectStandardOutput" in body
    assert "-RedirectStandardError" in body
    # Call sites keep their `$LASTEXITCODE -eq 0` checks, so the helper must
    # mirror the child's exit code into the automatic variable.
    assert "$global:LASTEXITCODE = $proc.ExitCode" in body


def test_baseline_import_gate_routes_through_encoding_safe_helper() -> None:
    text = _install_ps1()
    assert re.search(
        r"Invoke-PythonEncodingSafe -PythonExe \$venvPython "
        r'-Code "import dotenv, openai, rich, prompt_toolkit"',
        text,
    ), "the baseline-import probe must not use PowerShell stderr redirection"


def test_pyproject_extras_parse_routes_through_encoding_safe_helper() -> None:
    text = _install_ps1()
    assert re.search(
        r"Invoke-PythonEncodingSafe -PythonExe \$pythonExeForParse -Code @\"",
        text,
    ), "the pyproject.toml extras parse must not use PowerShell stderr redirection"


def test_no_python_invocation_merges_or_discards_stderr_via_powershell() -> None:
    """No direct python.exe call may still use ``2>&1`` or ``2>$null``.

    Any such line re-introduces the corp-managed-host launch failure from
    #60129 at that stage.
    """
    python_vars = ("$venvPython", "$pythonExe", "$pythonExeForParse")
    offenders = [
        f"line {lineno}: {line.strip()}"
        for lineno, line in enumerate(_install_ps1().splitlines(), start=1)
        if line.lstrip().startswith("&")
        and any(v in line for v in python_vars)
        and ("2>&1" in line or "2>$null" in line)
    ]
    assert not offenders, (
        "python invocations must go through Invoke-PythonEncodingSafe instead "
        "of PowerShell stderr redirection (#60129):\n" + "\n".join(offenders)
    )
