"""Regression tests for Windows desktop-update Unix timestamps.

PowerShell's ``Get-Date -UFormat %s`` is locale-sensitive. Under an ``es-ES``
culture it can produce a comma decimal separator, and parsing that string with
``InvariantCulture`` turns a ten-digit Unix timestamp plus fractional seconds
into a value too large for ``System.Int32``. The detached Windows updater must
use a locale-independent Unix timestamp API for both marker and result files.

The updater script is not executable on the Linux CI lane, so these tests lock
the source-level contract and reject the exact broken conversion.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_UPDATE_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def test_windows_update_uses_locale_independent_unix_seconds() -> None:
    source = WINDOWS_UPDATE_PS1.read_text(encoding="utf-8")
    safe_expression = "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()"
    unsafe_expression = "[int][double]::Parse((Get-Date -UFormat %s)"

    assert source.count(safe_expression) == 2, (
        "windows.ps1 must use DateTimeOffset Unix seconds for both the "
        "update marker and the finished result timestamp"
    )
    assert unsafe_expression not in source, (
        "windows.ps1 must not parse locale-sensitive Get-Date -UFormat %s "
        "output through System.Int32"
    )
