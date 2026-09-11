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

    assert source.count(safe_expression) >= 2, (
        "windows.ps1 must use DateTimeOffset Unix seconds for both the "
        "update marker and the finished result timestamp"
    )
    assert unsafe_expression not in source, (
        "windows.ps1 must not parse locale-sensitive Get-Date -UFormat %s "
        "output through System.Int32"
    )


def _function_code(name: str) -> str:
    """One function's CODE, comments stripped: the comments here discuss the
    very constructs these assertions forbid."""
    source = WINDOWS_UPDATE_PS1.read_text(encoding="utf-8")
    start = source.index(f"function {name}")
    block = source[start:source.index("\nfunction ", start + 1)]
    return "\n".join(line.split("#", 1)[0] for line in block.splitlines())


def test_the_handoff_ack_is_published_by_rename_not_by_copy() -> None:
    """The ack's whole job is to be readable by a Desktop polling for it.
    ``File.Copy(source, dest, overwrite)`` truncates the destination and
    rewrites it in place, so a reader that opens between those two steps gets an
    empty or partial body -- and the reader's rule is EXACTLY three lines, so a
    torn ack reads as "no ack" and the Desktop refuses a hand-off that really
    did happen. A rename publishes the whole file or nothing.

    Source-level because the race is not reachable from a test: the window is
    the microseconds between truncate and write inside one Win32 call.
    """
    published = _function_code("Write-HandoffAck")

    assert "[System.IO.File]::Copy(" not in published, (
        "the ack must not be published with File.Copy: it truncates the "
        "destination before rewriting it, so a partial ack is readable"
    )
    assert "MoveFileEx" in published and "[System.IO.File]::Move(" in published, (
        "the ack must be published by rename -- MoveFileEx with "
        "MOVEFILE_REPLACE_EXISTING, falling back to File.Move"
    )
    assert "Move-Item" not in published, (
        "Move-Item moves a file INSIDE a destination that is a directory and "
        "reports success, which would turn a failed ack into a silent pass"
    )
