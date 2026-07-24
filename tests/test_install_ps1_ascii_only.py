"""Regression: install.ps1 must stay pure ASCII.

Issues #66994 / #67000 reported the Windows GUI installer (Hermes-Setup.exe)
crashing before it did anything, with a cascade of PowerShell parser errors at
lines 1619 / 1770 ("Missing argument in parameter list", "A 'using' statement
must appear before any other statements in a script").

Root cause: ``scripts/install.ps1`` has no UTF-8 BOM, and a commit added two
non-ASCII characters *inside double-quoted string literals* (a bullet and an
em-dash). Windows PowerShell 5.1 -- which the bootstrap runs the cached script
under -- reads a BOM-less ``.ps1`` in the system ANSI code page (e.g. CP1252),
not UTF-8. The em-dash's UTF-8 tail byte (0x94) decodes to a "smart" close-quote
(U+201D), which the PowerShell tokenizer treats as a string delimiter. That
prematurely closes the string and desyncs the parser for the rest of the file,
surfacing as unrelated syntax errors far downstream.

Non-ASCII bytes inside ``#`` comments are harmless (the tokenizer skips a
comment to end-of-line regardless of what it contains), which is why the file
carried em-dashes in comments for months without breaking -- only a non-ASCII
byte in *code* (a string literal) triggers the desync.

This test is source-level because Linux CI cannot execute the PowerShell
installer. Keeping the whole file ASCII-only is the transport-independent
invariant: pure ASCII cannot be misdecoded under any code page, BOM or not, so
the bug class cannot recur -- in a comment or a string.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def test_install_ps1_is_pure_ascii() -> None:
    raw = INSTALL_PS1.read_bytes()

    offenders = []
    line_no = 1
    for byte in raw:
        if byte == 0x0A:
            line_no += 1
        elif byte >= 0x80:
            offenders.append(line_no)

    assert not offenders, (
        "scripts/install.ps1 must be pure ASCII so Windows PowerShell 5.1 "
        "(which reads a BOM-less .ps1 in the system ANSI code page, not UTF-8) "
        "cannot misdecode a byte into a stray quote and desync the parser "
        "(issues #66994 / #67000). Non-ASCII bytes found on line(s): "
        f"{sorted(set(offenders))}. Use ASCII equivalents (em-dash -> '--', "
        "bullet -> '-')."
    )
