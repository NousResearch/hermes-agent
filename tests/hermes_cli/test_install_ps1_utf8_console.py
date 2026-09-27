"""install.ps1 must decode child-process output as UTF-8 on Windows (#124526).

Windows PowerShell 5.1 decodes captured native stdout with the legacy OEM
code page, so a uv-resolved Python path under a profile like
``C:\\Users\\Balázs\\...`` arrived mojibake (UTF-8 ``á`` = 0xC3 0xA1 read as
CP437 = ``├í``) and the next ``& $bootPy`` failed with "term not recognized".
The contract below is static, like test_install_ps1_bootstrap_skips_the_same_
msys_links_as_pm: the installer script is exercised for real only on the
windows platform (see tests/ci/test_classify_changes.py).
"""

import re
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[2] / "scripts" / "install.ps1"


def _text() -> str:
    return INSTALLER.read_text(encoding="utf-8")


def test_entry_forces_utf8_console_decode_before_any_native_capture():
    """The entry prologue sets the console decode tables to UTF-8.

    Three orderings matter:
    * AFTER the dot-source guard's ``return`` -- a dot-sourced test host must
      not have its own console reconfigured;
    * BEFORE ``Initialize-ResolvedPaths`` and every switch dispatch, so no
      native capture (uv/git path resolution) can run first;
    * exception-guarded -- a host without an attached console (redirected CI)
      rejects the console setter, and an encoding preference must never fail
      an install.
    """
    text = _text()

    guard = re.search(r"^if \(\$script:IsDotSourced\) \{", text, re.M)
    assert guard, "dot-source guard moved; update this test's anchors"

    entry_call = re.search(r"^Initialize-ResolvedPaths", text, re.M)
    assert entry_call, "entry prologue call moved; update this test's anchors"

    fix = re.search(
        r"\[Console\]::OutputEncoding\s*=\s*New-Object System\.Text\.UTF8Encoding",
        text,
    )
    assert fix, (
        "install.ps1 no longer forces [Console]::OutputEncoding to UTF-8; "
        "non-ASCII profile paths captured from uv/git regress to OEM mojibake "
        "(#124526)"
    )

    assert guard.start() < fix.start(), (
        "UTF-8 fix must live after the dot-source guard (dot-sourced hosts "
        "keep their own console settings)"
    )
    assert fix.start() < entry_call.start(), (
        "UTF-8 fix must precede the entry prologue: path resolution captures "
        "native output from that point on"
    )

    # The pipe direction (PowerShell -> native stdin) is ASCII on 5.1 by
    # default; same block, same guarantee.
    pipe = re.search(r"\$OutputEncoding\s*=\s*New-Object System\.Text\.UTF8Encoding", text)
    assert pipe and abs(pipe.start() - fix.start()) < 400, (
        "$OutputEncoding should be set in the same guarded block"
    )

    block = text[max(0, fix.start() - 200) : fix.start() + 400]
    assert "try {" in block and "catch" in block, (
        "the encoding assignment must be exception-guarded (console-less hosts)"
    )
