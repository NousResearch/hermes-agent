"""Regression tests: install.ps1 must state the codec when reading native logs.

``_Invoke-NativeWithTimeout`` has cmd.exe redirect a child's stdout+stderr into
a log file.  No PowerShell decode happens on the way in, so the file holds the
child's own bytes and carries no BOM.  The children here are Node-based (npm,
npx, playwright) and write UTF-8.

Windows PowerShell 5.1's ``Get-Content`` sniffs a BOM and otherwise falls back
to the machine ANSI code page, so a bare read of one of those logs decodes
UTF-8 as cp932/cp936/cp1252.  Measured on ja-JP (ACP=932), PS 5.1.26100.9168::

    no BOM  + bare Get-Content        -> error: 綢輔ぃ 經、綢ォ縺瑚九九▽縺九...
    no BOM  + Get-Content -Encoding UTF8 -> error: ファイルが見つかりません
    with BOM + bare Get-Content       -> error: ファイルが見つかりません

The ``[Console]::OutputEncoding`` assignment near the top of install.ps1 does
not cover this: its own comment says it is "a DISPLAY-only fix".

Logs written through ``Tee-Object -FilePath`` are deliberately NOT in scope.
PS 5.1 writes those as UTF-16LE **with** a BOM (measured: ``FF FE 42 30 ...``
for a single "あ"), which Get-Content detects, so those reads are correct
as they stand.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


# The four reads whose file is written by a native child, not by PowerShell.
_NATIVE_LOG_READS = (
    r"Get-Content\s+-LiteralPath\s+\$logPath\s+-Tail\s+\$TailLines[^\r\n]*",
    r"Get-Content\s+\$path[^\r\n]*",
    r"Get-Content\s+\$logPath\s+-Raw[^\r\n]*",
    r"Get-Content\s+\$pwLog\s+-Raw[^\r\n]*",
)


def test_native_child_log_reads_state_the_codec() -> None:
    text = _install_ps1()
    for pattern in _NATIVE_LOG_READS:
        matches = re.findall(pattern, text)
        assert matches, f"expected a Get-Content matching {pattern!r}"
        for line in matches:
            assert "-Encoding" in line, (
                "a log written by a native child has no BOM, so this read "
                f"decodes with the machine ANSI code page: {line.strip()!r}"
            )


def test_cmd_redirect_helper_documents_why_the_codec_is_needed() -> None:
    """The helper that creates the BOM-less logs explains the constraint."""
    text = _install_ps1()
    assert re.search(
        r"cmd\.exe redirects the child's bytes[\s\S]{0,700}?"
        r"\$cmdLine = \"/d /s /c ",
        text,
    ), "_Invoke-NativeWithTimeout must say why its log needs an explicit codec"


def test_console_outputencoding_is_still_display_only() -> None:
    """Pin the premise: the console fix does not reach Get-Content.

    If a future change makes install.ps1 set a default read codec globally,
    this test should be revisited rather than silently kept passing.
    """
    text = _install_ps1()
    assert "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()" in text
    assert "DISPLAY-only fix" in text


def test_tee_object_reads_are_left_alone() -> None:
    """Tee-Object writes UTF-16LE with a BOM on 5.1; those reads are fine.

    Kept as an explicit statement of scope so the next person does not
    "finish the job" by adding -Encoding UTF8 to reads that would then
    misdecode a UTF-16 file.
    """
    text = _install_ps1()
    for var in ("$npmLog", "$buildLog"):
        assert re.search(
            r"Tee-Object\s+-FilePath\s+" + re.escape(var), text
        ), f"{var} is expected to be written through Tee-Object -FilePath"
