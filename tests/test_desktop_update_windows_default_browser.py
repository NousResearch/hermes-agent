"""Regression tests for the Windows default-browser ProgId resolution.

Windows 11 25H2 (build 26200) writes the user's browser choice only to
``<scheme>\\UserChoiceLatest\\ProgId`` and no longer mirrors it into the legacy
``<scheme>\\UserChoice\\ProgId`` (write-protected by UCPD.sys). Readers that look
only at the legacy key resolve a browser the user abandoned -- and when that
browser is also uninstalled they resolve nothing at all.

The updater script is not executable on the Linux CI lane, so these tests lock
the source-level contract for ``windows.ps1``: the ProgId read must consult
``UserChoiceLatest`` before the legacy ``UserChoice`` for every scheme probed.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_UPDATE_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _source() -> str:
    # windows.ps1 is eol=crlf in .gitattributes; normalize so regex anchors match.
    return WINDOWS_UPDATE_PS1.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_default_browser_progid_reads_userchoicelatest_before_legacy_key() -> None:
    source = _source()
    helper = re.search(
        r"function Get-DefaultBrowserProgId \{(?P<body>.*?)\n\}\n\nfunction Get-DefaultBrowserExe",
        source,
        re.DOTALL,
    )
    assert helper, (
        "Expected a Get-DefaultBrowserProgId helper in "
        "scripts/desktop-update/windows.ps1; the default-browser resolution "
        "changed -- update this guard."
    )
    subkeys = re.findall(r'@?\("([^"]+)",\s*"([^"]+)"\)', helper.group("body"))
    flat = [name for pair in subkeys for name in pair]
    assert flat and flat[0] == "UserChoiceLatest" and "UserChoice" in flat, (
        "The ProgId read must try UserChoiceLatest (the only key 25H2 Settings "
        "writes) before the legacy UserChoice key; found order: "
        f"{flat}."
    )


def test_default_browser_exe_resolves_through_the_helper() -> None:
    source = _source()
    get_exe = re.search(
        r"function Get-DefaultBrowserExe \{(?P<body>.*?)\n\}\n\n",
        source,
        re.DOTALL,
    )
    assert get_exe, "Expected Get-DefaultBrowserExe in the Windows hand-off script."
    body = get_exe.group("body")
    assert "Get-DefaultBrowserProgId" in body, (
        "Get-DefaultBrowserExe must resolve its ProgId through "
        "Get-DefaultBrowserProgId so every caller gets the 25H2-aware read."
    )
    # The https scheme is probed first, http as fallback.
    schemes = re.findall(r'foreach \(\$proto in @\("(\w+)",\s*"(\w+)"\)\)', body)
    assert schemes and schemes[0] == ("https", "http"), (
        f"https must be probed before http; found {schemes}."
    )


def test_no_direct_legacy_only_reads_remain() -> None:
    source = _source()
    get_exe = re.search(
        r"function Get-DefaultBrowserExe \{(?P<body>.*?)\n\}\n\n",
        source,
        re.DOTALL,
    )
    assert get_exe, "Expected Get-DefaultBrowserExe in the Windows hand-off script."
    assert "UrlAssociations" not in get_exe.group("body"), (
        "Get-DefaultBrowserExe must not read UrlAssociations directly; the "
        "25H2-aware resolution lives in Get-DefaultBrowserProgId."
    )
