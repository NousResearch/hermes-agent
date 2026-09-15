"""Windows 11 25H2 writes the default browser to UserChoiceLatest, not UserChoice.

`Get-DefaultBrowserExe` in ``scripts/desktop-update/windows.ps1`` used to read
only the legacy ``UrlAssociations\\<scheme>\\UserChoice\\ProgId``. On Windows 11
25H2 / build 26200, Settings writes the real choice to the
``UserChoiceLatest\\ProgId`` subkey and no longer mirrors it into
``UserChoice``. The ``UserChoiceLatest`` parent value remains only as a
defensive compatibility probe.

This test is source-level because Linux/macOS CI cannot execute the PowerShell
hand-off. It guards the ProgId SOURCE order (nested Latest → parent Latest →
ASSOCSTR_PROGID → legacy UserChoice), source de-duplication, the
existence-gated fallthrough for a missing Chromium exe, and the forbidden
naive heuristics that return stale Edge on the reporter's machine.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _read() -> str:
    # windows.ps1 is eol=crlf in .gitattributes; normalize so function-body
    # anchors match regardless of the working-copy line endings.
    return WINDOWS_PS1.read_text(encoding="utf-8").replace("\r\n", "\n")


def _extract_function(source: str, name: str) -> str:
    match = re.search(
        rf"function {re.escape(name)}[^{{]*\{{(?P<body>.*?)\n\}}\n",
        source,
        re.DOTALL,
    )
    assert match, f"Expected function {name} in scripts/desktop-update/windows.ps1"
    return match.group(0)


def _browser_detection_scope() -> str:
    """Get-DefaultBrowserExe plus same-file helpers it may call."""
    source = _read()
    parts = [_extract_function(source, "Get-DefaultBrowserExe")]
    for name in (
        "Get-AssocQueryStringProgId",
        "Get-SchemeProgIds",
        "Resolve-BrowserExeFromProgId",
    ):
        try:
            parts.append(_extract_function(source, name))
        except AssertionError:
            continue
    return "\n".join(parts)


def _prog_id_source_scope() -> str:
    """The function that owns the ordered ProgId source probes."""
    source = _read()
    try:
        return _extract_function(source, "Get-SchemeProgIds")
    except AssertionError:
        return _extract_function(source, "Get-DefaultBrowserExe")


def _code_only(source: str) -> str:
    """Drop comments so path anchors only inspect executable source."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def _index(source: str, pattern: str) -> int:
    match = re.search(pattern, source)
    return match.start() if match else -1


def test_get_default_browser_exe_prefers_nested_then_parent_userchoice_latest() -> None:
    scope = _browser_detection_scope()
    source_scope = _code_only(_prog_id_source_scope())
    source = _read()

    nested_latest_idx = _index(
        source_scope, r"\\UserChoiceLatest\\ProgId\b"
    )
    parent_latest_idx = _index(
        source_scope, r"\\UserChoiceLatest(?!\\ProgId)\b"
    )
    assoc_idx = source_scope.find("Get-AssocQueryStringProgId")
    legacy_idx = _index(source_scope, r"\\UserChoice(?!Latest)\b")

    assert nested_latest_idx != -1, (
        "Get-DefaultBrowserExe must first read the ProgId value from the "
        "UrlAssociations\\<scheme>\\UserChoiceLatest\\ProgId subkey used by "
        "Windows 11 25H2 (#108051)."
    )
    assert parent_latest_idx != -1, (
        "Get-DefaultBrowserExe must retain the UserChoiceLatest parent-value "
        "compatibility fallback after the nested subkey."
    )
    assert assoc_idx != -1
    assert legacy_idx != -1, (
        "Get-DefaultBrowserExe must keep the legacy UserChoice fallback for "
        "Win10 and older Windows 11 builds."
    )
    assert nested_latest_idx < parent_latest_idx < assoc_idx < legacy_idx, (
        "ProgId sources must remain ordered as nested UserChoiceLatest, parent "
        "UserChoiceLatest, ASSOCSTR_PROGID, then legacy UserChoice."
    )
    assert scope.count("-notcontains") >= 4, (
        "Each ProgId source must retain ordered de-duplication before browser "
        "resolution."
    )
    # Five fail-open boundaries: nested latest, parent latest, Assoc API /
    # Add-Type, legacy UserChoice, and the selected ProgId's command lookup.
    assert scope.count("catch {}") == 5, (
        "All five browser-resolution probes must remain fail-open so a failed "
        "source or path lookup does not abort the Desktop update hand-off."
    )

    assoc_scope = scope + "\n" + source
    has_assoc_progid = (
        "ASSOCSTR_PROGID" in assoc_scope
        or re.search(r"AssocQueryString\w*\s*\([^)]*PROGID", assoc_scope, re.I)
        or ("AssocQueryString" in assoc_scope and "ProgId" in assoc_scope)
    )
    assert has_assoc_progid, (
        "Get-DefaultBrowserExe (or a helper in windows.ps1) must query "
        "AssocQueryStringW with ASSOCSTR_PROGID as the second ProgId source. "
        "Do not use ASSOCSTR_EXECUTABLE -- that returns 0x80070483 on the "
        "reporter's 25H2 machine."
    )

    assert "Test-Path" in scope, (
        "After resolving an exe path, Test-Path must gate the return so a "
        "stale uninstalled Edge ProgId falls through to the next ProgId source."
    )
    assert "ChromeHTML" in scope and "MSEdgeHTM" in scope, (
        "ChromeHTML → Chrome and MSEdgeHTM → Edge family mappings must remain."
    )


def test_get_default_browser_exe_rejects_forbidden_default_sources() -> None:
    scope = _browser_detection_scope()

    assert "ASSOCSTR_EXECUTABLE" not in scope, (
        "Do not use AssocQueryStringW(ASSOCSTR_EXECUTABLE) as a default-browser "
        "source; it fails with 0x80070483 on Windows 11 25H2."
    )
    assert not re.search(
        r"(HKCR:|HKEY_CLASSES_ROOT)\\https?\\shell\\open\\command",
        scope,
        re.IGNORECASE,
    ), (
        "Do not use HKCR\\https\\shell\\open\\command (or http) as a ProgId / "
        "default source; it returns stale msedge. HKCR\\$progId\\shell\\open\\command "
        "after a correctly resolved ProgId is the existing exe-path lookup and must stay."
    )
    assert re.search(
        r"HKEY_CLASSES_ROOT\\\$progId\\shell\\open\\command",
        scope,
    ), (
        "Keep the existing HKCR\\$progId\\shell\\open\\command exe-path lookup "
        "after a correctly resolved ProgId."
    )
