"""Regression: Windows desktop updater must recognize Chromium family browsers.

The update shim (scripts/desktop-update/windows.ps1) displays its progress window
via ui.html in a chromeless browser window launched with --app --user-data-dir.
That requires a Chromium-family default browser (Chrome/Edge/Brave/Chromium/Vivaldi/Opera);
any other default (Firefox, Safari) gracefully degrades to the WinForms card.

Before this fix, Get-DefaultBrowserExe hardcoded two ProgId entries (ChromeHTML,
MSEdgeHTM) and returned $null for every other ProgId — including other Chromium
family browsers like Brave (BraveHTML) or Vivaldi (VivaldiHTM). That forced them
to the WinForms fallback, where the card became a frozen grey ghost window with
leaked PowerShell console (#105268).

The fix extends the recognition to all known stable Chromium ProgIds while
explicitly blocking channel builds (Beta/Dev/Canary) to fail closed and
avoid driving the wrong profile (#95549).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def test_chromium_family_progid_map_is_stable_complete() -> None:
    source = WINDOWS_PS1.read_text(encoding="utf-8").replace("\r\n", "\n")

    stable_progids = {
        "ChromeHTML": ("Google", "Chrome"),
        "MSEdgeHTM": ("Microsoft", "Edge"),
        "BraveHTML": ("BraveSoftware", "Brave-Browser"),
        "BraveOHTML": ("BraveSoftware", "Brave-Origin"),
        "ChromiumHTM": ("Chromium",),
        "VivaldiHTM": ("Vivaldi",),
        "OperaStableHTM": ("Opera",),
    }

    for progid_prefix, path_parts in stable_progids.items():
        # Each stable Chromium ProgId prefix must appear as a literal in a hashtable
        # entry paired with its product path segments.
        assert (
            f'prefix = "{progid_prefix}"' in source
        ), f"Desktop update script must recognize stable Chromium ProgId {progid_prefix} ({path_parts} install)."
        for segment in path_parts:
            assert (
                segment in source
            ), f"Desktop update script must carry the {segment} install tree segment for {progid_prefix}."


def test_channel_builds_explicitly_rejected() -> None:
    source = WINDOWS_PS1.read_text(encoding="utf-8").replace("\r\n", "\n")

    # Channel prefixes must be in the explicit-reject list so they fail closed
    # instead of resolving to the stable family (#95549).
    channel_progids = [
        "ChromeBHTML",
        "ChromeDHTML",
        "ChromeSSHTML",
        "ChromeCanaryHTML",
        "MSEdgeBHTML",
        "MSEdgeDHTML",
        "MSEdgeCHTML",
        "BraveBetaHTML",
        "BraveNightlyHTML",
        "BraveOBHTML",
        "BraveODHTML",
        "BraveOSHTM",
        "ChromiumBHTML",
        "ChromiumDHTML",
        "VivaldiBetaHTM",
        "VivaldiSnapshotHTM",
        "OperaBetaHTM",
        "OperaDevHTM",
    ]

    for chan in channel_progids:
        assert (
            f'"{chan}"' in source
        ), f"Desktop update script must explicitly reject channel ProgId {chan} to avoid driving the wrong profile."
