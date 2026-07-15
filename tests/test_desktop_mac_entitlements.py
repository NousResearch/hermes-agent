"""Regression for #37718: macOS microphone entitlement must be inherited.

Hermes Desktop signs with ``hardenedRuntime: true`` and points electron-builder
at two entitlement files (see ``apps/desktop/package.json``):

* ``entitlements`` → ``electron/entitlements.mac.plist`` (the main app), and
* ``entitlementsInherit`` → ``electron/entitlements.mac.inherit.plist`` (the
  Electron Helper / Setup processes).

Under the hardened runtime, the process that actually opens the microphone is a
Helper, which inherits the *inherit* plist. ``com.apple.security.device.audio-input``
lived only in the main plist, so macOS' TCC layer refused the microphone with::

    Prompting policy for hardened runtime; service: kTCCServiceMicrophone
    requires entitlement com.apple.security.device.audio-input but it is missing

and never showed the permission prompt. These tests pin that every device
entitlement granted to the main app is also granted to the inherited helpers.
"""

from __future__ import annotations

import json
import plistlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
ELECTRON_DIR = REPO_ROOT / "apps" / "desktop" / "electron"
MAIN_PLIST = ELECTRON_DIR / "entitlements.mac.plist"
INHERIT_PLIST = ELECTRON_DIR / "entitlements.mac.inherit.plist"
DESKTOP_PACKAGE_JSON = REPO_ROOT / "apps" / "desktop" / "package.json"

DEVICE_PREFIX = "com.apple.security.device."


def _load(plist: Path) -> dict:
    assert plist.is_file(), f"missing entitlements file: {plist}"
    with plist.open("rb") as fh:
        return plistlib.load(fh)


def test_inherit_plist_grants_microphone() -> None:
    """The helper-inherited plist must grant audio-input (regression #37718)."""
    inherit = _load(INHERIT_PLIST)
    assert inherit.get("com.apple.security.device.audio-input") is True, (
        "entitlements.mac.inherit.plist must grant "
        "`com.apple.security.device.audio-input`; without it the hardened-runtime "
        "Helper process is denied the microphone and no TCC prompt appears (#37718)."
    )


def test_device_entitlements_are_inherited() -> None:
    """Every device.* entitlement on the main app must also be inherited."""
    main = _load(MAIN_PLIST)
    inherit = _load(INHERIT_PLIST)

    main_device = {
        key: val
        for key, val in main.items()
        if key.startswith(DEVICE_PREFIX) and val is True
    }
    missing = [key for key in main_device if inherit.get(key) is not True]
    assert not missing, (
        "Device entitlements present in entitlements.mac.plist but missing from "
        f"entitlements.mac.inherit.plist: {missing}. Helper/Setup processes inherit "
        "the latter under hardenedRuntime, so any device access the app needs must "
        "be listed in both (#37718)."
    )


def test_desktop_package_declares_calendar_privacy_usage() -> None:
    """Packaged macOS builds must explain Calendar access to TCC."""
    package = json.loads(DESKTOP_PACKAGE_JSON.read_text(encoding="utf-8"))
    extend_info = package["build"]["mac"]["extendInfo"]
    required = (
        "NSCalendarsFullAccessUsageDescription",
        "NSCalendarsUsageDescription",
    )
    missing = [
        key
        for key in required
        if not isinstance(extend_info.get(key), str) or not extend_info[key].strip()
    ]
    assert not missing, f"missing non-empty macOS Calendar privacy descriptions: {missing}"


@pytest.mark.parametrize("plist", [MAIN_PLIST, INHERIT_PLIST])
def test_entitlement_files_are_valid_plists(plist: Path) -> None:
    """Both entitlement files must remain well-formed plist dictionaries."""
    data = _load(plist)
    assert isinstance(data, dict) and data, f"{plist.name} should be a non-empty dict"
