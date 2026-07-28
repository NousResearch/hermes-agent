"""Regression coverage for the macOS bootstrap installer bundle."""

from __future__ import annotations

import json
import plistlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TAURI_DIR = REPO_ROOT / "apps" / "bootstrap-installer" / "src-tauri"
TAURI_CONFIG = TAURI_DIR / "tauri.conf.json"
DESKTOP_PACKAGE = REPO_ROOT / "apps" / "desktop" / "package.json"


def test_only_bootstrap_launcher_is_hidden_from_the_macos_dock() -> None:
    """The setup hand-off stays dockless while the real desktop owns the icon."""
    setup_config = json.loads(TAURI_CONFIG.read_text(encoding="utf-8"))
    desktop_package = json.loads(DESKTOP_PACKAGE.read_text(encoding="utf-8"))
    info_plist = TAURI_DIR / setup_config["bundle"]["macOS"]["infoPlist"]

    with info_plist.open("rb") as handle:
        bundle_metadata = plistlib.load(handle)

    assert setup_config["identifier"] == "com.nousresearch.hermes.setup"
    assert desktop_package["build"]["appId"] == "com.nousresearch.hermes"
    assert bundle_metadata["LSUIElement"] is True
    assert "LSUIElement" not in desktop_package["build"]["mac"]["extendInfo"]
