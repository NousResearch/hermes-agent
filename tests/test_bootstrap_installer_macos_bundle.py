"""Regression coverage for the macOS bootstrap installer bundle."""

from __future__ import annotations

import json
import plistlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TAURI_DIR = REPO_ROOT / "apps" / "bootstrap-installer" / "src-tauri"
TAURI_CONFIG = TAURI_DIR / "tauri.conf.json"


def test_only_bootstrap_launcher_is_hidden_from_the_macos_dock() -> None:
    """The setup hand-off launcher is a dockless macOS helper app.

    The complementary Desktop package metadata contract lives in the Vitest
    suite because it reads a JavaScript-side ``package.json`` artifact.
    """
    setup_config = json.loads(TAURI_CONFIG.read_text(encoding="utf-8"))
    info_plist = TAURI_DIR / setup_config["bundle"]["macOS"]["infoPlist"]

    with info_plist.open("rb") as handle:
        bundle_metadata = plistlib.load(handle)

    assert setup_config["identifier"] == "com.nousresearch.hermes.setup"
    assert bundle_metadata["LSUIElement"] is True
