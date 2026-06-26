"""Packaging contract for the bundled Electron main process."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DESKTOP_DIR = REPO_ROOT / "apps" / "desktop"
BUNDLE_SCRIPT = DESKTOP_DIR / "scripts" / "bundle-electron-main.mjs"
DESKTOP_PKG = DESKTOP_DIR / "package.json"
PACKAGED_MAIN_VALIDATION = DESKTOP_DIR / "scripts" / "packaged-main-validation.cjs"
TEST_DESKTOP = DESKTOP_DIR / "scripts" / "test-desktop.mjs"
NIX_DESKTOP = REPO_ROOT / "nix" / "desktop.nix"


def _desktop_pkg() -> dict:
    assert DESKTOP_PKG.is_file(), f"missing {DESKTOP_PKG}"
    return json.loads(DESKTOP_PKG.read_text(encoding="utf-8"))


def test_bundle_script_stages_main_process_without_overwriting_source():
    src = BUNDLE_SCRIPT.read_text(encoding="utf-8")

    assert re.search(r"const\s+entry\s*=\s*resolve\(root,\s*['\"]electron/main\.cjs['\"]\)", src)
    assert re.search(r"const\s+outfile\s*=\s*resolve\(root,\s*['\"]build/electron/main\.cjs['\"]\)", src)
    assert not re.search(r"outfile\s*[:=]\s*entry\b", src)
    assert not re.search(r"(renameSync|copyFileSync|writeFileSync|rmSync|unlinkSync)\([^)]*\bentry\b", src)


def test_electron_builder_packages_staged_main_process_bundle():
    pkg = _desktop_pkg()
    files = pkg.get("build", {}).get("files", [])

    assert "electron/**" not in files

    source_idx = next(
        (
            i
            for i, item in enumerate(files)
            if isinstance(item, dict) and item.get("from") == "electron" and item.get("to") == "electron"
        ),
        None,
    )
    staged_idx = next(
        (
            i
            for i, item in enumerate(files)
            if isinstance(item, dict) and item.get("from") == "build/electron" and item.get("to") == "electron"
        ),
        None,
    )

    assert source_idx is not None, "electron source file set is missing"
    assert staged_idx is not None, "staged Electron main file set is missing"
    assert source_idx < staged_idx
    assert files[source_idx]["filter"] == ["**/*", "!main.cjs"]
    assert files[staged_idx]["filter"] == ["main.cjs"]
    assert pkg["main"] == "electron/main.cjs"


def test_direct_builder_regenerates_staged_main_process_bundle():
    prebuilder = _desktop_pkg().get("scripts", {}).get("prebuilder", "")

    assert "bundle-electron-main.mjs" in prebuilder


def test_nix_desktop_installs_staged_main_process_bundle():
    src = NIX_DESKTOP.read_text(encoding="utf-8")

    assert "node scripts/bundle-electron-main.mjs" in src
    assert "cp -f apps/desktop/build/electron/main.cjs $out/electron/main.cjs" in src


def test_packaged_main_validation_rejects_unbundled_requires():
    helper = PACKAGED_MAIN_VALIDATION.read_text(encoding="utf-8")
    desktop_test = TEST_DESKTOP.read_text(encoding="utf-8")

    assert "builtinModules" in helper
    assert "ALLOWED_BARE_REQUIRES" in helper
    assert "findUnexpectedPackagedMainRequires" in desktop_test
