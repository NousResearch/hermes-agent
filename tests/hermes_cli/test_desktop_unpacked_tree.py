"""An interrupted electron-builder run must not be promoted or trusted (#102013).

EBUSY on a locked file can leave the product exe in place while the Chromium payload
or the renderer bundle is missing; the PE check alone passes such a tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_cli import main_desktop

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="macOS bundles use a framework layout")

INDEX = '<script type="module" src="./assets/index.js"></script>'


def _tree(root: Path, *, skip: str | None = None) -> Path:
    files = {
        "Hermes.exe": "MZ",
        "icudtl.dat": "x",
        "resources.pak": "x",
        "locales/en-US.pak": "x",
        "resources/app.asar": "x",
        "resources/app.asar.unpacked/dist/index.html": INDEX,
        "resources/app.asar.unpacked/dist/assets/index.js": "x",
    }
    for rel, body in files.items():
        if rel == skip:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root / "Hermes.exe"


def test_complete_tree_passes(tmp_path):
    assert main_desktop._unpacked_tree_error(_tree(tmp_path)) is None


@pytest.mark.parametrize("missing", [
    "icudtl.dat",
    "resources.pak",
    "locales/en-US.pak",
    "resources/app.asar",
    "resources/app.asar.unpacked/dist/index.html",
])
def test_missing_runtime_file_is_reported(tmp_path, missing):
    error = main_desktop._unpacked_tree_error(_tree(tmp_path, skip=missing))
    assert error is not None and "incomplete Electron tree" in error


def test_torn_renderer_chunks_are_reported(tmp_path):
    exe = _tree(tmp_path, skip="resources/app.asar.unpacked/dist/assets/index.js")
    assert "renderer chunks" in main_desktop._unpacked_tree_error(exe)


def test_promote_refuses_a_tree_with_a_valid_exe_but_missing_icu(tmp_path, monkeypatch):
    desktop = tmp_path / "desktop"
    staging = desktop / ".staging-1-1"
    exe = _tree(staging / "win-unpacked", skip="icudtl.dat")
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable_in", lambda _d: exe)
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda *a, **k: True)
    swapped = []
    monkeypatch.setattr(main_desktop, "_swap_staged_desktop_app", lambda *a: swapped.append(a))

    with pytest.raises(RuntimeError, match="no launchable app"):
        main_desktop._promote_staged_desktop_app(desktop, staging, integrity_check=lambda _p: None)

    assert not swapped and not staging.exists()


def test_build_needed_when_live_tree_is_incomplete(tmp_path, monkeypatch):
    exe = _tree(tmp_path / "release" / "win-unpacked", skip="resources.pak")
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable", lambda _d: exe)
    assert main_desktop._desktop_build_needed(tmp_path, tmp_path, source_mode=False) is True
