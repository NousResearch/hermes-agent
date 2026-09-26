"""A matching source stamp must not hide a packaged app whose node-pty has no native binary.

Regression for #62462: ``hermes desktop`` reported "content stamp matches" and
launched a package that crashed on ``require('node-pty')``.
"""
import sys

import pytest

from hermes_cli import main_desktop, source_build


@pytest.fixture
def packaged(tmp_path, monkeypatch):
    desktop = tmp_path / "apps/desktop"
    dist = desktop / "release/fixture/resources/app.asar.unpacked/dist"
    pty = dist / "node_modules/node-pty"
    (pty / "lib").mkdir(parents=True)
    (pty / "package.json").write_text('{"main": "lib/index.js"}', encoding="utf-8")
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("apps/desktop/release/\n", encoding="utf-8")
    # Executable lookup and the per-OS resources layout are covered natively elsewhere.
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable", lambda _: dist / "Hermes")
    monkeypatch.setattr(main_desktop, "_renderer_bundle_dir", lambda *_a, **_k: dist)
    # The compiler owns receipts now. Hold that independent check current so
    # this test isolates the packaged native-binary boundary.
    monkeypatch.setattr(source_build, "source_product_current", lambda *_: True)
    return tmp_path, desktop, pty


def _needed(root, desktop):
    return main_desktop._desktop_build_needed(desktop, root, source_mode=False)


@pytest.mark.parametrize("stale", ["prebuilds/{platform}-x64/", "prebuilds/{other}-x64/pty.node"])
def test_current_stamp_does_not_hide_missing_native_binary(packaged, stale):
    root, desktop, pty = packaged
    other = "linux" if sys.platform == "win32" else "win32"
    path = pty / stale.format(platform=sys.platform, other=other)
    if stale.endswith("/"):
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"MZ")
    assert _needed(root, desktop)


@pytest.mark.parametrize("location", ["build/Release/pty.node", "prebuilds/{platform}-arm64/pty.node"])
def test_package_with_native_binary_stays_current(packaged, location):
    root, desktop, pty = packaged
    binary = pty / location.format(platform=sys.platform)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x7fELF")
    assert not _needed(root, desktop)
