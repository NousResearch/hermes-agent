"""Ownership of installed macOS ``Hermes.app`` bundles (#125245).

``_update_owned_macos_bundles`` decides which installed bundles the CLI updater may replace.
Ownership comes from the bundle's ``install-stamp.json`` (#52339): ``updateMechanism: self`` is
claimed, any other stamp is another mechanism's bundle and is never touched. A stamp-less bundle
with no Electron payload is the bootstrap installer occupying the documented install path, so it
is reclaimed instead of being silently skipped forever; a stamp-less bundle that DOES carry a
payload is an unknown build and stays unclaimed.
"""

import json
from pathlib import Path

from hermes_cli import main_desktop


def _bundle(root: Path, *, stamp: dict | None = None, payload: bool = False) -> Path:
    app = root / "Hermes.app"
    resources = app / "Contents" / "Resources"
    resources.mkdir(parents=True)
    (resources / "icon.icns").write_bytes(b"icns")
    if stamp is not None:
        (resources / "install-stamp.json").write_text(json.dumps(stamp))
    if payload:
        (resources / "app.asar").write_bytes(b"asar")
    return app


def test_installer_bundle_without_stamp_or_payload_is_reclaimed(tmp_path):
    installer = _bundle(tmp_path / "Applications")

    assert main_desktop._update_owned_macos_bundles([installer]) == [installer]


def test_stampless_bundle_with_payload_and_foreign_stamp_stay_unclaimed(tmp_path):
    unknown_build = _bundle(tmp_path / "Applications", payload=True)
    foreign_release = _bundle(
        tmp_path / "home" / "Applications", stamp={"updateMechanism": "external"}, payload=True)

    assert main_desktop._update_owned_macos_bundles([unknown_build, foreign_release]) == []


def test_reclaimed_installer_bundle_flows_through_the_swap(tmp_path, monkeypatch):
    """End to end: the installer-occupied path is replaced by the rebuilt bundle."""
    import shutil

    rebuilt = _bundle(tmp_path / "release", payload=True)
    installer = _bundle(tmp_path / "Applications")
    monkeypatch.setattr(
        main_desktop, "_stage_macos_bundle_copy",
        lambda src, dst: shutil.copytree(src, dst, symlinks=True))

    owned = main_desktop._update_owned_macos_bundles([installer])
    installed, problems = main_desktop._install_rebuilt_macos_bundles(
        rebuilt, main_desktop._update_owned_macos_bundles([installer]), running=set())

    assert owned == [installer] and installed == [installer] and problems == []
    assert (installer / "Contents" / "Resources" / "app.asar").read_bytes() == b"asar"
