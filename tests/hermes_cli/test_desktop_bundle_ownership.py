"""Ownership and reclamation of installed macOS ``Hermes.app`` bundles (#125245, #52339).

``_update_owned_macos_bundles`` decides which installed bundles the CLI updater may replace.
Ownership comes from the bundle's ``install-stamp.json`` (#52339): ``updateMechanism: self`` is
claimed, any other stamp is another mechanism's bundle and is never touched. A stamp-less bundle
is the one case the old rule could not classify: the macOS bootstrap installer ships exactly that
(no stamp, no Electron payload, bundle id ``com.nousresearch.hermes.setup``) under the documented
install path, so the old "no readable stamp, no claim" rule silently orphaned the installed copy
the docs promise the CLI refreshes. The installer is reclaimed by its exact bundle id — precise
enough that an unknown third-party bundle can never be misclaimed — and any other stamp-less
bundle gets a one-line notice instead of silence.
"""

import json
import plistlib
import shutil
from pathlib import Path

import pytest

from hermes_cli import main_desktop

SETUP_BUNDLE_ID = "com.nousresearch.hermes.setup"


def _release_bundle(root: Path, asar: bytes) -> Path:
    app = root / "Hermes.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "Hermes").write_bytes(b"\xcf\xfa\xed\xfe")
    (app / "Contents" / "Resources").mkdir()
    (app / "Contents" / "Resources" / "app.asar").write_bytes(asar)
    return app


def _installer_bundle(root: Path, bundle_id: str = SETUP_BUNDLE_ID) -> Path:
    app = root / "Hermes.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "Hermes-Setup").write_bytes(b"\xcf\xfa\xed\xfe")
    (app / "Contents" / "Resources").mkdir()
    (app / "Contents" / "Resources" / "icon.icns").write_bytes(b"icns")
    (app / "Contents" / "Info.plist").write_bytes(
        plistlib.dumps({"CFBundleIdentifier": bundle_id}))
    return app


def _stamped_bundle(root: Path, asar: bytes, stamp: dict) -> Path:
    app = _release_bundle(root, asar)
    (app / "Contents" / "Resources" / "install-stamp.json").write_text(json.dumps(stamp))
    return app


@pytest.fixture
def rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main_desktop, "_stage_macos_bundle_copy",
        lambda src, dst: shutil.copytree(src, dst, symlinks=True))
    return _release_bundle(tmp_path / "apps" / "desktop" / "release" / "mac-arm64", b"rebuilt")


def test_installer_bundle_left_by_the_dmg_is_claimed_by_bundle_id(tmp_path):
    """#125245: the stamp-less Hermes-Setup installer is claimed by its exact bundle id."""
    installer = _installer_bundle(tmp_path / "Applications")

    assert main_desktop._update_owned_macos_bundles([installer]) == [installer]


def test_installer_occupied_path_is_replaced_by_the_rebuilt_app(rebuilt, tmp_path):
    """#125245: end to end — the installer occupying the install path is swapped out."""
    installer = _installer_bundle(tmp_path / "Applications")

    installed, problems = main_desktop._install_rebuilt_macos_bundles(
        rebuilt, main_desktop._update_owned_macos_bundles([installer]), running=set())

    assert installed == [installer] and problems == []
    assert (installer / "Contents" / "Resources" / "app.asar").read_bytes() == b"rebuilt"
    assert not (installer / "Contents" / "MacOS" / "Hermes-Setup").exists()
    assert not (installer.parent / "Hermes.app.hermes-update-old").exists()


def test_stampless_unknown_bundle_with_payload_is_not_claimed_but_reported(rebuilt, tmp_path):
    """#125245: an unstamped bundle that is NOT the installer is never claimed, but is reported.

    The installer is excluded from the notice: it was reclaimed, not skipped.
    """
    foreign = _release_bundle(tmp_path / "Applications", b"unknown build")  # no stamp, has payload
    foreign = foreign.with_name("Hermes.app")
    # give it a distinct bundle id so it is provably not the setup installer
    (foreign / "Contents" / "Info.plist").write_bytes(
        plistlib.dumps({"CFBundleIdentifier": "com.example.other"}))

    assert main_desktop._update_owned_macos_bundles([foreign]) == []
    notices = main_desktop._unowned_macos_bundle_notices([foreign], rebuilt)
    assert len(notices) == 1
    assert str(foreign) in notices[0] and str(rebuilt) in notices[0]
    assert "no install stamp" in notices[0]


def test_foreign_stamp_and_missing_paths_stay_unclaimed(tmp_path):
    """A release mechanism's bundle and a non-existent path are never claimed or noticed."""
    foreign_release = _stamped_bundle(
        tmp_path / "home" / "Applications", b"release", {"updateMechanism": "electron-updater"})
    missing = tmp_path / "missing" / "Hermes.app"

    candidates = [foreign_release, missing]
    assert main_desktop._update_owned_macos_bundles(candidates) == []
    assert main_desktop._unowned_macos_bundle_notices(candidates, tmp_path / "rebuilt.app") == []
