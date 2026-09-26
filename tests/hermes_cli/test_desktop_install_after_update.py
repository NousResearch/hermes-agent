"""``hermes update`` refreshes the installed macOS ``Hermes.app`` from the rebuilt bundle (#52339).

``hermes desktop --build-only`` only packages into ``apps/desktop/release/``; Finder launches the
copy in ``/Applications``. These pin the contract of ``_install_rebuilt_macos_bundles``: a stale
installed copy is replaced, a current one and a running one are never touched, and a failed swap
leaves the previous bundle launchable.
"""

import shutil
from pathlib import Path

import pytest

from hermes_cli import main_desktop


def _bundle(root: Path, asar: bytes) -> Path:
    app = root / "Hermes.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "Hermes").write_bytes(b"\xcf\xfa\xed\xfe")
    (app / "Contents" / "Resources").mkdir()
    (app / "Contents" / "Resources" / "app.asar").write_bytes(asar)
    return app


def _asar(app: Path) -> bytes:
    return (app / "Contents" / "Resources" / "app.asar").read_bytes()


@pytest.fixture
def rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main_desktop, "_stage_macos_bundle_copy",
        lambda src, dst: shutil.copytree(src, dst, symlinks=True))
    return _bundle(tmp_path / "apps" / "desktop" / "release" / "mac-arm64", b"rebuilt")


def test_stale_bundle_is_replaced_current_and_running_are_left_alone(rebuilt, tmp_path):
    stale = _bundle(tmp_path / "Applications", b"stale")
    current = _bundle(tmp_path / "home" / "Applications", b"rebuilt")
    running = _bundle(tmp_path / "Volumes" / "Applications", b"older")
    current_marker = current / "Contents" / "marker"
    current_marker.write_text("untouched")

    installed, problems = main_desktop._install_rebuilt_macos_bundles(
        rebuilt, [stale, current, running, tmp_path / "missing" / "Hermes.app"],
        running={running.resolve()})

    assert installed == [stale]
    assert _asar(stale) == b"rebuilt"
    assert not (stale.parent / "Hermes.app.hermes-update-old").exists()
    assert not (stale.parent / "Hermes.app.hermes-update-new").exists()
    assert current_marker.read_text() == "untouched"
    # A live app is reported, never swapped under.
    assert _asar(running) == b"older"
    assert len(problems) == 1 and str(running) in problems[0]


def test_failed_swap_keeps_the_previous_bundle_launchable(rebuilt, tmp_path, monkeypatch):
    stale = _bundle(tmp_path / "Applications", b"stale")
    real_rename = Path.rename

    def fail_final_rename(self, target):
        if self.name.endswith(".hermes-update-new"):
            raise OSError("simulated rename failure")
        return real_rename(self, target)
    monkeypatch.setattr(Path, "rename", fail_final_rename)

    installed, problems = main_desktop._install_rebuilt_macos_bundles(rebuilt, [stale], running=set())

    assert installed == []
    assert len(problems) == 1
    assert stale.is_dir() and _asar(stale) == b"stale"
    assert not (stale.parent / "Hermes.app.hermes-update-new").exists()
