"""``hermes update`` refreshes the installed macOS ``Hermes.app`` from the rebuilt bundle (#52339).

``hermes desktop --build-only`` only packages into ``apps/desktop/release/``; Finder launches the
copy in ``/Applications``. These pin the contract of ``_install_rebuilt_macos_bundles``: a stale
or missing installed copy is (re)installed, a current one and a running one are never touched, and
a failed swap leaves the previous bundle launchable. Which paths count as installed is decided by
``_installed_desktop_apps``: a recorded copy that went missing comes back until Hermes' own GUI
uninstall drops the record.
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


def test_stale_and_missing_bundles_are_installed_current_and_running_are_left_alone(rebuilt, tmp_path):
    stale = _bundle(tmp_path / "Applications", b"stale")
    current = _bundle(tmp_path / "home" / "Applications", b"rebuilt")
    running = _bundle(tmp_path / "Volumes" / "Applications", b"older")
    current_marker = current / "Contents" / "marker"
    current_marker.write_text("untouched", encoding="utf-8")

    missing = tmp_path / "missing" / "Hermes.app"
    installed, problems = main_desktop._install_rebuilt_macos_bundles(
        rebuilt, [stale, current, running, missing], running={running.resolve()})

    assert installed == [stale, missing]
    assert _asar(stale) == b"rebuilt" and _asar(missing) == b"rebuilt"
    assert not (stale.parent / "Hermes.app.hermes-update-old").exists()
    assert not (stale.parent / "Hermes.app.hermes-update-new").exists()
    assert current_marker.read_text(encoding="utf-8") == "untouched"
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


@pytest.mark.platforms("macos")
def test_recorded_app_that_went_missing_comes_back_until_gui_uninstall(rebuilt, tmp_path, monkeypatch):
    from hermes_cli import gui_uninstall
    from hermes_cli import main as hermes_main

    home = tmp_path / "hermes-home"
    (home / "hermes-agent").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", home / "hermes-agent")
    (rebuilt / "Contents" / "Resources" / "install-stamp.json").write_text('{"updateMechanism": "self"}', encoding="utf-8")
    app = tmp_path / "Applications" / "Hermes.app"
    shutil.copytree(rebuilt, app)
    user_app = tmp_path / "user" / "Applications" / "Hermes.app"
    user_app.parent.mkdir(parents=True)
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths", lambda: [app, user_app])
    monkeypatch.setattr(gui_uninstall, "desktop_userdata_dir", lambda: tmp_path / "userdata")
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable",
                        lambda _d: rebuilt / "Contents" / "MacOS" / "Hermes")
    monkeypatch.setattr(main_desktop, "_running_macos_app_bundles", set)

    main_desktop._refresh_installed_desktop_apps(tmp_path)  # current copy: recorded, untouched
    app.rename(user_app)  # moved by the user: found by its stamp, never doubled
    main_desktop._refresh_installed_desktop_apps(tmp_path)
    assert not app.exists()
    shutil.rmtree(user_app)
    main_desktop._refresh_installed_desktop_apps(tmp_path)
    assert _asar(user_app) == b"rebuilt" and not app.exists()
    gui_uninstall.desktop_install_record().unlink()  # the reinstalled copy carries its own stamp,
    main_desktop._refresh_installed_desktop_apps(tmp_path)  # so it is recorded again from the copy
    shutil.rmtree(user_app)
    main_desktop._refresh_installed_desktop_apps(tmp_path)
    assert _asar(user_app) == b"rebuilt"

    gui_uninstall.uninstall_gui(tmp_path / "profile-home")  # the record is machine-wide
    main_desktop._refresh_installed_desktop_apps(tmp_path)
    assert not user_app.exists()
    assert not gui_uninstall.desktop_install_record().exists()
