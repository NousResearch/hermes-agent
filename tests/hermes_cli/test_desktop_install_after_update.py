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


def _patch_sigs(monkeypatch, summaries):
    """Stub the codesign probe: ``summaries(app) -> dict | None``."""
    monkeypatch.setattr(
        main_desktop.shutil, "which",
        lambda name: "/usr/bin/codesign" if name == "codesign" else None)
    monkeypatch.setattr(main_desktop, "_macos_signature_summary", lambda codesign, app: summaries(app))


def _publisher_sig(app, team="TEAM0123", ident="com.nousresearch.hermes", verified=True):
    return {"team": team, "identifier": ident, "verified": verified}


def test_publisher_signed_install_is_never_replaced_by_local_build(rebuilt, tmp_path, monkeypatch):
    """#123748: a Team-ID-signed installation must survive an ad-hoc local rebuild.

    Swapping a publisher-signed app for a locally signed build invalidates the
    code-hash-bound keychain ACLs safeStorage credentials live under — the
    update must retain the working app and report it instead.
    """
    stale = _bundle(tmp_path / "Applications", b"stale")
    _patch_sigs(monkeypatch, lambda app: None if app == rebuilt else _publisher_sig(stale))

    installed, problems = main_desktop._install_rebuilt_macos_bundles(rebuilt, [stale], running=set())

    assert installed == []
    assert len(problems) == 1
    assert "publisher-signed app (Team ID TEAM0123)" in problems[0]
    assert "kept the existing app" in problems[0]
    assert _asar(stale) == b"stale"
    assert not (stale.parent / "Hermes.app.hermes-update-new").exists()
    assert not (stale.parent / "Hermes.app.hermes-update-old").exists()


def test_team_id_mismatch_refuses_swap(rebuilt, tmp_path, monkeypatch):
    stale = _bundle(tmp_path / "Applications", b"stale")
    sigs = {rebuilt: _publisher_sig(rebuilt, team="TEAM9999"), stale: _publisher_sig(stale)}
    _patch_sigs(monkeypatch, lambda app: sigs.get(app))

    installed, problems = main_desktop._install_rebuilt_macos_bundles(rebuilt, [stale], running=set())

    assert installed == [] and len(problems) == 1
    assert "Team ID TEAM0123 does not match rebuilt TEAM9999" in problems[0]
    assert _asar(stale) == b"stale"


def test_bundle_identifier_mismatch_refuses_swap(rebuilt, tmp_path, monkeypatch):
    stale = _bundle(tmp_path / "Applications", b"stale")
    sigs = {
        rebuilt: _publisher_sig(rebuilt, ident="com.other.hermes"),
        stale: _publisher_sig(stale),
    }
    _patch_sigs(monkeypatch, lambda app: sigs.get(app))

    installed, problems = main_desktop._install_rebuilt_macos_bundles(rebuilt, [stale], running=set())

    assert installed == [] and len(problems) == 1
    assert "bundle identifier" in problems[0]
    assert _asar(stale) == b"stale"


def test_unverifiable_rebuilt_refuses_swap(rebuilt, tmp_path, monkeypatch):
    stale = _bundle(tmp_path / "Applications", b"stale")
    sigs = {
        rebuilt: _publisher_sig(rebuilt, verified=False),
        stale: _publisher_sig(stale),
    }
    _patch_sigs(monkeypatch, lambda app: sigs.get(app))

    installed, problems = main_desktop._install_rebuilt_macos_bundles(rebuilt, [stale], running=set())

    assert installed == [] and len(problems) == 1
    assert "strict signature verification" in problems[0]
    assert _asar(stale) == b"stale"


def test_matching_publisher_signing_swaps(rebuilt, tmp_path, monkeypatch):
    """A rebuilt bundle with the SAME publisher identity is a legitimate update: swap."""
    stale = _bundle(tmp_path / "Applications", b"stale")
    sigs = {rebuilt: _publisher_sig(rebuilt), stale: _publisher_sig(stale)}
    _patch_sigs(monkeypatch, lambda app: sigs.get(app))

    installed, problems = main_desktop._install_rebuilt_macos_bundles(rebuilt, [stale], running=set())

    assert installed == [stale] and problems == []
    assert _asar(stale) == b"rebuilt"


def test_adhoc_installed_still_swaps(rebuilt, tmp_path, monkeypatch):
    """Ad-hoc installs (the local development flow) keep swapping regardless of rebuild sig."""
    stale = _bundle(tmp_path / "Applications", b"stale")
    sigs = {stale: {"team": None, "identifier": "com.nousresearch.hermes", "verified": True}}
    _patch_sigs(monkeypatch, lambda app: sigs.get(app))

    installed, problems = main_desktop._install_rebuilt_macos_bundles(rebuilt, [stale], running=set())

    assert installed == [stale] and problems == []
    assert _asar(stale) == b"rebuilt"


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
