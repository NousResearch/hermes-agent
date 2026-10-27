"""On macOS an update leaves a running Hermes.app alone and parks the rebuilt bundle; once the app
quits, the parked bundle is promoted into ``release/`` AND copied over the installed
``Applications/Hermes.app`` Finder and the Dock launch (#139235, #52339). Real bundles, a real
running process, real codesign and ditto."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import psutil
import pytest

from hermes_cli import main_desktop, main_desktop_pending
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("macos")

REPO_DESKTOP = Path(__file__).resolve().parents[2] / "apps" / "desktop"


def _bundle(root: Path, tag: str) -> Path:
    """A minimal ad-hoc-signable ``Hermes.app`` whose executable is a real Mach-O (a copy of sleep)."""
    app = root / "mac-arm64" / "Hermes.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Resources").mkdir()
    exe = app / "Contents" / "MacOS" / "Hermes"
    shutil.copyfile("/bin/sleep", exe)  # copy2 would copy the system file flags (EPERM)
    os.chmod(exe, 0o755)
    (app / "Contents" / "Info.plist").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>'
        "<key>CFBundleExecutable</key><string>Hermes</string>"
        "<key>CFBundleIdentifier</key><string>com.nousresearch.hermes</string>"
        "<key>CFBundleName</key><string>Hermes</string>"
        "<key>CFBundlePackageType</key><string>APPL</string>"
        "</dict></plist>\n", encoding="utf-8")
    (app / "Contents" / "Resources" / "app.asar").write_bytes(tag.encode())
    (app / "Contents" / "Resources" / "install-stamp.json").write_text(
        json.dumps({"commit": tag, "updateMechanism": "self"}), encoding="utf-8")
    return app


def _asar(app: Path) -> bytes:
    return (app / "Contents" / "Resources" / "app.asar").read_bytes()


def test_an_update_parks_under_a_running_mac_app_and_installs_into_applications_after_quit(tmp_path, monkeypatch):
    from hermes_cli import gui_uninstall
    from hermes_cli import main as hermes_main

    home = tmp_path / "hermes-home"
    (home / "hermes-agent").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", home / "hermes-agent")
    desktop_dir = tmp_path / "repo" / "apps" / "desktop"
    (desktop_dir / "electron").mkdir(parents=True)
    for plist in ("entitlements.mac.plist", "entitlements.mac.inherit.plist"):
        shutil.copy2(REPO_DESKTOP / "electron" / plist, desktop_dir / "electron" / plist)
    live = _bundle(desktop_dir / "release", "OLD")
    installed = tmp_path / "Applications" / "Hermes.app"
    installed.parent.mkdir()
    subprocess.run(["/usr/bin/ditto", str(live), str(installed)], check=True)
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths",
                        lambda: [installed, tmp_path / "user" / "Applications" / "Hermes.app"])
    waiters = []
    monkeypatch.setattr(main_desktop_pending, "spawn_promotion_waiter", waiters.append)

    exe = live / "Contents" / "MacOS" / "Hermes"
    running = subprocess.Popen([str(exe), "60"])
    try:
        wait_until(lambda: Path(psutil.Process(running.pid).exe()).resolve() == exe.resolve(), timeout=10)
        staging = main_desktop._desktop_staging_dir(desktop_dir)
        _bundle(staging, "NEW")

        promoted = main_desktop._promote_staged_desktop_app(desktop_dir, staging)
        assert running.poll() is None, "the running Desktop was stopped"
        assert promoted is None
        main_desktop._refresh_installed_desktop_apps(desktop_dir)  # what the update runs next
        assert _asar(live) == b"OLD" and _asar(installed) == b"OLD"
        assert waiters == [desktop_dir]
    finally:
        running.terminate()
        running.wait()

    assert main_desktop_pending.wait_then_promote(desktop_dir) == exe
    assert _asar(live) == b"NEW"
    assert _asar(installed) == b"NEW"
    assert not main_desktop_pending.pending_dir(desktop_dir).exists()
    # The installed copy carries the promoted bundle's signature intact (ditto, never re-signed).
    subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(installed)], check=True)
    assert json.loads(gui_uninstall.desktop_install_record().read_text(encoding="utf-8-sig"))["apps"] == [str(installed)]
