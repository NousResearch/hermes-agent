"""Tests for _install_rebuilt_desktop_app — CLI update installs rebuilt app.

Verifies that ``hermes update`` (via ``_install_rebuilt_desktop_app``) copies
the freshly rebuilt desktop app to the system install location when the
installed copy is stale, and skips when hashes already match.
"""

import hashlib
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def fake_desktop_dir(tmp_path):
    """A fake desktop_dir with a rebuilt macOS app bundle."""
    release = tmp_path / "apps" / "desktop" / "release" / "mac-arm64" / "Hermes.app"
    # Mimic the real structure: Contents/MacOS/Hermes + Contents/Resources/app.asar
    exe = release / "Contents" / "MacOS" / "Hermes"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"\xcf\xfa\xed\xfe")  # minimal Mach-O magic
    asar = release / "Contents" / "Resources" / "app.asar"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"rebuilt-asar-content")
    return tmp_path / "apps" / "desktop"


@pytest.fixture
def fake_installed_app(tmp_path):
    """A fake installed Hermes.app with a DIFFERENT app.asar."""
    installed = tmp_path / "Applications" / "Hermes.app"
    exe = installed / "Contents" / "MacOS" / "Hermes"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"\xcf\xfa\xed\xfe")
    asar = installed / "Contents" / "Resources" / "app.asar"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"stale-asar-content")
    return installed


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy_ditto_bundle(command, **_kwargs):
    """Emulate ``ditto SOURCE DEST`` with an ordinary directory copy."""
    shutil.copytree(Path(command[1]), Path(command[2]))
    return MagicMock(returncode=0)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS .app bundle test")
def test_install_when_stale(fake_desktop_dir, fake_installed_app):
    """_install_rebuilt_desktop_app copies the rebuilt app when hashes differ."""
    from hermes_cli.main import _install_rebuilt_desktop_app

    # Patch _desktop_packaged_executable to find our fake rebuilt app
    fake_rebuilt_exe = (
        fake_desktop_dir
        / "release"
        / "mac-arm64"
        / "Hermes.app"
        / "Contents"
        / "MacOS"
        / "Hermes"
    )
    with (
        patch(
            "hermes_cli.main._desktop_packaged_executable",
            return_value=fake_rebuilt_exe,
        ),
        patch(
            "hermes_cli.gui_uninstall.packaged_gui_app_paths",
            return_value=[fake_installed_app],
        ),
        patch("hermes_cli.main._macos_adhoc_sign_bundle") as mock_sign,
    ):
        # Let ditto/codesign run for real — they exist on macOS
        result = _install_rebuilt_desktop_app(fake_desktop_dir)

    assert result == fake_installed_app
    mock_sign.assert_called_once_with(fake_installed_app)
    # The installed app.asar should now match the rebuilt one
    rebuilt_asar = (
        fake_desktop_dir
        / "release"
        / "mac-arm64"
        / "Hermes.app"
        / "Contents"
        / "Resources"
        / "app.asar"
    ).read_bytes()
    installed_asar = (
        fake_installed_app / "Contents" / "Resources" / "app.asar"
    ).read_bytes()
    assert rebuilt_asar == installed_asar


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS .app bundle test")
def test_skip_when_hashes_match(fake_desktop_dir, tmp_path):
    """_install_rebuilt_desktop_app returns None when hashes already match."""
    from hermes_cli.main import _install_rebuilt_desktop_app, _app_asar_hash

    # Create an installed app with the SAME app.asar as the rebuilt one
    rebuilt_asar = (
        fake_desktop_dir
        / "release"
        / "mac-arm64"
        / "Hermes.app"
        / "Contents"
        / "Resources"
        / "app.asar"
    ).read_bytes()

    installed = tmp_path / "Applications" / "Hermes.app"
    asar = installed / "Contents" / "Resources" / "app.asar"
    asar.parent.mkdir(parents=True)
    asar.write_bytes(rebuilt_asar)

    fake_rebuilt_exe = (
        fake_desktop_dir
        / "release"
        / "mac-arm64"
        / "Hermes.app"
        / "Contents"
        / "MacOS"
        / "Hermes"
    )

    with (
        patch(
            "hermes_cli.main._desktop_packaged_executable",
            return_value=fake_rebuilt_exe,
        ),
        patch(
            "hermes_cli.gui_uninstall.packaged_gui_app_paths",
            return_value=[installed],
        ),
        patch("shutil.which", return_value="/usr/bin/ditto"),
        patch("subprocess.run") as mock_run,
    ):
        result = _install_rebuilt_desktop_app(fake_desktop_dir)

    assert result is None
    # ditto should NOT have been called
    mock_run.assert_not_called()


def test_returns_none_when_no_installed_app(fake_desktop_dir):
    """_install_rebuilt_desktop_app returns None when nothing is installed."""
    from hermes_cli.main import _install_rebuilt_desktop_app

    fake_rebuilt_exe = (
        fake_desktop_dir
        / "release"
        / "mac-arm64"
        / "Hermes.app"
        / "Contents"
        / "MacOS"
        / "Hermes"
    )

    with (
        patch(
            "hermes_cli.main._desktop_packaged_executable",
            return_value=fake_rebuilt_exe,
        ),
        patch(
            "hermes_cli.gui_uninstall.packaged_gui_app_paths",
            return_value=[],
        ),
    ):
        result = _install_rebuilt_desktop_app(fake_desktop_dir)

    assert result is None


def test_returns_none_when_no_rebuilt_app(fake_desktop_dir):
    """_install_rebuilt_desktop_app returns None when no rebuilt app exists."""
    from hermes_cli.main import _install_rebuilt_desktop_app

    with patch(
        "hermes_cli.main._desktop_packaged_executable",
        return_value=None,
    ):
        result = _install_rebuilt_desktop_app(fake_desktop_dir)

    assert result is None


def test_app_asar_hash_consistency(fake_desktop_dir):
    """_app_asar_hash returns consistent hashes for the same content."""
    from hermes_cli.main import _app_asar_hash

    rebuilt_app = fake_desktop_dir / "release" / "mac-arm64" / "Hermes.app"
    h1 = _app_asar_hash(rebuilt_app)
    h2 = _app_asar_hash(rebuilt_app)
    assert h1 is not None
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_app_asar_hash_none_for_missing(tmp_path):
    """_app_asar_hash returns None when app.asar doesn't exist."""
    from hermes_cli.main import _app_asar_hash

    fake_app = tmp_path / "Fake.app"
    fake_app.mkdir()
    assert _app_asar_hash(fake_app) is None


def test_backup_rename_failure_leaves_installed_bundle_untouched(
    fake_desktop_dir, fake_installed_app, monkeypatch
):
    """Failure to park the old bundle must not delete or replace it."""
    from hermes_cli import main as hermes_main

    rebuilt_exe = (
        fake_desktop_dir
        / "release"
        / "mac-arm64"
        / "Hermes.app"
        / "Contents"
        / "MacOS"
        / "Hermes"
    )
    original_rename = Path.rename

    def fail_backup_rename(path, target):
        if path == fake_installed_app:
            raise OSError("cannot move installed bundle aside")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_backup_rename)
    with (
        patch.object(hermes_main.sys, "platform", "darwin"),
        patch.object(
            hermes_main,
            "_desktop_packaged_executable",
            return_value=rebuilt_exe,
        ),
        patch(
            "hermes_cli.gui_uninstall.packaged_gui_app_paths",
            return_value=[fake_installed_app],
        ),
        patch.object(hermes_main.shutil, "which", return_value="/usr/bin/ditto"),
        patch.object(hermes_main.subprocess, "run", side_effect=_copy_ditto_bundle),
        patch.object(hermes_main, "_macos_adhoc_sign_bundle") as mock_sign,
    ):
        result = hermes_main._install_rebuilt_desktop_app(fake_desktop_dir)

    assert result is None
    assert (
        fake_installed_app / "Contents" / "Resources" / "app.asar"
    ).read_bytes() == b"stale-asar-content"
    assert not fake_installed_app.with_name("Hermes.app.hermes-update-new").exists()
    assert not fake_installed_app.with_name("Hermes.app.hermes-update-old").exists()
    mock_sign.assert_not_called()


def test_new_bundle_rename_failure_restores_installed_bundle(
    fake_desktop_dir, fake_installed_app, monkeypatch
):
    """Failure to install the staged bundle must roll the old bundle back."""
    from hermes_cli import main as hermes_main

    rebuilt_exe = (
        fake_desktop_dir
        / "release"
        / "mac-arm64"
        / "Hermes.app"
        / "Contents"
        / "MacOS"
        / "Hermes"
    )
    tmp = fake_installed_app.with_name("Hermes.app.hermes-update-new")
    old = fake_installed_app.with_name("Hermes.app.hermes-update-old")
    original_rename = Path.rename

    def fail_new_bundle_rename(path, target):
        if path == tmp and target == fake_installed_app:
            raise OSError("cannot move staged bundle into place")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_new_bundle_rename)
    with (
        patch.object(hermes_main.sys, "platform", "darwin"),
        patch.object(
            hermes_main,
            "_desktop_packaged_executable",
            return_value=rebuilt_exe,
        ),
        patch(
            "hermes_cli.gui_uninstall.packaged_gui_app_paths",
            return_value=[fake_installed_app],
        ),
        patch.object(hermes_main.shutil, "which", return_value="/usr/bin/ditto"),
        patch.object(hermes_main.subprocess, "run", side_effect=_copy_ditto_bundle),
        patch.object(hermes_main, "_macos_adhoc_sign_bundle") as mock_sign,
    ):
        result = hermes_main._install_rebuilt_desktop_app(fake_desktop_dir)

    assert result is None
    assert (
        fake_installed_app / "Contents" / "Resources" / "app.asar"
    ).read_bytes() == b"stale-asar-content"
    assert not tmp.exists()
    assert not old.exists()
    mock_sign.assert_not_called()


def test_linux_desktop_launchers_are_not_install_payloads(fake_desktop_dir, tmp_path):
    """Linux .desktop launchers must never be copytree destinations."""
    from hermes_cli import main as hermes_main

    rebuilt_exe = fake_desktop_dir / "release" / "linux-unpacked" / "hermes"
    rebuilt_exe.parent.mkdir(parents=True)
    rebuilt_exe.write_bytes(b"linux executable")
    launcher = tmp_path / "share" / "applications" / "hermes.desktop"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("[Desktop Entry]\nExec=hermes\n", encoding="utf-8")

    with (
        patch.object(hermes_main.sys, "platform", "linux"),
        patch.object(
            hermes_main,
            "_desktop_packaged_executable",
            return_value=rebuilt_exe,
        ),
        patch(
            "hermes_cli.gui_uninstall.packaged_gui_app_paths",
            return_value=[launcher],
        ) as mock_paths,
        patch.object(hermes_main.shutil, "copytree") as mock_copytree,
    ):
        result = hermes_main._install_rebuilt_desktop_app(fake_desktop_dir)

    assert result is None
    assert launcher.read_text(encoding="utf-8") == "[Desktop Entry]\nExec=hermes\n"
    mock_paths.assert_not_called()
    mock_copytree.assert_not_called()
