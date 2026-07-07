"""Tests for _install_rebuilt_desktop_app — CLI update installs rebuilt app.

Verifies that ``hermes update`` (via ``_install_rebuilt_desktop_app``) copies
the freshly rebuilt desktop app to the system install location when the
installed copy is stale, and skips when hashes already match.
"""

import hashlib
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
