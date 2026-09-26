"""Tests for the Windows Desktop launcher shortcut helper."""

from pathlib import Path
from unittest.mock import patch

from hermes_cli.main_desktop import (
    _create_windows_shortcut,
    _resolve_windows_known_folder,
    _windows_shortcut_locations,
)


def test_known_folder_resolver_uses_native_api_for_redirected_and_localized_paths():
    native_paths = {
        "desktop": Path("D:/OneDrive/Área de Trabalho"),
        "programs": Path("C:/Menu Iniciar/Programas"),
    }

    locations = _windows_shortcut_locations(native_paths.__getitem__)

    assert locations == (
        native_paths["programs"] / "Hermes.lnk",
        native_paths["desktop"] / "Hermes.lnk",
    )


def test_known_folder_resolver_reports_unavailable_api_without_profile_fallback():
    def unavailable(_folder_id):
        raise OSError("SHGetKnownFolderPath unavailable")

    try:
        _resolve_windows_known_folder("desktop", api=unavailable)
    except OSError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("unavailable Known Folder API must not fall back to USERPROFILE")


def test_windows_shortcut_uses_builtin_powershell_com(tmp_path: Path):
    shortcut = tmp_path / "Start Menu" / "Hermes.lnk"
    target = tmp_path / "Hermes.exe"

    with patch("hermes_cli.main_desktop.subprocess.run") as run:
        _create_windows_shortcut(shortcut, target, target.parent)

    command = run.call_args.args[0]
    assert command[:4] == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
    ]
    script = command[4]
    assert "New-Object -ComObject WScript.Shell" in script
    assert "com.nousresearch.hermes" in script
    assert "System.AppUserModel.ID" in script
    assert str(shortcut) in script
    assert str(target) in script
    assert run.call_args.kwargs == {
        "check": True,
        "capture_output": True,
        "text": True,
    }


def test_windows_shortcut_quotes_single_quotes_for_powershell(tmp_path: Path):
    shortcut = tmp_path / "Fabio's Desktop" / "Hermes.lnk"
    target = tmp_path / "Fabio's Hermes" / "Hermes.exe"

    with patch("hermes_cli.main_desktop.subprocess.run") as run:
        _create_windows_shortcut(shortcut, target, target.parent)

    script = run.call_args.args[0][4]
    assert "Fabio''s Desktop" in script
    assert "Fabio''s Hermes" in script


def test_windows_shortcut_preserves_shell_properties_and_canonical_aumid(tmp_path: Path):
    shortcut = tmp_path / "Hermes.lnk"
    target = tmp_path / "Hermes.exe"

    with patch("hermes_cli.main_desktop.subprocess.run") as run:
        _create_windows_shortcut(shortcut, target, target.parent)

    script = run.call_args.args[0][4]
    for property_name in ("TargetPath", "WorkingDirectory", "IconLocation", "Description"):
        assert f"$shortcut.{property_name}" in script
    assert "System.AppUserModel.ID" in script
    assert "com.nousresearch.hermes" in script
