"""Tests for the Windows Desktop launcher shortcut helper."""

from pathlib import Path
from unittest.mock import patch

from hermes_cli.main_desktop import _create_windows_shortcut


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
