"""Tests for the Linux Hermes Desktop launcher installer/wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from hermes_cli import main as cli_main
from hermes_cli.linux_desktop_launcher import (
    DESKTOP_FILENAME,
    LAUNCHER_NAME,
    finalize_packaged_linux_desktop,
    install_linux_desktop_shortcuts,
    render_desktop_entry,
    render_launcher_script,
    resolve_packaged_executable,
)


def _make_agent_tree(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-agent"
    desktop = root / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    (desktop / "assets").mkdir(parents=True)
    (desktop / "assets" / "icon.png").write_bytes(b"png")
    exe = desktop / "release" / "linux-unpacked" / "Hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    return root


def test_resolve_packaged_executable_prefers_newest(tmp_path):
    root = _make_agent_tree(tmp_path)
    older = root / "apps" / "desktop" / "release" / "linux-unpacked" / "hermes"
    older.write_text("", encoding="utf-8")
    resolved = resolve_packaged_executable(root)
    assert resolved == root / "apps" / "desktop" / "release" / "linux-unpacked" / "Hermes"


def test_render_launcher_script_points_at_python_module(tmp_path):
    root = _make_agent_tree(tmp_path)
    script = render_launcher_script(root)
    assert 'python" -m hermes_cli.linux_desktop_launcher launch' in script
    assert 'export HERMES_AGENT="${HERMES_AGENT:-$HERMES_HOME/hermes-agent}"' in script


def test_render_desktop_entry_uses_wrapper_not_raw_binary(tmp_path):
    launcher = tmp_path / "bin" / LAUNCHER_NAME
    icon = tmp_path / "icon.png"
    entry = render_desktop_entry(launcher_path=launcher, icon_path=icon)
    assert f"Exec={launcher}" in entry
    assert "StartupWMClass=Hermes" in entry
    assert "Terminal=false" in entry


def test_install_linux_desktop_shortcuts_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.linux_desktop_launcher.sys.platform", "linux")
    monkeypatch.setattr("hermes_cli.linux_desktop_launcher.Path.home", lambda: tmp_path)
    root = _make_agent_tree(tmp_path)

    written = install_linux_desktop_shortcuts(root)

    launcher = tmp_path / ".local" / "bin" / LAUNCHER_NAME
    desktop = tmp_path / ".local" / "share" / "applications" / DESKTOP_FILENAME
    assert written == [launcher, desktop]
    assert launcher.is_file()
    assert desktop.is_file()
    assert "hermes_cli.linux_desktop_launcher launch" in launcher.read_text(encoding="utf-8")


def test_finalize_packaged_linux_desktop_runs_fixup_and_install(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.linux_desktop_launcher.sys.platform", "linux")
    monkeypatch.setattr("hermes_cli.linux_desktop_launcher.Path.home", lambda: tmp_path)
    root = _make_agent_tree(tmp_path)
    exe = resolve_packaged_executable(root)
    assert exe is not None

    with patch("hermes_cli.linux_desktop_launcher.ensure_electron_sandbox_fixup", return_value=True) as mock_fixup:
        assert finalize_packaged_linux_desktop(root, exe) is True

    mock_fixup.assert_called_once_with(exe)
    assert (tmp_path / ".local" / "bin" / LAUNCHER_NAME).is_file()


def test_gui_build_only_finalizes_linux_desktop(tmp_path, monkeypatch):
    root = _make_agent_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    packaged = resolve_packaged_executable(root)
    assert packaged is not None

    with patch("hermes_cli.main._desktop_packaged_executable", return_value=packaged), \
         patch("hermes_cli.main._finalize_packaged_linux_desktop") as mock_finalize:
        cli_main.cmd_gui(argparse.Namespace(
            skip_build=True,
            build_only=True,
            force_build=False,
            source=False,
            fake_boot=False,
            ignore_existing=False,
            hermes_root=None,
            cwd=None,
        ))

    mock_finalize.assert_called_once_with(packaged)
