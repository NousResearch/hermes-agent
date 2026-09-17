"""A headless host must not build (or rebuild) the Electron desktop app.

`hermes update` rebuilds the desktop app whenever its artifacts exist, so a GUI app that once
landed on a display-less server is rebuilt on **every** update — ~1.75 GB of Electron +
node_modules that nobody on that host can launch. The host still legitimately runs
`hermes serve --isolated` backends for a Desktop on another machine; the app itself is the
waste. `--force-build` / `--skip-build` stay explicit overrides, and a forwarded or Xvfb
display counts as present (CI drives the desktop e2e under `xvfb-run`).
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main_desktop, update_cmd, update_cmd_deps


@pytest.fixture(autouse=True)
def _clean_display_env(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)


def test_display_available_on_non_linux(monkeypatch):
    # A Mac or PC always has a window server; only Linux servers are routinely headless.
    for platform in ("darwin", "win32"):
        monkeypatch.setattr(sys, "platform", platform)
        assert main_desktop.desktop_display_available() is True


@pytest.mark.parametrize("var", ["DISPLAY", "WAYLAND_DISPLAY"])
def test_display_available_from_env(monkeypatch, var):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(main_desktop.glob, "glob", lambda pattern: [])
    monkeypatch.setenv(var, ":0")
    assert main_desktop.desktop_display_available() is True


def test_display_available_from_x_socket(monkeypatch):
    """xvfb-run publishes /tmp/.X11-unix/X99 and sets DISPLAY; the socket alone must suffice."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(main_desktop.glob, "glob", lambda pattern: ["/tmp/.X11-unix/X99"])
    assert main_desktop.desktop_display_available() is True


def test_display_unavailable_on_a_bare_server(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(main_desktop.glob, "glob", lambda pattern: [])
    assert main_desktop.desktop_display_available() is False


def test_display_probe_failure_is_not_a_refusal(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    def _boom(pattern):
        raise OSError("no /tmp")

    monkeypatch.setattr(main_desktop.glob, "glob", _boom)
    assert main_desktop.desktop_display_available() is False


def _gui_args(**overrides) -> argparse.Namespace:
    args = {"source": False, "skip_build": False, "force_build": False, "build_only": False}
    args.update(overrides)
    return argparse.Namespace(**args)


def _fake_project_root(tmp_path):
    (tmp_path / "apps" / "desktop").mkdir(parents=True)
    (tmp_path / "apps" / "desktop" / "package.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_build_only_on_a_headless_host_warns_but_still_builds(tmp_path, monkeypatch):
    """An explicit `hermes desktop` is a human decision: warn, never silently skip."""
    build = MagicMock(return_value=tmp_path / "apps" / "desktop" / "release" / "linux-unpacked" / "Hermes")
    with patch("hermes_cli.main.PROJECT_ROOT", _fake_project_root(tmp_path)), \
         patch.object(main_desktop, "desktop_display_available", return_value=False), \
         patch.object(main_desktop, "_build_desktop_app", build), \
         patch.object(main_desktop, "_register_linux_desktop_entry"), \
         patch("hermes_cli.main_install_repair._resolve_node_runtime_npm", return_value="npm"):
        assert main_desktop.cmd_gui(_gui_args(build_only=True)) is None
    build.assert_called_once()


def test_force_build_silences_the_headless_notice(tmp_path, monkeypatch, capsys):
    build = MagicMock(return_value=tmp_path / "apps" / "desktop" / "release" / "linux-unpacked" / "Hermes")
    with patch("hermes_cli.main.PROJECT_ROOT", _fake_project_root(tmp_path)), \
         patch.object(main_desktop, "desktop_display_available", return_value=False), \
         patch.object(main_desktop, "_build_desktop_app", build), \
         patch.object(main_desktop, "_register_linux_desktop_entry"), \
         patch("hermes_cli.main_install_repair._resolve_node_runtime_npm", return_value="npm"):
        main_desktop.cmd_gui(_gui_args(build_only=True, force_build=True))
    assert "No display on this host" not in capsys.readouterr().out


def test_update_skips_the_desktop_rebuild_without_a_display(tmp_path):
    desktop_dir = _fake_project_root(tmp_path) / "apps" / "desktop"
    run_build = MagicMock(side_effect=AssertionError("must not rebuild without a display"))
    with patch("hermes_cli.main_desktop.desktop_display_available", return_value=False), \
         patch.object(update_cmd, "_m") as m:
        m.return_value._desktop_app_present.return_value = True
        m.return_value._resolve_node_runtime_npm.return_value = "npm"
        m.return_value.PROJECT_ROOT = tmp_path
        m.return_value._run_logged_subprocess = run_build
        assert update_cmd_deps._rebuild_desktop_after_update(
            desktop_dir, had_desktop_app_before_update=True) is True
    run_build.assert_not_called()


def test_update_still_rebuilds_when_a_display_exists(tmp_path, monkeypatch):
    desktop_dir = _fake_project_root(tmp_path) / "apps" / "desktop"
    result = MagicMock(returncode=0, stdout="", stderr="")
    with patch("hermes_cli.main_desktop.desktop_display_available", return_value=True), \
         patch.object(update_cmd, "_m") as m:
        m.return_value._desktop_app_present.return_value = True
        m.return_value._resolve_node_runtime_npm.return_value = "npm"
        m.return_value.PROJECT_ROOT = tmp_path
        m.return_value._desktop_build_needed.return_value = True
        m.return_value._run_logged_subprocess.return_value = result
        assert update_cmd_deps._rebuild_desktop_after_update(
            desktop_dir, had_desktop_app_before_update=True) is True
    m.return_value._run_logged_subprocess.assert_called_once()
