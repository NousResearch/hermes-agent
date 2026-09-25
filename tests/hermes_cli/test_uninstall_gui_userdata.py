"""Keep-data uninstall must not delete the desktop app's Electron userData dir.

Regression for #122548: ``hermes uninstall`` option 1 (Keep data) wiped
``desktop_userdata_dir()`` (connections.json, OAuth partitions, renderer state)
because ``_perform_uninstall`` never forwarded ``remove_userdata``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.uninstall as uninstall
from hermes_cli.gui_uninstall import uninstall_gui


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Temp home + checkout with sentinel data; uninstall side effects are stubbed."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text("{}", encoding="utf-8")
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(uninstall, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: home)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: True)
    for name in ("remove_path_from_shell_configs", "remove_wrapper_script",
                 "remove_node_symlinks", "remove_legacy_runtime_trees"):
        monkeypatch.setattr(uninstall, name, lambda *a, **k: [])
    return home, project_root


def test_keep_data_uninstall_preserves_desktop_userdata(homes, monkeypatch):
    home, _ = homes
    calls: list[bool | None] = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("remove_userdata", "absent"))
        return [Path("sentinel")]

    monkeypatch.setattr("hermes_cli.gui_uninstall.uninstall_gui", spy)
    uninstall._perform_uninstall(
        project_root=homes[1], hermes_home=home, full_uninstall=False,
        remove_profiles=False, named_profiles=[])
    assert calls == [False], "keep-data must forward remove_userdata=False to uninstall_gui"
    assert (home / "config.yaml").exists(), "keep-data must keep $HERMES_HOME data"


def test_full_uninstall_still_removes_desktop_userdata(homes, monkeypatch):
    home, _ = homes
    calls: list[bool | None] = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("remove_userdata", "absent"))
        return [Path("sentinel")]

    monkeypatch.setattr("hermes_cli.gui_uninstall.uninstall_gui", spy)
    uninstall._perform_uninstall(
        project_root=homes[1], hermes_home=home, full_uninstall=True,
        remove_profiles=False, named_profiles=[])
    assert calls == [True], "full wipe must forward remove_userdata=True to uninstall_gui"
    assert not home.exists(), "full wipe must remove $HERMES_HOME"


def test_uninstall_gui_keeps_userdata_dir_when_told_to(tmp_path, monkeypatch):
    """The gui_uninstall half of the contract: remove_userdata=False leaves the dir on disk."""
    home = tmp_path / "home"
    home.mkdir()
    built = home / "hermes-agent" / "apps" / "desktop" / "node_modules"
    built.mkdir(parents=True)
    userdata = tmp_path / "appdata" / "Hermes"
    (userdata / "Partitions").mkdir(parents=True)
    (userdata / "connections.json").write_text('{"id":"local"}', encoding="utf-8")
    monkeypatch.setattr("hermes_cli.gui_uninstall.desktop_userdata_dir", lambda: userdata)

    removed = uninstall_gui(home, remove_userdata=False)
    assert userdata.exists(), "remove_userdata=False must keep the Electron userData dir"
    assert (userdata / "connections.json").read_text(encoding="utf-8") == '{"id":"local"}'
    assert not built.exists(), "built GUI artifacts are still removed"
    assert userdata not in removed

    uninstall_gui(home, remove_userdata=True)
    assert not userdata.exists(), "remove_userdata=True must remove the Electron userData dir"
