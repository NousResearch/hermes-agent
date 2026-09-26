"""Keep-data uninstall must not delete the desktop app's Electron userData dir.

Regression for #122548: ``hermes uninstall`` option 1 (Keep data) wiped
``desktop_userdata_dir()`` (connections.json, OAuth partitions, renderer state)
because ``_perform_uninstall`` never forwarded ``remove_userdata``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.uninstall as uninstall


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


@pytest.mark.parametrize("full_uninstall", [False, True])
def test_uninstall_forwards_desktop_userdata_policy(homes, monkeypatch, full_uninstall):
    """Keep-data preserves Electron userData; only the full wipe removes it."""
    home, project_root = homes
    calls: list = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("remove_userdata", "absent"))
        return [Path("sentinel")]

    monkeypatch.setattr("hermes_cli.gui_uninstall.uninstall_gui", spy)
    uninstall._perform_uninstall(
        project_root=project_root, hermes_home=home, full_uninstall=full_uninstall,
        remove_profiles=False, named_profiles=[])
    assert calls == [full_uninstall]
