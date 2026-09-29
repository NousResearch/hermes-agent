"""Keep-data uninstall must not delete the desktop app's Electron userData dir.

Regression for #122548: ``hermes uninstall`` option 1 (Keep data) wiped
``desktop_userdata_dir()`` (connections.json, OAuth partitions, renderer state)
because ``_perform_uninstall`` never forwarded ``remove_userdata``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import hermes_cli.gui_uninstall as gui_uninstall
import hermes_cli.uninstall as uninstall


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Temp home/checkout/userData; every step that reaches outside tmp_path is stubbed.

    The full wipe otherwise boots out real launchd jobs, rmtrees real
    ~/Library/Caches dirs and /Applications/Hermes.app, and (on Windows) edits
    the registry — so stub those helpers and refuse any stray rmtree or
    subprocess instead of silently touching the developer's machine. Refusals
    are recorded in ``violations`` because the uninstall steps catch and log
    exceptions, so a raised AssertionError alone would leave the test green.
    """
    home = tmp_path / "hermes-home"
    home.mkdir()
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    userdata = tmp_path / "userdata"
    userdata.mkdir()
    (userdata / "connections.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(uninstall, "_is_windows", lambda: False)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: True)
    for name in ("remove_path_from_shell_configs", "remove_wrapper_script",
                 "remove_node_symlinks", "remove_legacy_runtime_trees",
                 "remove_dashboard_launchd_jobs", "_macos_cache_leftover_dirs"):
        monkeypatch.setattr(uninstall, name, lambda *a, **k: [])
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths", lambda: [])
    monkeypatch.setattr(gui_uninstall, "desktop_userdata_dir", lambda: userdata)

    real_rmtree = shutil.rmtree
    violations: list[str] = []

    def confined_rmtree(path, *args, **kwargs):
        if not Path(path).is_relative_to(tmp_path):
            violations.append(f"rmtree outside tmp: {path}")
            raise AssertionError(violations[-1])
        return real_rmtree(path, *args, **kwargs)

    def no_subprocess(*args, **kwargs):
        violations.append(f"unexpected subprocess: {args}")
        raise AssertionError(violations[-1])

    monkeypatch.setattr(shutil, "rmtree", confined_rmtree)
    monkeypatch.setattr(uninstall.subprocess, "run", no_subprocess)
    return home, project_root, userdata, violations


@pytest.mark.parametrize("full_uninstall", [False, True])
def test_uninstall_desktop_userdata_kept_unless_full(homes, full_uninstall):
    """Keep-data preserves Electron userData; only the full wipe removes it."""
    home, project_root, userdata, violations = homes
    uninstall._perform_uninstall(
        project_root=project_root, hermes_home=home, full_uninstall=full_uninstall,
        remove_profiles=False, named_profiles=[])
    assert violations == []
    assert (userdata / "connections.json").exists() is not full_uninstall
