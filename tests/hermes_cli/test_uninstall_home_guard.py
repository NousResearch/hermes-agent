"""Destructive uninstall refuses paths that can escape Hermes-owned state."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import uninstall


def test_full_uninstall_refuses_the_user_home_before_removing_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    user_home = tmp_path / "user"
    user_home.mkdir()
    personal_sentinel = user_home / "personal.txt"
    personal_sentinel.write_text("keep", encoding="utf-8")

    project = tmp_path / "install" / "hermes-agent"
    (project / ".git").mkdir(parents=True)
    code_sentinel = project / "hermes_cli.py"
    code_sentinel.write_text("keep", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: user_home)
    monkeypatch.setattr(uninstall, "get_project_root", lambda: project)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: True)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda home: [])
    monkeypatch.setattr(uninstall, "remove_desktop_app_leftovers", lambda **kwargs: [])
    monkeypatch.setattr(uninstall, "remove_legacy_runtime_trees", lambda home: [])
    monkeypatch.setattr(uninstall, "remove_dashboard_launchd_jobs", lambda: [])
    monkeypatch.setattr(uninstall, "_macos_cache_leftover_dirs", lambda: [])
    monkeypatch.setattr("hermes_cli.gui_uninstall.uninstall_gui", lambda home, **kwargs: True)

    with pytest.raises(SystemExit) as failure:
        uninstall.main(["--mode", "full"])

    assert failure.value.code != 0
    assert personal_sentinel.read_text(encoding="utf-8-sig") == "keep"
    assert code_sentinel.read_text(encoding="utf-8-sig") == "keep"


def test_canonical_guard_rejects_unsafe_roots_aliases_and_directory_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from hermes_cli import data_cleanup

    user_home = tmp_path / "user"
    normal_home = user_home / ".hermes"
    project = normal_home / "hermes-agent"
    project.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: user_home)

    assert data_cleanup.guard_data_removal_home(normal_home, project) == normal_home.resolve()

    unsafe = [Path(tmp_path.anchor), user_home, tmp_path, project]
    linked_home = tmp_path / "linked-home"
    try:
        linked_home.symlink_to(normal_home, target_is_directory=True)
    except OSError:
        pass
    else:
        unsafe.append(linked_home)

    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(user_home, target_is_directory=True)
    except OSError:
        pass
    else:
        unsafe.append(linked_parent / ".hermes")

    for candidate in unsafe:
        with pytest.raises(ValueError):
            data_cleanup.guard_data_removal_home(candidate, project)

    junction_home = tmp_path / "junction-home"
    junction_home.mkdir()
    monkeypatch.setattr(data_cleanup, "is_junction", lambda path: path == junction_home)
    with pytest.raises(ValueError):
        data_cleanup.guard_data_removal_home(junction_home, project)
