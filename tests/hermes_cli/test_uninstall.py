from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import uninstall as uninstall_mod


def test_full_uninstall_also_removes_named_profiles(monkeypatch, tmp_path):
    project_root = tmp_path / "repo"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    hermes_home.mkdir()

    profiles = [
        SimpleNamespace(name="coder", path=tmp_path / ".hermes" / "profiles" / "coder"),
        SimpleNamespace(name="writer", path=tmp_path / ".hermes" / "profiles" / "writer"),
    ]
    removed_profiles: list[str] = []
    removed_paths: list[Path] = []
    prompts: list[str] = []
    answers = iter(["2", "yes"])

    monkeypatch.setattr(uninstall_mod, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall_mod, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall_mod, "_is_default_hermes_home", lambda _home: True)
    monkeypatch.setattr(uninstall_mod, "_discover_named_profiles", lambda: profiles)
    monkeypatch.setattr(uninstall_mod, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall_mod, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall_mod, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall_mod, "_is_windows", lambda: False)
    monkeypatch.setattr(uninstall_mod, "_uninstall_profile", lambda profile: removed_profiles.append(profile.name))
    monkeypatch.setattr(uninstall_mod.shutil, "rmtree", lambda path: removed_paths.append(Path(path)))
    monkeypatch.setattr(
        builtins,
        "input",
        lambda prompt="": prompts.append(prompt) or next(answers),
    )

    uninstall_mod.run_uninstall(SimpleNamespace())

    assert removed_profiles == ["coder", "writer"]
    assert project_root in removed_paths
    assert hermes_home in removed_paths
    assert len(prompts) == 2


def test_keep_data_does_not_touch_named_profiles(monkeypatch, tmp_path):
    project_root = tmp_path / "repo"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    hermes_home.mkdir()

    profiles = [SimpleNamespace(name="coder", path=tmp_path / ".hermes" / "profiles" / "coder")]
    removed_profiles: list[str] = []
    removed_paths: list[Path] = []
    answers = iter(["1", "yes"])

    monkeypatch.setattr(uninstall_mod, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall_mod, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall_mod, "_is_default_hermes_home", lambda _home: True)
    monkeypatch.setattr(uninstall_mod, "_discover_named_profiles", lambda: profiles)
    monkeypatch.setattr(uninstall_mod, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall_mod, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall_mod, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall_mod, "_is_windows", lambda: False)
    monkeypatch.setattr(uninstall_mod, "_uninstall_profile", lambda profile: removed_profiles.append(profile.name))
    monkeypatch.setattr(uninstall_mod.shutil, "rmtree", lambda path: removed_paths.append(Path(path)))
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    uninstall_mod.run_uninstall(SimpleNamespace())

    assert removed_profiles == []
    assert project_root in removed_paths
    assert hermes_home not in removed_paths
