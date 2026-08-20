"""Full uninstall must honor remove_profiles when wiping HERMES_HOME."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import uninstall


def _seed_default_home(hermes_home: Path) -> Path:
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (hermes_home / "sessions").mkdir()
    (hermes_home / ".env").write_text("KEY=secret\n", encoding="utf-8")
    profile_home = hermes_home / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("profile: coder\n", encoding="utf-8")
    return profile_home


def test_remove_hermes_home_preserves_profiles_subtree(tmp_path):
    hermes_home = tmp_path / ".hermes"
    profile_home = _seed_default_home(hermes_home)

    uninstall._remove_hermes_home(hermes_home, preserve_profiles=True)

    assert hermes_home.is_dir()
    assert profile_home.is_dir()
    assert (profile_home / "config.yaml").read_text(encoding="utf-8") == "profile: coder\n"
    assert not (hermes_home / "config.yaml").exists()
    assert not (hermes_home / "sessions").exists()
    assert not (hermes_home / ".env").exists()


def test_remove_hermes_home_wipes_everything_when_not_preserving(tmp_path):
    hermes_home = tmp_path / ".hermes"
    _seed_default_home(hermes_home)

    uninstall._remove_hermes_home(hermes_home, preserve_profiles=False)

    assert not hermes_home.exists()


def test_perform_uninstall_full_without_remove_profiles_keeps_data(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "repo"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    profile_home = _seed_default_home(hermes_home)
    removed_profiles: list[str] = []

    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda _home: [])
    monkeypatch.setattr(uninstall, "_is_windows", lambda: False)
    monkeypatch.setattr(
        uninstall,
        "_uninstall_profile",
        lambda profile: removed_profiles.append(profile.name),
    )

    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=hermes_home,
        full_uninstall=True,
        remove_profiles=False,
        named_profiles=[
            SimpleNamespace(name="coder", path=profile_home, alias_path=None)
        ],
    )

    assert removed_profiles == []
    assert not project_root.exists()
    assert profile_home.is_dir()
    assert (profile_home / "config.yaml").read_text(encoding="utf-8") == "profile: coder\n"
    assert not (hermes_home / "config.yaml").exists()


def test_perform_uninstall_full_with_remove_profiles_wipes_home(
    monkeypatch, tmp_path
):
    project_root = tmp_path / "repo"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    profile_home = _seed_default_home(hermes_home)
    removed_profiles: list[str] = []

    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda _home: [])
    monkeypatch.setattr(uninstall, "_is_windows", lambda: False)

    def _fake_uninstall_profile(profile):
        removed_profiles.append(profile.name)
        # Mirror production: wipe the profile home before step 5b.
        if profile.path.exists():
            shutil.rmtree(profile.path)

    monkeypatch.setattr(uninstall, "_uninstall_profile", _fake_uninstall_profile)

    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=hermes_home,
        full_uninstall=True,
        remove_profiles=True,
        named_profiles=[
            SimpleNamespace(name="coder", path=profile_home, alias_path=None)
        ],
    )

    assert removed_profiles == ["coder"]
    assert not project_root.exists()
    assert not hermes_home.exists()


def test_yes_full_uninstall_does_not_wipe_named_profiles(monkeypatch, tmp_path):
    """Non-interactive --yes --full claims not to remove named profiles."""
    project_root = tmp_path / "repo"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    profile_home = _seed_default_home(hermes_home)

    monkeypatch.setattr(uninstall, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall, "_is_default_hermes_home", lambda _home: True)
    monkeypatch.setattr(
        uninstall,
        "_discover_named_profiles",
        lambda: [SimpleNamespace(name="coder", path=profile_home, alias_path=None)],
    )
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda _home: [])
    monkeypatch.setattr(uninstall, "_is_windows", lambda: False)

    uninstall.run_uninstall(SimpleNamespace(yes=True, full=True, dry_run=False, gui=False))

    assert profile_home.is_dir()
    assert (profile_home / "config.yaml").read_text(encoding="utf-8") == "profile: coder\n"
    assert not (hermes_home / "config.yaml").exists()
