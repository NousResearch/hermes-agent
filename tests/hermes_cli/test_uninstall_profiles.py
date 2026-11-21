"""Full uninstall keeps named profiles unless the user opts in, and never reports a surviving home as removed."""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gui_uninstall, uninstall


@pytest.fixture
def profile_install(tmp_path, monkeypatch):
    """An isolated default home with two named profiles; no removal may escape ``tmp_path``."""
    user_home = tmp_path / "user"
    hermes_home = user_home / ".hermes"
    project_root = tmp_path / "checkout"
    (project_root / ".git").mkdir(parents=True)
    (hermes_home / "sessions").mkdir(parents=True)
    (hermes_home / "config.yaml").write_text("default: true\n", encoding="utf-8")
    (hermes_home / "sessions" / "default.json").write_text("default", encoding="utf-8")

    profiles = []
    for name in ("work", "personal"):
        profile_home = hermes_home / "profiles" / name
        witness = profile_home / "nested" / "keep.txt"
        witness.parent.mkdir(parents=True)
        witness.write_text(f"{name} data", encoding="utf-8")
        alias = user_home / ".local" / "bin" / name
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("profile alias", encoding="utf-8")
        profiles.append(SimpleNamespace(name=name, path=profile_home, alias_path=alias, gateway_running=False))

    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(uninstall, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: profiles)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: True)
    for name in (
        "remove_path_from_shell_configs", "remove_wrapper_script", "remove_node_symlinks",
        "remove_legacy_runtime_trees", "remove_desktop_app_leftovers", "remove_dashboard_launchd_jobs",
        "_macos_cache_leftover_dirs", "remove_path_from_windows_registry", "remove_hermes_env_vars_windows",
        "remove_windows_bin_launchers", "remove_portable_tooling_windows",
    ):
        monkeypatch.setattr(uninstall, name, lambda *args, **kwargs: [])
    monkeypatch.setattr(gui_uninstall, "uninstall_gui", lambda *args, **kwargs: False)
    monkeypatch.setattr(gui_uninstall, "desktop_userdata_dir", lambda: tmp_path / "desktop-userdata")
    service_calls = []
    monkeypatch.setattr(uninstall.subprocess, "run", lambda command, **kwargs: service_calls.append(command))
    return SimpleNamespace(home=hermes_home, project=project_root, profiles=profiles, service_calls=service_calls)


@pytest.mark.parametrize("noninteractive", [False, True], ids=["explicit-decline", "yes-full"])
def test_full_uninstall_preserves_named_profiles_without_opt_in(profile_install, monkeypatch, noninteractive):
    install = profile_install
    if noninteractive:
        monkeypatch.setattr("builtins.input", lambda *_args: pytest.fail("--yes --full must stay noninteractive"))
    else:
        answers = iter(["2", "no", "yes"])
        monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    uninstall.run_uninstall(SimpleNamespace(dry_run=False, yes=noninteractive, full=True))

    assert not install.project.exists()
    assert set(install.home.iterdir()) == {install.home / "profiles"}
    assert [(p.path / "nested" / "keep.txt").read_text(encoding="utf-8-sig") for p in install.profiles] == [
        "work data", "personal data"]
    assert all(p.alias_path.is_file() for p in install.profiles)
    assert install.service_calls == []


@pytest.mark.parametrize("branch", ["profiles-kept", "profiles-opted-in"])
def test_full_uninstall_fails_when_the_home_survives(profile_install, monkeypatch, capsys, branch):
    """Both erase branches: a surviving default-profile child, or a whole home that could not go."""
    install = profile_install
    blocked = install.home / "sessions" if branch == "profiles-kept" else install.home
    real_rmtree = shutil.rmtree

    def refuse_blocked(path, *args, **kwargs):
        if Path(path) == blocked:
            raise PermissionError("locked by another process (fixture)")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", refuse_blocked)
    answers = iter(["2", "yes" if branch == "profiles-opted-in" else "no", "yes"])
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    with pytest.raises(SystemExit) as failure:
        uninstall.run_uninstall(SimpleNamespace(dry_run=False, yes=False, full=False))

    assert failure.value.code == 1
    assert blocked.is_dir()
    output = capsys.readouterr().out
    assert "Uninstall did not finish" in output
    assert "Uninstall Complete" not in output


@pytest.mark.platforms("posix")
def test_full_uninstall_keeping_profiles_removes_a_stale_gateway_socket(profile_install, monkeypatch):
    """A crashed gateway leaves its socket behind: neither file nor directory, it must still be unlinked."""
    import socket

    install = profile_install
    monkeypatch.chdir(install.home)  # a relative bind path stays under the AF_UNIX length cap
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind("gateway.sock")
    monkeypatch.chdir(install.home.parent)
    monkeypatch.setattr("builtins.input", lambda *_args: pytest.fail("--yes --full must stay noninteractive"))

    uninstall.run_uninstall(SimpleNamespace(dry_run=False, yes=True, full=True))

    assert set(install.home.iterdir()) == {install.home / "profiles"}


@pytest.mark.platforms("windows")
def test_rmtree_step_removes_read_only_git_objects(tmp_path):
    """install.ps1 puts the git checkout, whose objects are read-only, inside HERMES_HOME."""
    target = tmp_path / "home"
    obj = target / ".git" / "objects" / "pack"
    obj.parent.mkdir(parents=True)
    obj.write_text("ro", encoding="utf-8")
    obj.chmod(0o444)

    assert uninstall._rmtree_step(target) is True
    assert not target.exists()
