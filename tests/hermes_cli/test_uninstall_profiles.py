"""Full uninstall keeps named profiles unless the user explicitly opts in."""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gui_uninstall as gui_uninstall
import hermes_cli.uninstall as uninstall


@pytest.fixture
def profile_install(tmp_path, monkeypatch):
    """An isolated default home; no uninstall operation may escape ``tmp_path``."""
    user_home = tmp_path / "user"
    hermes_home = user_home / ".hermes"
    project_root = tmp_path / "checkout"
    project_root.mkdir(parents=True)
    (hermes_home / "config.yaml").parent.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text("default: true\n", encoding="utf-8")
    (hermes_home / "sessions" / "default.json").parent.mkdir()
    (hermes_home / "sessions" / "default.json").write_text("default", encoding="utf-8")

    profiles = []
    aliases = []
    for name in ("work", "personal"):
        profile_home = hermes_home / "profiles" / name
        witness = profile_home / "nested" / "keep.txt"
        witness.parent.mkdir(parents=True)
        witness.write_text(f"{name} data", encoding="utf-8")
        alias = user_home / ".local" / "bin" / name
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text("profile alias", encoding="utf-8")
        aliases.append(alias)
        profiles.append(SimpleNamespace(
            name=name, path=profile_home, alias_path=alias, gateway_running=False,
        ))

    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(uninstall, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: profiles)
    monkeypatch.setattr(uninstall, "_refuse_if_steward_owned", lambda: None)
    monkeypatch.setattr(uninstall, "_is_windows", lambda: False)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: True)
    for name in (
        "remove_path_from_shell_configs",
        "remove_wrapper_script",
        "remove_node_symlinks",
        "remove_legacy_runtime_trees",
        "remove_desktop_app_leftovers",
        "remove_dashboard_launchd_jobs",
        "_macos_cache_leftover_dirs",
    ):
        monkeypatch.setattr(uninstall, name, lambda *args, **kwargs: [])
    monkeypatch.setattr(gui_uninstall, "uninstall_gui", lambda *args, **kwargs: False)
    monkeypatch.setattr(gui_uninstall, "desktop_userdata_dir", lambda: tmp_path / "desktop-userdata")

    real_rmtree = shutil.rmtree
    violations = []

    def confined_rmtree(path, *args, **kwargs):
        if not Path(path).is_relative_to(tmp_path):
            violations.append(Path(path))
            raise AssertionError(f"rmtree escaped isolated home: {path}")
        return real_rmtree(path, *args, **kwargs)

    service_calls = []
    monkeypatch.setattr(shutil, "rmtree", confined_rmtree)
    monkeypatch.setattr(
        uninstall.subprocess,
        "run",
        lambda command, **kwargs: service_calls.append(command),
    )
    return SimpleNamespace(
        home=hermes_home,
        project=project_root,
        profiles=profiles,
        aliases=aliases,
        service_calls=service_calls,
        violations=violations,
    )


@pytest.mark.parametrize("noninteractive", [False, True], ids=["explicit-decline", "yes-full"])
def test_full_uninstall_preserves_named_profiles_without_opt_in(
    profile_install, monkeypatch, noninteractive,
):
    install = profile_install
    if noninteractive:
        monkeypatch.setattr(
            "builtins.input",
            lambda *_args: pytest.fail("--yes --full must remain noninteractive"),
        )
    else:
        answers = iter(["2", "no", "yes"])
        monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    uninstall.run_uninstall(SimpleNamespace(dry_run=False, yes=noninteractive, full=True))

    assert install.violations == []
    assert not install.project.exists()
    assert not (install.home / "config.yaml").exists()
    assert not (install.home / "sessions").exists()
    assert set(install.home.iterdir()) == {install.home / "profiles"}
    assert [
        (profile.path / "nested" / "keep.txt").read_text(encoding="utf-8-sig")
        for profile in install.profiles
    ] == ["work data", "personal data"]
    assert all(alias.read_text(encoding="utf-8-sig") == "profile alias" for alias in install.aliases)
    assert install.service_calls == []


def test_full_uninstall_reports_partial_default_cleanup(
    profile_install, monkeypatch, capsys,
):
    install = profile_install
    blocked = install.home / "sessions"
    confined_rmtree = shutil.rmtree

    def refuse_one_default_child(path, *args, **kwargs):
        if Path(path) == blocked:
            raise PermissionError("fixture holds sessions")
        return confined_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", refuse_one_default_child)

    with pytest.raises(SystemExit) as failure:
        uninstall.run_uninstall(SimpleNamespace(dry_run=False, yes=True, full=True))

    assert failure.value.code != 0
    assert blocked.is_dir()
    assert all((profile.path / "nested" / "keep.txt").is_file() for profile in install.profiles)
    output = capsys.readouterr().out
    assert "Default-profile data was only partially removed" in output
    assert "Uninstall Complete" not in output


def test_full_uninstall_profile_opt_in_cleans_services_aliases_and_data(
    profile_install, monkeypatch,
):
    install = profile_install
    answers = iter(["2", "yes", "yes"])
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    uninstall.run_uninstall(SimpleNamespace(dry_run=False, yes=False, full=False))

    assert install.violations == []
    assert not install.home.exists()
    assert not install.project.exists()
    assert all(not profile.path.exists() for profile in install.profiles)
    assert all(not alias.exists() for alias in install.aliases)
    assert install.service_calls == [
        [uninstall.sys.executable, "-m", "hermes_cli.main", "--profile", profile.name,
         "gateway", subcommand]
        for profile in install.profiles
        for subcommand in ("stop", "uninstall")
    ]


def _refuse_rmtree_of(monkeypatch, blocked: Path) -> None:
    confined_rmtree = shutil.rmtree

    def refuse(path, *args, **kwargs):
        if Path(path) == blocked:
            raise PermissionError("[WinError 5] Access is denied (fixture)")
        return confined_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", refuse)


def test_full_uninstall_with_profiles_opted_in_reports_failed_home_removal(
    profile_install, monkeypatch, capsys,
):
    """The sibling branch (whole home, profiles included) must not claim success either."""
    install = profile_install
    _refuse_rmtree_of(monkeypatch, install.home)
    answers = iter(["2", "yes", "yes"])
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    with pytest.raises(SystemExit) as failure:
        uninstall.run_uninstall(SimpleNamespace(dry_run=False, yes=False, full=False))

    assert failure.value.code != 0
    assert (install.home / "config.yaml").is_file()
    output = capsys.readouterr().out
    assert "Uninstall did not finish" in output
    assert "Uninstall Complete" not in output


def test_full_uninstall_of_named_profile_home_reports_failed_removal(
    profile_install, monkeypatch, capsys,
):
    """``hermes -p work uninstall --full`` erases a non-default home via the same branch."""
    install = profile_install
    work_home = install.profiles[0].path
    monkeypatch.setenv("HERMES_HOME", str(work_home))
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: work_home)
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: [])
    _refuse_rmtree_of(monkeypatch, work_home)

    with pytest.raises(SystemExit) as failure:
        uninstall.run_uninstall(SimpleNamespace(dry_run=False, yes=True, full=True))

    assert failure.value.code != 0
    assert (work_home / "nested" / "keep.txt").is_file()
    output = capsys.readouterr().out
    assert "Uninstall did not finish" in output
    assert "Uninstall Complete" not in output


@pytest.mark.parametrize("windows", [False, True])
def test_rmtree_step_clears_read_only_bits_on_windows(tmp_path, monkeypatch, windows):
    """Git objects are read-only; on Windows plain rmtree raises WinError 5 on them."""
    target = tmp_path / "home"
    (target / ".git" / "objects").mkdir(parents=True)
    (target / ".git" / "objects" / "pack").write_text("ro", encoding="utf-8")
    real_rmtree = shutil.rmtree

    def windows_like_rmtree(path, *args, onerror=None, **kwargs):
        if onerror is None:
            raise PermissionError("[WinError 5] Access is denied (read-only git object)")
        return real_rmtree(path, *args, onerror=onerror, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", windows_like_rmtree)
    monkeypatch.setattr(uninstall, "_is_windows", lambda: windows)

    assert uninstall._rmtree_step(target) is windows
    assert target.exists() is not windows
