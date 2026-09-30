"""The installer stage and later desktop product build are separate processes."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _resolved_git(*command: str) -> SimpleNamespace:
    """Stand-in for ``locate_command(...)`` shaped like its ``Resolution.command``."""
    return SimpleNamespace(command=command)


@pytest.mark.platforms("windows")
def test_packaged_desktop_build_restores_pm_git_for_stamp_and_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pm
    from hermes_cli import main_desktop

    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    original = {"PATH": "C:\\without-git", "HERMES_HOME": str(tmp_path)}
    calls: list[tuple[list[str], dict[str, str]]] = []

    def installed_git(*names: str, base_env: dict[str, str]) -> dict[str, str]:
        assert names == ("git",)
        assert base_env == original
        return {**base_env, "PATH": "C:\\pm-pinned-git\\cmd;" + base_env["PATH"]}

    def run(command: list[str], *, env: dict[str, str], **_kwargs: object) -> None:
        calls.append((command, env))

    # The CI host ships its own Git for Windows on PATH; this PATH has none.
    monkeypatch.setattr("hermes_platform.resolver.locate_command", lambda _name: _resolved_git())
    monkeypatch.setattr(pm, "ensure", lambda *names, base_env: SimpleNamespace(env=installed_git(*names, base_env=base_env)))
    monkeypatch.setattr(main_desktop.subprocess, "run", run)
    monkeypatch.setattr(main_desktop, "_promote_staged_desktop_app", lambda *_args: desktop / "Hermes.exe")
    main_desktop.build_prepared_desktop(desktop, source_mode=False, npm="C:\\node\\npm.cmd", env=original)

    assert [command[2] for command, _ in calls] == ["build", "builder"]
    assert all(env["PATH"].startswith("C:\\pm-pinned-git\\cmd;") for _, env in calls)
    assert original["PATH"] == "C:\\without-git"


@pytest.mark.platforms("windows")
def test_packaged_desktop_build_keeps_a_system_git_and_skips_the_store_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pm
    from hermes_cli import main_desktop

    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    original = {"PATH": r"C:\Program Files\Git\cmd;" + str(tmp_path), "HERMES_HOME": str(tmp_path)}
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(command: list[str], *, env: dict[str, str], **_kwargs: object) -> None:
        calls.append((command, env))

    def unexpected_install(*_names: str, **_kwargs: object) -> None:
        raise AssertionError("a PATH-resolvable system git must not re-enter pm.ensure")

    store = (tmp_path / "pm-store").resolve()
    monkeypatch.setattr("pm.paths.store_root", lambda: store)
    monkeypatch.setattr(
        "hermes_platform.resolver.locate_command",
        lambda _name: _resolved_git(r"C:\Program Files\Git\cmd\git.exe"),
    )
    monkeypatch.setattr(pm, "ensure", unexpected_install)
    monkeypatch.setattr(main_desktop.subprocess, "run", run)
    monkeypatch.setattr(main_desktop, "_promote_staged_desktop_app", lambda *_args: desktop / "Hermes.exe")
    main_desktop.build_prepared_desktop(desktop, source_mode=False, npm="C:\\node\\npm.cmd", env=original)

    assert [command[2] for command, _ in calls] == ["build", "builder"]
    assert all(env == original for _, env in calls)


@pytest.mark.platforms("windows")
def test_packaged_desktop_build_still_ensures_an_unrecorded_store_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pm
    from hermes_cli import main_desktop

    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    original = {"PATH": "C:\\staged-by-installer", "HERMES_HOME": str(tmp_path)}
    calls: list[tuple[list[str], dict[str, str]]] = []
    ensured: list[tuple[str, ...]] = []

    def installed_git(*names: str, base_env: dict[str, str]) -> dict[str, str]:
        return {**base_env, "PATH": "C:\\pm-recorded-git\\cmd;" + base_env["PATH"]}

    def run(command: list[str], *, env: dict[str, str], **_kwargs: object) -> None:
        calls.append((command, env))

    # install.ps1 stages pinned Git without recording it; a git found under the
    # store is that copy, so the build still routes through pm.ensure.
    store = (tmp_path / "pm-store").resolve()
    staged = store / "git-2.53.0+3-win32-x64" / "cmd" / "git.exe"
    monkeypatch.setattr("pm.paths.store_root", lambda: store)
    monkeypatch.setattr(
        "hermes_platform.resolver.locate_command", lambda _name: _resolved_git(str(staged))
    )
    monkeypatch.setattr(
        pm, "ensure",
        lambda *names, base_env: ensured.append(names) or SimpleNamespace(env=installed_git(*names, base_env=base_env)),
    )
    monkeypatch.setattr(main_desktop.subprocess, "run", run)
    monkeypatch.setattr(main_desktop, "_promote_staged_desktop_app", lambda *_args: desktop / "Hermes.exe")
    main_desktop.build_prepared_desktop(desktop, source_mode=False, npm="C:\\node\\npm.cmd", env=original)

    assert ensured == [("git",)]
    assert [command[2] for command, _ in calls] == ["build", "builder"]
    assert all(env["PATH"].startswith("C:\\pm-recorded-git\\cmd;") for _, env in calls)
