"""The installer stage and later desktop product build are separate processes."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


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

    monkeypatch.setattr(pm, "ensure", lambda *names, base_env: SimpleNamespace(env=installed_git(*names, base_env=base_env)))
    monkeypatch.setattr(main_desktop.subprocess, "run", run)
    monkeypatch.setattr(main_desktop, "_promote_staged_desktop_app", lambda *_args: desktop / "Hermes.exe")
    main_desktop.build_prepared_desktop(desktop, source_mode=False, npm="C:\\node\\npm.cmd", env=original)

    assert [command[2] for command, _ in calls] == ["build", "builder"]
    assert all(env["PATH"].startswith("C:\\pm-pinned-git\\cmd;") for _, env in calls)
    assert original["PATH"] == "C:\\without-git"


@pytest.mark.platforms("windows")
def test_packaged_desktop_build_keeps_system_git_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import main_desktop
    from hermes_platform.resolver import Candidate, Resolution

    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    original = {"PATH": "C:\\system-git\\cmd", "HERMES_HOME": str(tmp_path)}
    calls: list[dict[str, str]] = []

    monkeypatch.setattr(
        "hermes_platform.resolver.locate_command",
        lambda *_args, **_kwargs: Resolution(
            "path_executable", (Candidate("C:\\system-git\\cmd\\git.exe", "PATH", True),)
        ),
    )
    monkeypatch.setattr(main_desktop.subprocess, "run", lambda _command, *, env, **_kwargs: calls.append(env))
    monkeypatch.setattr(main_desktop, "_promote_staged_desktop_app", lambda *_args: desktop / "Hermes.exe")

    main_desktop.build_prepared_desktop(desktop, source_mode=False, npm="npm.cmd", env=original)

    assert len(calls) == 2
    assert all(env["PATH"] == original["PATH"] for env in calls)


@pytest.mark.platforms("windows")
def test_packaged_desktop_build_reacquires_store_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pm
    from hermes_cli import main_desktop
    from hermes_platform.resolver import Candidate, Resolution

    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    original = {"PATH": "C:\\portable-git\\cmd", "HERMES_HOME": str(tmp_path)}
    calls: list[dict[str, str]] = []
    ensures: list[tuple[str, dict[str, str]]] = []
    store = tmp_path / "sealed" / "tools"
    writable_store = tmp_path / "writable" / "tools"

    monkeypatch.setattr("pm.paths.store_root", lambda: store)
    monkeypatch.setattr("pm.paths.writable_store_root", lambda: writable_store)
    monkeypatch.setattr(
        "hermes_platform.resolver.locate_command",
        lambda *_args, **_kwargs: Resolution(
            "path_executable", (Candidate(str(store / "PortableGit" / "cmd" / "git.exe"), "PATH", True),)
        ),
    )
    monkeypatch.setattr(
        pm,
        "ensure",
        lambda name, *, base_env: ensures.append((name, base_env)) or SimpleNamespace(env=base_env),
    )
    monkeypatch.setattr(main_desktop.subprocess, "run", lambda _command, *, env, **_kwargs: calls.append(env))
    monkeypatch.setattr(main_desktop, "_promote_staged_desktop_app", lambda *_args: desktop / "Hermes.exe")

    main_desktop.build_prepared_desktop(desktop, source_mode=False, npm="npm.cmd", env=original)

    assert ensures == [("git", original)]
    assert len(calls) == 2
