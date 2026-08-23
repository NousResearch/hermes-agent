"""Uninstall must not leave a dangling ``hermes`` command on Windows.

Every uninstall mode deletes the code checkout, but the launcher copies
staged onto PATH in the managed binary dir (the default Hermes root's
``bin``) live outside it. A surviving launcher makes ``hermes`` in a new
terminal resolve and then error on its missing venv target — worse than
command-not-found. The dir is wholly hermes-owned (pm keeps uv in its own
store entry), so its owning checkout's sweep takes everything in it.
Unrelated source checkouts must preserve these launchers.

Pure cleanup tests inject platform verdicts. Orchestration and native path
comparisons run on Windows without faking the host.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import gui_uninstall, uninstall


@pytest.fixture
def managed_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Default-root ``bin`` holding staged launcher copies."""
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "hermes.exe").write_bytes(b"MZ launcher")
    (bin_dir / "hermes-acp.cmd").write_text("@echo off\r\n", encoding="ascii")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return bin_dir


@pytest.mark.parametrize("checkout_name", ["hermes-agent", "renamed-checkout"])
def test_removes_the_whole_managed_bin_dir(managed_bin: Path, checkout_name: str):
    removed = uninstall.remove_windows_bin_launchers(
        managed_bin.parent / checkout_name, windows=True
    )

    assert sorted(p.name for p in removed) == ["hermes-acp.cmd", "hermes.exe"]
    assert not managed_bin.exists()


def test_anchors_on_default_root_not_profile_home(
    managed_bin: Path, monkeypatch: pytest.MonkeyPatch
):
    """The launcher dir is per-machine; a profile HERMES_HOME must not
    redirect the sweep into ``profiles/<name>/bin``."""
    home = managed_bin.parent
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "work"))

    removed = uninstall.remove_windows_bin_launchers(
        home / "hermes-agent", windows=True
    )

    assert sorted(p.name for p in removed) == ["hermes-acp.cmd", "hermes.exe"]
    assert not (managed_bin / "hermes.exe").exists()


def test_noop_on_posix(managed_bin: Path):
    assert (
        uninstall.remove_windows_bin_launchers(
            managed_bin.parent / "hermes-agent", windows=False
        )
        == []
    )
    assert (managed_bin / "hermes.exe").exists()


def test_noop_when_no_bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert (
        uninstall.remove_windows_bin_launchers(home / "hermes-agent", windows=True)
        == []
    )


def test_unrelated_checkout_preserves_managed_bin(
    managed_bin: Path, tmp_path: Path, capsys
):
    assert (
        uninstall.remove_windows_bin_launchers(
            tmp_path / "src" / "hermes-agent", windows=True
        )
        == []
    )
    assert (managed_bin / "hermes.exe").read_bytes() == b"MZ launcher"
    assert (managed_bin / "hermes-acp.cmd").exists()
    assert "this checkout does not own" in capsys.readouterr().out


@pytest.mark.platforms("windows")
def test_case_and_separator_alias_matches_launcher_creation(managed_bin: Path):
    root = Path(str(managed_bin.parent / "renamed-checkout").upper().replace("\\", "/"))

    assert len(uninstall.remove_windows_bin_launchers(root)) == 2
    assert not managed_bin.exists()


def test_locked_trampoline_is_renamed_aside(
    managed_bin: Path, monkeypatch: pytest.MonkeyPatch
):
    trampoline = managed_bin / "hermes.exe"
    unlink = Path.unlink

    def locked_unlink(path, *args, **kwargs):
        if path == trampoline:
            raise PermissionError("running trampoline")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_unlink)

    removed = uninstall.remove_windows_bin_launchers(
        managed_bin.parent / "hermes-agent", windows=True
    )

    assert trampoline in removed
    assert not trampoline.exists()
    assert len(list(managed_bin.glob("hermes.exe.uninstalled.*"))) == 1
    assert not (managed_bin / "hermes-acp.cmd").exists()


@pytest.fixture
def isolated_uninstall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise orchestration with real temporary files, never workstation services."""
    path_calls: list[bool] = []

    def no_subprocess(*args, **kwargs):
        pytest.fail("Uninstall test attempted to operate a real subprocess")

    monkeypatch.setattr(uninstall.subprocess, "run", no_subprocess)
    monkeypatch.setattr(uninstall.subprocess, "Popen", no_subprocess)
    for name in (
        "uninstall_gateway_service",
        "remove_path_from_shell_configs",
        "remove_hermes_env_vars_windows",
        "remove_wrapper_script",
    ):
        monkeypatch.setattr(uninstall, name, list)
    for name in (
        "remove_node_symlinks",
        "remove_portable_tooling_windows",
        "remove_legacy_runtime_trees",
    ):
        monkeypatch.setattr(uninstall, name, lambda _home: [])
    monkeypatch.setattr(uninstall, "remove_desktop_app_leftovers", lambda **kwargs: [])
    monkeypatch.setattr(uninstall, "_xdg_leftover_paths", lambda **kwargs: ([], []))
    monkeypatch.setattr(gui_uninstall, "uninstall_gui", lambda _home, **kwargs: [])
    monkeypatch.setattr(
        gui_uninstall, "desktop_userdata_dir", lambda: tmp_path / "desktop-userdata"
    )

    def remove_path(_home, *, include_managed_bin=False):
        path_calls.append(include_managed_bin)
        return []

    monkeypatch.setattr(uninstall, "remove_path_from_windows_registry", remove_path)
    return path_calls


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("managed,full", [(False, False), (True, False), (True, True)])
def test_uninstall_only_removes_owned_launchers(
    managed_bin: Path,
    tmp_path: Path,
    isolated_uninstall: list[bool],
    managed: bool,
    full: bool,
):
    home = managed_bin.parent
    project_root = (home if managed else tmp_path / "src") / "hermes-agent"
    project_root.mkdir(parents=True)
    config = home / "config.yaml"
    config.write_text("model: test\n", encoding="utf-8")

    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=home,
        full_uninstall=full,
        remove_profiles=False,
        named_profiles=[],
    )

    assert not project_root.exists()
    assert managed_bin.exists() is not managed
    assert config.exists() is not full
    assert isolated_uninstall == [full]
