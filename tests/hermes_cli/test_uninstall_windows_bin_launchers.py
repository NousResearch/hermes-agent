"""Uninstall must not leave a dangling ``hermes`` command on Windows.

Every uninstall mode deletes the code checkout, but the launchers install.ps1
staged in the managed binary dir (the default Hermes root's ``bin``, shared
with the managed uv) live outside it. A surviving launcher makes ``hermes``
in a new terminal resolve and then error on its missing venv target — worse
than command-not-found. The managed uv next to them must survive keep-data
uninstalls, so the PATH sweep takes the ``bin`` entry only on a full wipe.

Platform verdicts are injected parameters (input→output, not host fakes).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import gui_uninstall, uninstall
from hermes_cli._install_repair import _WINDOWS_BIN_LAUNCHERS


@pytest.fixture
def managed_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Default-root ``bin`` holding launchers of both forms plus managed uv."""
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "hermes.exe").write_bytes(b"MZ launcher")
    (bin_dir / "hermes-acp.cmd").write_text("@echo off\r\n", encoding="ascii")
    (bin_dir / "uv.exe").write_bytes(b"MZ managed uv")
    (bin_dir / "uvx.exe").write_bytes(b"MZ managed uvx")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return bin_dir


def test_removes_both_launcher_forms_and_keeps_managed_uv(managed_bin: Path):
    removed = uninstall.remove_windows_bin_launchers(
        managed_bin.parent / "hermes-agent", windows=True
    )

    assert sorted(p.name for p in removed) == ["hermes-acp.cmd", "hermes.exe"]
    assert not (managed_bin / "hermes.exe").exists()
    assert not (managed_bin / "hermes-acp.cmd").exists()
    # The managed uv stays — keep-data reinstalls still need it.
    assert (managed_bin / "uv.exe").exists()
    assert (managed_bin / "uvx.exe").exists()


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


def test_noop_when_no_launchers_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert (
        uninstall.remove_windows_bin_launchers(home / "hermes-agent", windows=True)
        == []
    )


def test_unrelated_project_cannot_remove_managed_launchers(managed_bin: Path):
    unrelated_root = managed_bin.parent.parent / "source-checkout"

    assert uninstall.remove_windows_bin_launchers(unrelated_root, windows=True) == []
    assert (managed_bin / "hermes.exe").exists()
    assert (managed_bin / "hermes-acp.cmd").exists()
    assert (managed_bin / "uv.exe").exists()


@pytest.fixture
def isolated_windows_uninstall(monkeypatch: pytest.MonkeyPatch):
    """Keep _perform_uninstall focused on launcher/PATH ownership."""
    path_calls: list[bool] = []

    monkeypatch.setattr(uninstall, "_is_windows", lambda: True)
    monkeypatch.setattr(uninstall, "uninstall_gateway_service", lambda: False)
    monkeypatch.setattr(uninstall, "remove_path_from_shell_configs", lambda: [])
    monkeypatch.setattr(
        uninstall,
        "remove_path_from_windows_registry",
        lambda _home, *, include_managed_bin=False: (
            path_calls.append(include_managed_bin) or []
        ),
    )
    monkeypatch.setattr(uninstall, "remove_hermes_env_vars_windows", lambda: [])
    monkeypatch.setattr(uninstall, "remove_wrapper_script", lambda: [])
    monkeypatch.setattr(uninstall, "remove_node_symlinks", lambda _home: [])
    monkeypatch.setattr(uninstall, "remove_portable_tooling_windows", lambda _home: [])
    monkeypatch.setattr(gui_uninstall, "uninstall_gui", lambda _home: [])
    return path_calls


def test_unrelated_keep_data_uninstall_preserves_managed_bin(
    managed_bin: Path,
    tmp_path: Path,
    isolated_windows_uninstall: list[bool],
):
    unrelated_root = tmp_path / "source-checkout" / "hermes-agent"
    unrelated_root.mkdir(parents=True)

    uninstall._perform_uninstall(
        project_root=unrelated_root,
        hermes_home=managed_bin.parent,
        full_uninstall=False,
        remove_profiles=False,
        named_profiles=[],
    )

    assert not unrelated_root.exists()
    assert (managed_bin / "hermes.exe").exists()
    assert (managed_bin / "hermes-acp.cmd").exists()
    assert (managed_bin / "uv.exe").exists()
    assert isolated_windows_uninstall == [False]


def test_managed_keep_data_uninstall_removes_only_owned_launchers(
    managed_bin: Path,
    isolated_windows_uninstall: list[bool],
):
    project_root = managed_bin.parent / "hermes-agent"
    project_root.mkdir()

    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=managed_bin.parent,
        full_uninstall=False,
        remove_profiles=False,
        named_profiles=[],
    )

    assert not (managed_bin / "hermes.exe").exists()
    assert not (managed_bin / "hermes-acp.cmd").exists()
    assert (managed_bin / "uv.exe").exists()
    assert isolated_windows_uninstall == [False]


def test_managed_full_uninstall_sweeps_bin_path_and_home(
    managed_bin: Path,
    isolated_windows_uninstall: list[bool],
):
    project_root = managed_bin.parent / "hermes-agent"
    project_root.mkdir()

    uninstall._perform_uninstall(
        project_root=project_root,
        hermes_home=managed_bin.parent,
        full_uninstall=True,
        remove_profiles=False,
        named_profiles=[],
    )

    assert not managed_bin.parent.exists()
    assert isolated_windows_uninstall == [True]


def test_launcher_names_stay_in_lockstep_with_install_ps1():
    """The sweep must cover exactly the names install.ps1 stages, and no
    generic name it could clobber. Reads the real installer list so the two
    sides cannot drift apart silently."""
    import re

    install_ps1 = (
        Path(uninstall.__file__).resolve().parents[1] / "scripts" / "install.ps1"
    ).read_text(encoding="ascii")
    match = re.search(r"foreach \(\$launcher in @\(([^)]*)\)\)", install_ps1)
    assert match, "launcher staging loop not found in install.ps1"
    staged = set(re.findall(r'"([^"]+)"', match.group(1)))

    assert staged == set(_WINDOWS_BIN_LAUNCHERS)
    for name in _WINDOWS_BIN_LAUNCHERS:
        assert name.startswith("hermes")  # never a generic name it could clobber


class TestManagedBinPathMarker:
    """The managed ``bin`` PATH entry goes only when the dir itself goes.

    Markers match against Windows registry PATH entries, so the inputs here
    are Windows-shaped path strings regardless of the host — feeding
    ``tmp_path`` would make the test pass only on Windows hosts.
    """

    HOME = r"C:\Users\me\AppData\Local\hermes"
    BIN_ENTRY = r"C:\Users\me\AppData\Local\hermes\bin"

    def test_keep_data_markers_spare_the_managed_bin(self):
        markers = [m.lower() for m in uninstall._hermes_path_markers(Path(self.HOME))]

        assert not any(self.BIN_ENTRY.lower().startswith(m) for m in markers)

    def test_full_wipe_markers_take_the_managed_bin(self):
        markers = [
            m.lower()
            for m in uninstall._hermes_path_markers(
                Path(self.HOME), include_managed_bin=True
            )
        ]

        assert any(self.BIN_ENTRY.lower().startswith(m) for m in markers)
