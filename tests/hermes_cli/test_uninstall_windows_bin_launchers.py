"""Uninstall must not leave a dangling ``hermes`` command on Windows.

Every uninstall mode deletes the code checkout, but the launcher copies
staged onto PATH in the managed binary dir (the default Hermes root's
``bin``) live outside it. A surviving launcher makes ``hermes`` in a new
terminal resolve and then error on its missing venv target — worse than
command-not-found. The sweep takes only the launcher names
(``_launchers.WINDOWS_BIN_LAUNCHERS``, any extension): a pre-PM leftover
``uv.exe`` and the user's own scripts are the fail-closed cleanup's or the
user's to remove, and the dir itself goes only when nothing else remains —
a kept keep-data PATH entry must not dangle on an emptied dir.

Platform verdicts are injected parameters (input→output, not host fakes).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import uninstall


@pytest.fixture
def managed_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Default-root ``bin`` holding staged launcher copies."""
    home = tmp_path / "hermes"
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "hermes.exe").write_bytes(b"MZ launcher")
    # write_bytes: write_text's newline translation would turn the CRLF into
    # CR CR LF on Windows, as in the fixture below.
    (bin_dir / "hermes-acp.cmd").write_bytes(b"@echo off\r\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return bin_dir


def test_removes_only_the_launcher_names(managed_bin: Path):
    removed = uninstall.remove_windows_bin_launchers(windows=True)

    assert sorted(p.name for p in removed) == ["hermes-acp.cmd", "hermes.exe"]
    assert not managed_bin.exists(), "a launcher-only dir goes with its launchers"


def test_leaves_the_pre_pm_uv_and_user_files_behind(managed_bin: Path):
    """A pre-PM leftover ``uv.exe`` and the user's own script survive the sweep:
    the legacy-uv cleanup fails closed on Windows and names the manual step, so
    the launcher sweep taking the whole dir would silently delete what that
    policy keeps — and a keep-data PATH entry would dangle on the emptied dir."""
    (managed_bin / "uv.exe").write_bytes(b"MZ" + b"\0" * 64)
    # write_bytes: write_text's newline translation would turn the CRLF into
    # CR CR LF on Windows, so the byte assertion below would fail on a file
    # nothing touched.
    (managed_bin / "my-script.cmd").write_bytes(b"@echo off\r\n")

    removed = uninstall.remove_windows_bin_launchers(windows=True)

    assert sorted(p.name for p in removed) == ["hermes-acp.cmd", "hermes.exe"]
    assert (managed_bin / "uv.exe").is_file()
    assert (managed_bin / "my-script.cmd").read_bytes() == b"@echo off\r\n"


def test_leaves_non_launcher_suffixes_alone(managed_bin: Path):
    """Only the launcher LEAVES go: ``hermes.bak``/``hermes.old`` are user data,
    not launchers. A stem-based match would take them (the old whole-dir sweep
    took everything); the narrow leaf list must not."""
    for name in ("hermes.bak", "hermes.old", "hermes-notes.txt", "hermes-agent.exe"):
        (managed_bin / name).write_text("user data", encoding="ascii")

    removed = uninstall.remove_windows_bin_launchers(windows=True)

    assert sorted(p.name for p in removed) == ["hermes-acp.cmd", "hermes.exe"]
    for name in ("hermes.bak", "hermes.old", "hermes-notes.txt", "hermes-agent.exe"):
        assert (managed_bin / name).read_text(encoding="ascii") == "user data"


def test_leaves_the_rename_aside_residue_for_manual_cleanup(managed_bin: Path, monkeypatch: pytest.MonkeyPatch):
    """A locked running launcher is renamed aside ``<leaf>.uninstalled.<pid>``; that
    residue carries no receipt proving Hermes created it, so a later run leaves it
    for manual cleanup rather than gambling that a numeric lookalike is ours."""
    import os

    locked = True
    real_unlink = Path.unlink

    def maybe_refuse(self, *args, **kwargs):
        if locked and self.name == "hermes.exe":
            raise PermissionError("mandatory-locked (the running trampoline)")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", maybe_refuse)
    assert sorted(p.name for p in uninstall.remove_windows_bin_launchers(windows=True)) == [
        "hermes-acp.cmd", "hermes.exe"
    ]
    residue = f"hermes.exe.uninstalled.{os.getpid()}"
    assert (managed_bin / residue).is_file()
    assert managed_bin.is_dir()

    # Lock gone: the residue is not a launcher leaf, so it stays.
    locked = False
    assert uninstall.remove_windows_bin_launchers(windows=True) == []
    assert (managed_bin / residue).is_file()


def test_numeric_residue_lookalikes_survive(managed_bin: Path):
    """No ``<leaf>.uninstalled.<...>`` file is reclaimed: a user's
    ``hermes.uninstalled.123`` (the numeric collision) and
    ``hermes.uninstalled.notes`` are both the user's, not Hermes' — a numeric
    shape is not a receipt."""
    user_files = (
        "hermes.uninstalled.123",      # the numeric collision: shape, not proof
        "hermes.uninstalled.notes",
        "hermes.exe.uninstalled.bak",
        "hermes-acp.uninstalled.old",
        "hermes.uninstalled.\u00b2",  # superscript two: isdigit() True, not ASCII
    )
    for name in user_files:
        (managed_bin / name).write_text("user data", encoding="ascii")

    removed = sorted(p.name for p in uninstall.remove_windows_bin_launchers(windows=True))

    assert removed == ["hermes-acp.cmd", "hermes.exe"]
    for name in user_files:
        assert (managed_bin / name).read_text(encoding="ascii") == "user data"


def test_stacked_rename_aside_residue_survives(managed_bin: Path):
    """A stacked residue (``<leaf>.uninstalled.<pid>.uninstalled.<pid>``) carries
    no receipt either, so it stays for manual cleanup."""
    double = "hermes.exe.uninstalled.1.uninstalled.123"
    (managed_bin / double).write_text("residue", encoding="ascii")

    removed = sorted(p.name for p in uninstall.remove_windows_bin_launchers(windows=True))

    assert double not in removed
    assert (managed_bin / double).read_text(encoding="ascii") == "residue"


def test_anchors_on_default_root_not_profile_home(
    managed_bin: Path, monkeypatch: pytest.MonkeyPatch
):
    """The launcher dir is per-machine; a profile HERMES_HOME must not
    redirect the sweep into ``profiles/<name>/bin``."""
    home = managed_bin.parent
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "work"))

    removed = uninstall.remove_windows_bin_launchers(windows=True)

    assert sorted(p.name for p in removed) == ["hermes-acp.cmd", "hermes.exe"]
    assert not (managed_bin / "hermes.exe").exists()


def test_noop_on_posix(managed_bin: Path):
    assert uninstall.remove_windows_bin_launchers(windows=False) == []
    assert (managed_bin / "hermes.exe").exists()


def test_noop_when_no_bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert uninstall.remove_windows_bin_launchers(windows=True) == []
