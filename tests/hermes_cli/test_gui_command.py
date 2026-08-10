"""Tests for ``hermes gui`` desktop launcher wiring."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main


def _ns(**kw):
    defaults = dict(
        skip_build=False,
        build_only=False,
        force_build=False,
        source=False,
        fake_boot=False,
        ignore_existing=False,
        hermes_root=None,
        cwd=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _make_desktop_tree(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    return root


def _make_packaged_executable(root: Path, monkeypatch, platform: str = "darwin") -> Path:
    monkeypatch.setattr(cli_main.sys, "platform", platform)
    desktop_dir = root / "apps" / "desktop"
    if platform == "darwin":
        exe = desktop_dir / "release" / "mac-arm64" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    elif platform == "win32":
        exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    else:
        exe = desktop_dir / "release" / "linux-unpacked" / "hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    return exe


def test_gui_installs_packages_and_launches_desktop_app(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)
    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok) as mock_install, \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_install.assert_called_once()
    assert mock_install.call_args.args == ("/usr/bin/npm", root)
    assert mock_install.call_args.kwargs["capture_output"] is False
    install_env = mock_install.call_args.kwargs["env"]
    assert install_env is not None and "PATH" in install_env
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "pack"]
    assert mock_run.call_args_list[0].kwargs["cwd"] == desktop_dir
    assert mock_run.call_args_list[1].args[0] == [str(packaged_exe)]
    assert mock_run.call_args_list[1].kwargs["cwd"] == desktop_dir


def test_gui_install_env_prepends_managed_node_on_bare_path(tmp_path, monkeypatch):
    """Regression: npm's child scripts (electron-winstaller's select-7z-arch.js)
    shell out to bare ``node``. When Desktop is launched from the updater chain
    the parent PATH is stripped, so the install env MUST carry the Hermes-managed
    Node ahead of that bare PATH or the install dies with ``node: not found``.
    """
    import os

    from hermes_constants import iter_hermes_node_dirs

    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch, platform="win32")

    home = tmp_path / "hermes-home"
    (home / "node" / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    launch_ok = subprocess.CompletedProcess(["hermes"], 0)

    with patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok) as mock_install, \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[subprocess.CompletedProcess([], 0), launch_ok]), \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns(skip_build=False))

    managed_dirs = [str(p) for p in iter_hermes_node_dirs() if p.is_dir()]
    assert managed_dirs, "managed node tree not discovered"
    install_env = mock_install.call_args.kwargs["env"]
    path_parts = install_env["PATH"].split(os.pathsep)
    assert path_parts[: len(managed_dirs)] == managed_dirs
    assert "/usr/bin" in path_parts


def test_gui_uses_flatpak_client_with_native_backend_on_linux(monkeypatch, tmp_path):
    """Linux Flatpak path: ensure the Flatpak helper is invoked with the native
    backend command and working directory, instead of building Electron locally.
    """
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    launched = []
    monkeypatch.setattr(
        "hermes_cli.flatpak_desktop.launch_with_native_backend",
        lambda command, *, cwd: launched.append((command, cwd)) or 0,
    )

    with pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(cwd=str(tmp_path)))

    assert exc.value.code == 0
    assert launched == [([sys.executable, "-m", "hermes_cli.main"], str(tmp_path))]


# ── Content-hash stamp tests ──────────────────────────────────────────


# ── Electron build-cache recovery tests ───────────────────────────────


def _write_zip(path: Path) -> None:
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("electron", "fake binary payload")


def test_purge_electron_build_cache_clears_all_zips_and_unpacked_dir(tmp_path, monkeypatch):
    """Purge is unconditional: it removes every electron-*.zip (regardless of
    whether stdlib zipfile thinks it's corrupt) plus the half-written unpacked
    dir, because @electron/get's own SHASUM check on re-download is the real
    validator — not a self-rolled one."""
    cache = tmp_path / "electron-cache"
    clean = cache / "electron-v40.9.3-linux-x64.zip"
    prepended = cache / "hashdir" / "electron-v40.9.3-linux-x64.zip"
    _write_zip(clean)
    _write_zip(prepended)
    prepended.write_bytes(b"\x00" * 4096 + prepended.read_bytes())

    desktop_dir = tmp_path / "apps" / "desktop"
    unpacked = desktop_dir / "release" / "linux-unpacked"
    unpacked.mkdir(parents=True)
    (unpacked / "LICENSE.electron.txt").write_text("x", encoding="utf-8")
    (unpacked / "resources.pak").write_text("x", encoding="utf-8")

    monkeypatch.setattr(cli_main, "_electron_download_cache_dirs", lambda: [cache])

    removed = cli_main._purge_electron_build_cache(desktop_dir)

    assert clean in removed
    assert prepended in removed
    assert unpacked in removed
    assert not clean.exists()