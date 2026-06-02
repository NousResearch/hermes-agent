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
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_install.assert_called_once_with("/usr/bin/npm", root, capture_output=False)
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "pack"]
    assert mock_run.call_args_list[0].kwargs["cwd"] == desktop_dir
    assert mock_run.call_args_list[1].args[0] == [str(packaged_exe)]
    assert mock_run.call_args_list[1].kwargs["cwd"] == desktop_dir


def test_gui_forwards_desktop_environment_overrides(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    hermes_root = tmp_path / "custom-hermes"
    cwd = tmp_path / "project"
    hermes_root.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns(
            fake_boot=True,
            ignore_existing=True,
            hermes_root=str(hermes_root),
            cwd=str(cwd),
        ))

    launch_env = mock_run.call_args_list[1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_BOOT_FAKE"] == "1"
    assert launch_env["HERMES_DESKTOP_IGNORE_EXISTING"] == "1"
    assert launch_env["HERMES_DESKTOP_HERMES_ROOT"] == str(hermes_root)
    assert launch_env["HERMES_DESKTOP_CWD"] == str(cwd)


def test_gui_exits_when_npm_missing(tmp_path, monkeypatch, capsys):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    with patch("hermes_cli.main.shutil.which", return_value=None), \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    assert "npm was not found" in capsys.readouterr().out


def test_gui_skip_build_requires_existing_packaged_app(tmp_path, monkeypatch, capsys):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")

    with pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 1
    assert "no packaged desktop app" in capsys.readouterr().out


def test_gui_skip_build_launches_existing_packaged_app_without_npm(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main.shutil.which", return_value=None), \
         patch("hermes_cli.main._run_npm_install_deterministic") as mock_install, \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 0
    mock_install.assert_not_called()
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == [str(packaged_exe)]


def test_gui_source_mode_uses_renderer_build_and_electron(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    build_ok = subprocess.CompletedProcess(["npm", "run", "build"], 0)
    launch_ok = subprocess.CompletedProcess(["npm", "exec", "--", "electron", "."], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main.subprocess.run", side_effect=[build_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(source=True))

    assert exc.value.code == 0
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "build"]
    assert mock_run.call_args_list[0].kwargs["cwd"] == desktop_dir
    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/npm", "exec", "--", "electron", "."]
    assert mock_run.call_args_list[1].kwargs["cwd"] == desktop_dir


@pytest.mark.parametrize(
    "argv",
    [
        ["hermes", "gui"],
        ["hermes", "-m", "gpt5", "gui"],
    ],
)
def test_gui_is_known_builtin_for_plugin_gating(argv):
    with patch.object(sys, "argv", argv):
        assert cli_main._plugin_cli_discovery_needed() is False


def _write_zip(path: Path) -> None:
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("electron", "fake binary payload")


def test_purge_corrupt_electron_cache_removes_only_bad_zips(tmp_path, monkeypatch):
    cache = tmp_path / "electron-cache"
    good = cache / "electron-v40.9.3-linux-x64.zip"
    bad = cache / "hashdir" / "electron-v40.9.3-linux-x64.zip"
    _write_zip(good)
    _write_zip(bad)
    # Corrupt the second zip by truncating its central directory.
    bad.write_bytes(bad.read_bytes()[:20])

    monkeypatch.setattr(cli_main, "_electron_download_cache_dirs", lambda: [cache])

    removed = cli_main._purge_corrupt_electron_cache()

    assert removed == [bad]
    assert good.exists()
    assert not bad.exists()


def test_purge_corrupt_electron_cache_noop_when_all_valid(tmp_path, monkeypatch):
    cache = tmp_path / "electron-cache"
    good = cache / "electron-v40.9.3-linux-x64.zip"
    _write_zip(good)
    monkeypatch.setattr(cli_main, "_electron_download_cache_dirs", lambda: [cache])

    assert cli_main._purge_corrupt_electron_cache() == []
    assert good.exists()


def test_gui_retries_pack_after_purging_corrupt_electron_cache(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)
    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._purge_corrupt_electron_cache", return_value=[Path("/c/electron.zip")]) as mock_purge, \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_fail, pack_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_purge.assert_called_once()
    # First pack fails, purge runs, second pack succeeds, then launch.
    assert mock_run.call_count == 3
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "pack"]
    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/npm", "run", "pack"]
    assert mock_run.call_args_list[2].args[0] == [str(packaged_exe)]


def test_gui_does_not_retry_when_cache_is_clean(tmp_path, monkeypatch, capsys):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch, platform="linux")

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._purge_corrupt_electron_cache", return_value=[]) as mock_purge, \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_fail]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    # Purge was attempted but found nothing, so no retry pack runs.
    mock_purge.assert_called_once()
    assert mock_run.call_count == 1
    assert "Desktop GUI build failed" in capsys.readouterr().out
