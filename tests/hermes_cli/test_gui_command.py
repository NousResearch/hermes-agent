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
    mock_install.assert_called_once_with("/usr/bin/npm", root, capture_output=False, env=None)
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
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
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


def test_gui_linux_configures_sandbox_before_launch(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    sandbox = packaged_exe.parent / "chrome-sandbox"
    sandbox.write_text("", encoding="utf-8")
    sandbox.chmod(0o755)
    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/sudo"), \
         patch("hermes_cli.main.subprocess.run", return_value=ok) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 0
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/sudo", "chown", "root:root", str(sandbox)]
    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/sudo", "chmod", "4755", str(sandbox)]
    assert mock_run.call_args_list[2].args[0] == [str(packaged_exe)]


def test_gui_linux_rejects_symlink_sandbox(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    # Point chrome-sandbox at an unrelated file via symlink
    target = tmp_path / "dangerous"
    target.write_text("pwned", encoding="utf-8")
    sandbox = packaged_exe.parent / "chrome-sandbox"
    sandbox.symlink_to(target)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/sudo"), \
         patch("hermes_cli.main.subprocess.run") as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 1
    # Must NOT have called sudo chown/chmod on the symlink target
    for call in mock_run.call_args_list:
        assert "chown" not in call.args[0]
        assert "chmod" not in call.args[0]


def test_gui_linux_skips_fixup_when_already_configured(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    sandbox = packaged_exe.parent / "chrome-sandbox"
    sandbox.write_text("", encoding="utf-8")
    # Simulate root-owned 4755 — lstat().st_uid==0 and mode==0o4755
    # We can't actually chown to root in tests, so mock lstat to return
    # the expected values directly.
    import stat as stat_mod
    fake_stat = type("s", (), {"st_uid": 0, "st_mode": 0o4755 | stat_mod.S_IFREG})()
    sandbox_lstat_orig = type(sandbox).lstat
    monkeypatch.setattr(type(sandbox), "lstat", lambda self: fake_stat)

    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/sudo"), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 0
    # Only the launch call — no sudo chown/chmod
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
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
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


# ── Content-hash stamp tests ──────────────────────────────────────────


def test_desktop_build_stamp_skips_build_when_up_to_date(tmp_path, monkeypatch):
    """When the stamp matches and the artifact exists, build is skipped entirely."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    launch_ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main._run_npm_install_deterministic") as mock_install, \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok) as mock_run, \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_install.assert_not_called()
    mock_run.assert_called_once()  # only the launch call, no build


def test_desktop_force_build_overrides_stamp(tmp_path, monkeypatch):
    """--force-build forces a rebuild even when the stamp says up-to-date."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)
    launch_ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok) as mock_install, \
         patch("hermes_cli.main._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main._write_desktop_build_stamp") as mock_stamp, \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(force_build=True))

    assert exc.value.code == 0
    mock_install.assert_called_once()
    mock_stamp.assert_called_once()
    # pack + launch = 2 calls
    assert mock_run.call_count == 2


def test_compute_desktop_content_hash_stable(tmp_path, monkeypatch):
    """_compute_desktop_content_hash returns the same digest for identical trees."""
    root = _make_desktop_tree(tmp_path)
    (root / "apps" / "desktop" / "main.js").write_text("console.log('hi')", encoding="utf-8")
    (root / "package.json").write_text('{"name":"hermes"}', encoding="utf-8")
    (root / "package-lock.json").write_text('{}', encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    h1 = cli_main._compute_desktop_content_hash(root)
    h2 = cli_main._compute_desktop_content_hash(root)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_compute_desktop_content_hash_changes_on_edit(tmp_path, monkeypatch):
    """Editing a file under apps/desktop/ changes the hash."""
    root = _make_desktop_tree(tmp_path)
    (root / "apps" / "desktop" / "main.js").write_text("v1", encoding="utf-8")
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    h1 = cli_main._compute_desktop_content_hash(root)
    (root / "apps" / "desktop" / "main.js").write_text("v2", encoding="utf-8")
    h2 = cli_main._compute_desktop_content_hash(root)
    assert h1 != h2


def test_desktop_build_needed_detects_missing_artifact(tmp_path, monkeypatch):
    """Even with a valid stamp, missing artifact means build is needed."""
    root = _make_desktop_tree(tmp_path)
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # Write a stamp that matches current content
    cli_main._write_desktop_build_stamp(root, source_mode=False)
    # No packaged executable exists → build needed
    assert cli_main._desktop_build_needed(
        root / "apps" / "desktop", root, source_mode=False
    ) is True


def test_desktop_build_stamp_round_trip(tmp_path, monkeypatch):
    """Write stamp, then _desktop_build_needed returns False when artifact exists."""
    root = _make_desktop_tree(tmp_path)
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # Create the artifact so the "artifact exists" check passes
    _make_packaged_executable(root, monkeypatch)
    # Write stamp
    cli_main._write_desktop_build_stamp(root, source_mode=False)
    # Build should NOT be needed
    assert cli_main._desktop_build_needed(
        root / "apps" / "desktop", root, source_mode=False
    ) is False


def test_compute_desktop_content_hash_works_without_gitignore(tmp_path, monkeypatch):
    """When no .gitignore exists, _compute_desktop_content_hash still works (matches everything)."""
    root = _make_desktop_tree(tmp_path)
    (root / "apps" / "desktop" / "main.js").write_text("v1", encoding="utf-8")
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    # No .gitignore → pathspec matches nothing → all files hashed
    h = cli_main._compute_desktop_content_hash(root)
    assert len(h) == 64  # valid sha256 hex

    # Edit a file → hash changes
    (root / "apps" / "desktop" / "main.js").write_text("v2", encoding="utf-8")
    h2 = cli_main._compute_desktop_content_hash(root)
    assert h != h2


def test_compute_desktop_content_hash_respects_gitignore(tmp_path, monkeypatch):
    """Files matched by .gitignore are excluded from the hash."""
    root = _make_desktop_tree(tmp_path)
    (root / "apps" / "desktop" / "main.js").write_text("hello", encoding="utf-8")
    (root / "apps" / "desktop" / "secrets.env").write_text("API_KEY=xxx", encoding="utf-8")
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    (root / ".gitignore").write_text("*.env\n", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    # Reset cached spec
    cli_main._DESKTOP_STAMP_SPEC = None

    h1 = cli_main._compute_desktop_content_hash(root)

    # Change the .env file (ignored) — hash should NOT change
    (root / "apps" / "desktop" / "secrets.env").write_text("API_KEY=yyy", encoding="utf-8")
    cli_main._DESKTOP_STAMP_SPEC = None  # reset since gitignore hasn't changed
    h2 = cli_main._compute_desktop_content_hash(root)
    assert h1 == h2, "changing an ignored file should not change the hash"

    # Change the .js file (not ignored) — hash SHOULD change
    (root / "apps" / "desktop" / "main.js").write_text("world", encoding="utf-8")
    cli_main._DESKTOP_STAMP_SPEC = None
    h3 = cli_main._compute_desktop_content_hash(root)
    assert h1 != h3, "changing a tracked file should change the hash"


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
    # A "clean" zip and a prepended-junk zip — the latter is the real-world
    # corruption that zipfile.testzip() silently passes (it reads from the
    # end-of-central-directory backward), which is why we don't gate on it.
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
    assert not prepended.exists()
    assert not unpacked.exists()


def test_purge_electron_build_cache_empty_when_nothing_present(tmp_path, monkeypatch):
    """No cached zips and no unpacked dir → nothing removed, so the caller
    knows a retry is pointless."""
    cache = tmp_path / "electron-cache"
    cache.mkdir()
    desktop_dir = tmp_path / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "_electron_download_cache_dirs", lambda: [cache])

    assert cli_main._purge_electron_build_cache(desktop_dir) == []


def test_gui_retries_pack_once_after_purging_build_cache(tmp_path, monkeypatch):
    """First pack fails, purge clears the cache, second pack succeeds, launch."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)
    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[Path("/c/electron.zip")]) as mock_purge, \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_fail, pack_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_purge.assert_called_once()
    # pack(fail) → purge → pack(ok) → launch = 3 subprocess.run calls
    assert mock_run.call_count == 3
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "pack"]
    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/npm", "run", "pack"]
    assert mock_run.call_args_list[2].args[0] == [str(packaged_exe)]


def test_gui_falls_back_to_mirror_when_purge_finds_nothing(tmp_path, monkeypatch, capsys):
    """Purge clears nothing (not a cache problem) → fall back to an Electron
    mirror once before failing, so a GitHub-blocked download self-heals."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch, platform="linux")
    monkeypatch.delenv("ELECTRON_MIRROR", raising=False)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[]) as mock_purge, \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_fail, pack_fail]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    mock_purge.assert_called_once()
    # pack(fail) → purge(nothing) → pack via mirror(fail) = 2 subprocess.run calls
    assert mock_run.call_count == 2
    # The retry runs the same build but with ELECTRON_MIRROR injected.
    assert "ELECTRON_MIRROR" not in (mock_run.call_args_list[0].kwargs.get("env") or {})
    assert mock_run.call_args_list[1].kwargs["env"]["ELECTRON_MIRROR"]
    assert "Desktop GUI build failed" in capsys.readouterr().out


def test_gui_does_not_override_user_electron_mirror(tmp_path, monkeypatch, capsys):
    """A user-pinned ELECTRON_MIRROR is respected: no extra mirror fallback
    attempt (and we never swap in our default mirror)."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch, platform="linux")
    monkeypatch.setenv("ELECTRON_MIRROR", "https://mirror.example/electron/")

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[]) as mock_purge, \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_fail]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    mock_purge.assert_called_once()
    assert mock_run.call_count == 1
    assert mock_run.call_args_list[0].kwargs["env"]["ELECTRON_MIRROR"] == "https://mirror.example/electron/"
    assert "Desktop GUI build failed" in capsys.readouterr().out


class _FakeProc:
    """Minimal psutil.Process stand-in for the lock-breaker tests."""

    def __init__(self, pid: int, exe: str | None):
        self.pid = pid
        self.info = {"pid": pid, "exe": exe}
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_stop_desktop_build_lock_noop_off_windows(tmp_path, monkeypatch):
    """POSIX can unlink a running binary, so the helper is a no-op there."""
    desktop_dir = tmp_path / "apps" / "desktop"
    exe = desktop_dir / "release" / "linux-unpacked" / "hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli_main.sys, "platform", "linux")

    proc = _FakeProc(4321, str(exe))
    with patch("psutil.process_iter", return_value=[proc]) as it:
        assert cli_main._stop_desktop_processes_locking_build(desktop_dir) == []
    it.assert_not_called()
    assert proc.terminated is False


def test_stop_desktop_build_lock_terminates_only_release_procs(tmp_path, monkeypatch):
    desktop_dir = tmp_path / "apps" / "desktop"
    release = desktop_dir / "release" / "win-unpacked"
    release.mkdir(parents=True)
    locker_exe = release / "Hermes.exe"
    locker_exe.write_text("", encoding="utf-8")
    other_exe = tmp_path / "elsewhere" / "Hermes.exe"
    other_exe.parent.mkdir(parents=True)
    other_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    monkeypatch.setattr(cli_main.os, "getpid", lambda: 999)

    locker = _FakeProc(101, str(locker_exe))
    unrelated = _FakeProc(102, str(other_exe))
    selfish = _FakeProc(999, str(locker_exe))  # our own PID — never killed
    no_exe = _FakeProc(103, None)

    captured = {}

    def _wait(procs, timeout=None):
        captured["waited"] = list(procs)
        return procs, []

    with patch("psutil.process_iter", return_value=[locker, unrelated, selfish, no_exe]), \
         patch("psutil.wait_procs", side_effect=_wait):
        stopped = cli_main._stop_desktop_processes_locking_build(desktop_dir)

    assert stopped == [101]
    assert locker.terminated is True
    assert unrelated.terminated is False
    assert selfish.terminated is False
    assert captured["waited"] == [locker]


def test_stop_desktop_build_lock_no_release_dir(tmp_path, monkeypatch):
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    with patch("psutil.process_iter") as it:
        assert cli_main._stop_desktop_processes_locking_build(desktop_dir) == []
    it.assert_not_called()
