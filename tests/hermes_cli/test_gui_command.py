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
    # The install now runs with a resolved env (managed-Node PATH), never a bare
    # ``env=None`` that would leave npm's child scripts unable to find ``node``.
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

    # A managed Node tree on disk so with_hermes_node_path() actually prepends it.
    home = tmp_path / "hermes-home"
    (home / "node" / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Simulate the stripped PATH the desktop updater chain hands us.
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
    assert "/usr/bin" in path_parts  # the bare updater PATH is preserved, just after managed Node


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


def test_gui_does_not_retry_after_packaged_executable_exists(tmp_path, monkeypatch, capsys):
    """A build that already produced a packaged executable did NOT fail from the
    Electron-download problem the cache purge + mirror retries exist to repair.

    Regression for #40187: a late failure such as macOS code signing leaves
    Hermes.app/Contents/MacOS/Hermes in place. Re-downloading Electron can't
    repair a signing failure, so the destructive purge + slow mirror retry must
    be skipped — we fail directly instead of grinding through an identical retry.
    """
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # Executable EXISTS at failure time → late failure, not a corrupt download.
    _make_packaged_executable(root, monkeypatch, platform="darwin")
    monkeypatch.delenv("ELECTRON_MIRROR", raising=False)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[Path("/c/electron.zip")]) as mock_purge, \
         patch("hermes_cli.main._redownload_electron_dist", return_value=True) as mock_dl, \
         patch("hermes_cli.main.subprocess.run", return_value=pack_fail) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    # Neither destructive recovery runs, and there is exactly ONE pack attempt.
    mock_purge.assert_not_called()
    mock_dl.assert_not_called()
    assert mock_run.call_count == 1
    assert "Desktop GUI build failed" in capsys.readouterr().out


def test_gui_does_not_override_user_electron_mirror(tmp_path, monkeypatch, capsys):
    """A user-pinned ELECTRON_MIRROR is respected: no extra mirror fallback
    attempt (and we never swap in our default mirror)."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # No packaged executable: the build failure is the download-class the
    # mirror fallback handles (and we assert the user's pin is respected).
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
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


# ── electronDist (re)download helper tests (#47266) ───────────────────


@pytest.mark.parametrize(
    "platform,rel",
    [
        ("linux", "dist/electron"),
        ("win32", "dist/electron.exe"),
        ("darwin", "dist/Electron.app/Contents/MacOS/Electron"),
    ],
)
def test_electron_dist_ok_per_platform(tmp_path, monkeypatch, platform, rel):
    monkeypatch.setattr(cli_main.sys, "platform", platform)
    electron = tmp_path / "node_modules" / "electron"
    # A dist dir that exists but lacks the binary is NOT ok (partial extraction).
    (electron / "dist").mkdir(parents=True)
    assert cli_main._electron_dist_ok(tmp_path) is False

    binp = electron / rel
    binp.parent.mkdir(parents=True, exist_ok=True)
    binp.write_text("", encoding="utf-8")
    assert cli_main._electron_dist_ok(tmp_path) is True


def test_electron_dir_prefers_workspace_local_package(tmp_path):
    """npm may nest electron under apps/desktop; resolve there over the root hoist."""
    root_electron = tmp_path / "node_modules" / "electron"
    local_electron = tmp_path / "apps" / "desktop" / "node_modules" / "electron"
    root_electron.mkdir(parents=True)
    local_electron.mkdir(parents=True)

    assert cli_main._electron_dir(tmp_path) == local_electron


def test_electron_dist_ok_finds_workspace_local_binary(tmp_path, monkeypatch):
    """A nested apps/desktop electron with a valid binary counts as ok."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    binp = tmp_path / "apps" / "desktop" / "node_modules" / "electron" / "dist" / "electron"
    binp.parent.mkdir(parents=True)
    binp.write_text("", encoding="utf-8")
    assert cli_main._electron_dist_ok(tmp_path) is True


def test_redownload_electron_dist_noop_when_present(tmp_path, monkeypatch):
    """Already-healthy dist → no download, so an unrelated build failure can't
    trigger a needless ~200 MB refetch."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    binp = tmp_path / "node_modules" / "electron" / "dist" / "electron"
    binp.parent.mkdir(parents=True)
    binp.write_text("", encoding="utf-8")

    with patch("hermes_cli.main.subprocess.run") as mock_run:
        assert cli_main._redownload_electron_dist(tmp_path, {}) is True
    mock_run.assert_not_called()


def test_redownload_electron_dist_returns_false_when_download_fails(tmp_path, monkeypatch):
    """install.js ran but produced no binary (still blocked) → False, so the
    caller skips a doomed pack."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    electron = tmp_path / "node_modules" / "electron"
    electron.mkdir(parents=True)
    (electron / "install.js").write_text("// stub", encoding="utf-8")

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/node"), \
         patch("hermes_cli.main.subprocess.run",
               return_value=subprocess.CompletedProcess(["node"], 1)):
        assert cli_main._redownload_electron_dist(tmp_path, {}) is False


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


def test_stop_desktop_build_lock_no_release_dir(tmp_path, monkeypatch):
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    with patch("psutil.process_iter") as it:
        assert cli_main._stop_desktop_processes_locking_build(desktop_dir) == []
    it.assert_not_called()


def test_force_adhoc_signing_disables_discovery_on_local_packaged_rebuild(monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    env = {}
    assert cli_main._force_adhoc_macos_signing(env, source_mode=False) is True
    assert env["CSC_IDENTITY_AUTO_DISCOVERY"] == "false"


@pytest.mark.parametrize("key", ["CSC_LINK", "APPLE_SIGNING_IDENTITY"])
def test_force_adhoc_signing_preserves_real_identity(monkeypatch, key):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    env = {key: "secret"}
    assert cli_main._force_adhoc_macos_signing(env, source_mode=False) is False
    assert "CSC_IDENTITY_AUTO_DISCOVERY" not in env


def test_force_adhoc_signing_respects_explicit_caller_flag(monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    env = {"CSC_IDENTITY_AUTO_DISCOVERY": "true"}
    assert cli_main._force_adhoc_macos_signing(env, source_mode=False) is False
    assert env["CSC_IDENTITY_AUTO_DISCOVERY"] == "true"


# --- macOS TCC-stable local signing (relaunch fixup) -----------------------


def _write_info_plist(bundle: Path, identifier: str) -> None:
    import plistlib

    info = bundle / "Contents" / "Info.plist"
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_bytes(plistlib.dumps({"CFBundleIdentifier": identifier}))


def _make_signable_app(desktop_dir: Path) -> Path:
    """Build a fake packaged Hermes.app with the pieces the signer must find."""
    ent_dir = desktop_dir / "electron"
    ent_dir.mkdir(parents=True, exist_ok=True)
    (ent_dir / "entitlements.mac.plist").write_text("<plist/>", encoding="utf-8")
    (ent_dir / "entitlements.mac.inherit.plist").write_text("<plist/>", encoding="utf-8")

    app = desktop_dir / "release" / "mac-arm64" / "Hermes.app"
    _write_info_plist(app, "com.nousresearch.hermes")
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "Hermes").write_text("", encoding="utf-8")

    helper = app / "Contents" / "Frameworks" / "Hermes Helper.app"
    _write_info_plist(helper, "com.nousresearch.hermes.helper")

    native_dir = app / "Contents" / "Resources" / "app.asar.unpacked" / "node_modules" / "pty"
    native_dir.mkdir(parents=True)
    (native_dir / "pty.node").write_text("", encoding="utf-8")
    (app / "Contents" / "Frameworks" / "chrome_crashpad_handler").write_text("", encoding="utf-8")
    return app


def _collect_codesign_calls(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        cli_main.shutil, "which", lambda name: "/usr/bin/codesign" if name == "codesign" else None
    )
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    return calls


def test_desktop_macos_local_codesign_signs_native_binaries(tmp_path, monkeypatch):
    """The standalone Mach-O pass must actually find files inside the bundle.

    Regression: an absolute-path parts check always matches the outer
    Hermes.app component, silently skipping every .node/.dylib/crashpad
    binary — codesign then rejects the outer signature (nested code unsigned).
    """
    desktop_dir = tmp_path / "apps" / "desktop"
    app = _make_signable_app(desktop_dir)
    calls = _collect_codesign_calls(monkeypatch)

    assert cli_main._desktop_macos_local_codesign(app, desktop_dir=desktop_dir) is True

    signed = [c[-1] for c in calls if c[:3] == ["/usr/bin/codesign", "--force", "--sign"]]
    assert str(app / "Contents" / "Resources" / "app.asar.unpacked" / "node_modules" / "pty" / "pty.node") in signed
    assert str(app / "Contents" / "Frameworks" / "chrome_crashpad_handler") in signed


def test_relaunchable_fixup_noop_when_publisher_signing_configured(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    monkeypatch.setenv("CSC_LINK", "publisher-cert")

    with patch("hermes_cli.main.subprocess.run") as run:
        assert cli_main._desktop_macos_relaunchable_fixup(root / "apps" / "desktop") is True
    run.assert_not_called()


def test_relaunchable_fixup_falls_back_to_legacy_adhoc_on_failure(tmp_path, monkeypatch, capsys):
    """A failing stable sign must still leave a launchable (deep ad-hoc) bundle."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.delenv("CSC_LINK", raising=False)
    monkeypatch.delenv("APPLE_SIGNING_IDENTITY", raising=False)
    exe = _make_packaged_executable(root, monkeypatch, platform="darwin")
    app = exe.parents[2]

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        cli_main.shutil, "which", lambda name: "/usr/bin/codesign" if name == "codesign" else None
    )
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_main, "_desktop_macos_has_valid_real_signature", lambda a: False)
    monkeypatch.setattr(cli_main, "_desktop_macos_local_signing_identity", lambda: None)

    def boom(*a, **kw):
        raise subprocess.CalledProcessError(1, ["codesign"])

    monkeypatch.setattr(cli_main, "_desktop_macos_local_codesign", boom)

    assert cli_main._desktop_macos_relaunchable_fixup(desktop_dir) is False
    assert ["xattr", "-cr", str(app)] in calls
    assert ["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app)] in calls


# --- desktop.* launch options (config.yaml) -------------------------------


def test_desktop_launch_options_survives_config_error():
    with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
        flags, gpu = cli_main._desktop_launch_options()
    assert flags == []
    assert gpu == "auto"
