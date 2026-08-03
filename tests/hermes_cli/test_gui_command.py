"""Tests for ``hermes gui`` desktop launcher wiring."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main


@pytest.fixture(autouse=True)
def _isolate_xdg_data_home(tmp_path, monkeypatch):
    """Keep desktop-entry writes out of the developer's real home directory.

    ``cmd_gui`` registers an XDG launcher entry, and ``desktop_entry_path()``
    resolves it under ``XDG_DATA_HOME`` (falling back to ``~/.local/share``).
    While these tests faked the host as darwin the Linux-only registration
    never ran, so nothing escaped. Running them on their real host makes that
    call live, and on a Linux dev box it wrote a ``hermes.desktop`` pointing
    ``Exec=`` at the test's throwaway npm stub into the user's actual
    applications menu.

    The hermetic conftest deliberately does NOT redirect ``HOME`` (subprocesses
    depend on it being stable), so this has to be pinned per-file.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))


@pytest.fixture(autouse=True)
def _stable_keychain_detection(monkeypatch):
    """Pin Linux keychain detection to the fast GNOME env path.

    On Linux, ``cmd_gui`` falls back to a D-Bus ping via ``subprocess.run``
    when no keychain env var is present. Tests here mock ``subprocess.run``
    with strict ``side_effect`` lists, so an unpinned probe would silently
    consume an item meant for the build/launch calls. Detection-specific
    tests clear these vars again via ``_clear_keychain_env``.
    """
    monkeypatch.delenv("KDE_SESSION_VERSION", raising=False)
    monkeypatch.delenv("KDE_FULL_SESSION", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_PASSWORD_STORE", raising=False)
    monkeypatch.setenv("GNOME_KEYRING_CONTROL", "/run/user/1000/keyring")


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
        setup_tcc_identity=False,
        identity=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _make_desktop_tree(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    return root


def _make_packaged_executable(root: Path, monkeypatch) -> Path:
    """Create the packaged-app path layout electron-builder emits on THIS host.

    The layout is keyed off the real ``sys.platform`` rather than a caller-
    supplied override: ``cmd_gui`` resolves the executable through the same
    branch, so faking the platform here only proved the test and the code
    agreed about a host neither was running on.

    Note the Linux arm also lays down ``chrome-sandbox``. ``cmd_gui`` refuses to
    launch without it (Electron's setuid sandbox helper), which the old
    darwin-by-default fake concealed — on Linux the packaged tree genuinely has
    to include it.
    """
    desktop_dir = root / "apps" / "desktop"
    if sys.platform == "darwin":
        exe = desktop_dir / "release" / "mac-arm64" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    elif sys.platform == "win32":
        exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    else:
        exe = desktop_dir / "release" / "linux-unpacked" / "hermes"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    if sys.platform not in ("darwin", "win32"):
        (exe.parent / "chrome-sandbox").write_text("", encoding="utf-8")
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
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main._register_linux_desktop_entry"), \
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
    _make_packaged_executable(root, monkeypatch)

    # A managed Node tree on disk so with_hermes_node_path() actually prepends it.
    home = tmp_path / "hermes-home"
    (home / "node" / "bin").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Simulate the stripped PATH the desktop updater chain hands us.
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    launch_ok = subprocess.CompletedProcess(["hermes"], 0)

    # A plain return_value rather than a fixed side_effect list: this test only
    # cares about the env handed to the npm install, and pinning an exact
    # sequence of subprocess.run calls makes it fail (StopIteration) whenever
    # cmd_gui legitimately shells out one extra time — e.g. the Linux sandbox
    # fixup, which fires on hosts where chrome-sandbox isn't already
    # root-owned+4755. Assert on the install env, not on a call count.
    with patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok) as mock_install, \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok), \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns(skip_build=False))

    managed_dirs = [str(p) for p in iter_hermes_node_dirs() if p.is_dir()]
    assert managed_dirs, "managed node tree not discovered"
    install_env = mock_install.call_args.kwargs["env"]
    path_parts = install_env["PATH"].split(os.pathsep)
    assert path_parts[: len(managed_dirs)] == managed_dirs
    assert "/usr/bin" in path_parts  # the bare updater PATH is preserved, just after managed Node








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
    _make_packaged_executable(root, monkeypatch)
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




# ── electronDist (re)download helper tests (#47266) ───────────────────


def test_electron_dist_ok_on_this_host():
    """A dist dir that exists but lacks the binary is NOT ok (partial extraction).

    The binary's basename is per-OS (``electron`` / ``electron.exe`` /
    ``Electron.app/…/Electron``), and ``_electron_dist_binary()`` picks it from
    the real ``sys.platform``. Asking the implementation for the path it
    expects — instead of hardcoding one and faking the platform to match —
    makes this a genuine round-trip on whichever lane runs it.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        electron = root / "node_modules" / "electron"
        (electron / "dist").mkdir(parents=True)
        assert cli_main._electron_dist_ok(root) is False

        binp = cli_main._electron_dist_binary(root)
        # The resolved binary must live under the dist dir we just created.
        assert (electron / "dist") in binp.parents
        binp.parent.mkdir(parents=True, exist_ok=True)
        binp.write_text("", encoding="utf-8")
        assert cli_main._electron_dist_ok(root) is True


@pytest.mark.linux_only
def test_electron_dist_binary_basename_linux():
    """``dist/electron`` on Linux — asserted against the live function.

    Split per-OS rather than parametrized over a platform table: the old
    ``@parametrize(("linux", …), ("win32", …), ("darwin", …))`` skipped the two
    non-host rows, so outside the Linux lane those two branches were asserted
    nowhere at all. One marked test per OS puts each row on the lane that can
    actually execute it.
    """
    root = Path("/tmp/does-not-need-to-exist")
    assert cli_main._electron_dist_binary(root) == (
        root / "node_modules" / "electron" / "dist" / "electron"
    )


@pytest.mark.windows_only
def test_electron_dist_binary_basename_windows():
    """``dist/electron.exe`` on Windows — the ``.exe`` suffix is the whole point."""
    root = Path("C:/does-not-need-to-exist")
    assert cli_main._electron_dist_binary(root) == (
        root / "node_modules" / "electron" / "dist" / "electron.exe"
    )


@pytest.mark.macos_only
def test_electron_dist_binary_basename_macos():
    """``dist/Electron.app/Contents/MacOS/Electron`` on macOS.

    The nested ``.app`` bundle path is why #47266's "dist exists but the
    binary doesn't" check can't just stat the dist directory.
    """
    root = Path("/tmp/does-not-need-to-exist")
    assert cli_main._electron_dist_binary(root) == (
        root
        / "node_modules"
        / "electron"
        / "dist"
        / "Electron.app"
        / "Contents"
        / "MacOS"
        / "Electron"
    )










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




@pytest.mark.macos_only
def test_relaunchable_fixup_falls_back_to_legacy_adhoc_on_failure(tmp_path, monkeypatch, capsys):
    """A failing stable sign must still leave a launchable (deep ad-hoc) bundle.

    ``macos_only``: the subject is ``codesign`` against a real ``.app`` bundle
    layout (``exe.parents[2]``), which only the macOS packaged tree produces.
    """
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.delenv("CSC_LINK", raising=False)
    monkeypatch.delenv("APPLE_SIGNING_IDENTITY", raising=False)
    exe = _make_packaged_executable(root, monkeypatch)
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


# --- desktop --setup-tcc-identity ------------------------------------------


def _fake_proc(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def test_setup_tcc_identity_creates_cert_imports_and_configures(tmp_path, monkeypatch, capsys):
    """Fresh identity: openssl generates, security imports, config is written."""
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    monkeypatch.setattr(
        cli_main.shutil,
        "which",
        lambda name: {"openssl": "/usr/bin/openssl", "security": "/usr/bin/security", "codesign": "/usr/bin/codesign"}.get(name),
    )
    monkeypatch.setattr(cli_main.Path, "home", classmethod(lambda cls: tmp_path))

    identity = "Hermes Local Signing"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        # `security find-identity` on first run reports no matching identity.
        if cmd[:3] == ["/usr/bin/security", "find-identity", "-p"]:
            return _fake_proc(cmd, stdout="     0 identities found")
        return _fake_proc(cmd)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_main, "_desktop_packaged_executable", lambda d: None)
    monkeypatch.setattr(cli_main, "_desktop_macos_relaunchable_fixup", lambda d: True)
    # Avoid writing the real user config.
    monkeypatch.setattr("hermes_cli.config.set_config_value", lambda key, value: None)

    assert cli_main._desktop_macos_setup_tcc_identity(identity) is True

    out = capsys.readouterr().out
    assert "created and imported self-signed identity" in out
    assert "set desktop.macos_signing_identity" in out
    # openssl cert generation + pkcs12 export + security import all ran.
    assert any(c[0] == "/usr/bin/openssl" and "req" in c for c in calls)
    assert any(c[0] == "/usr/bin/openssl" and "pkcs12" in c for c in calls)
    assert any(c[0] == "/usr/bin/security" and c[1] == "import" for c in calls)
    # Temp files cleaned up.
    assert not list(tmp_path.glob("hermes-tcc-*"))


def test_setup_tcc_identity_skips_generation_when_already_present(tmp_path, monkeypatch, capsys):
    """Idempotent: an existing identity is reused, not regenerated."""
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    monkeypatch.setattr(
        cli_main.shutil,
        "which",
        lambda name: {"openssl": "/usr/bin/openssl", "security": "/usr/bin/security", "codesign": "/usr/bin/codesign"}.get(name),
    )
    monkeypatch.setattr(cli_main.Path, "home", classmethod(lambda cls: tmp_path))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == ["/usr/bin/security", "find-identity", "-p"]:
            return _fake_proc(cmd, stdout='  "Hermes Local Signing" (CSSMERR_TP_NOT_TRUSTED)')
        return _fake_proc(cmd)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_main, "_desktop_packaged_executable", lambda d: None)
    monkeypatch.setattr(cli_main, "_desktop_macos_relaunchable_fixup", lambda d: True)
    monkeypatch.setattr("hermes_cli.config.set_config_value", lambda key, value: None)

    assert cli_main._desktop_macos_setup_tcc_identity("Hermes Local Signing") is True

    out = capsys.readouterr().out
    assert "already in keychain" in out
    # No openssl generation, no security import — only find-identity + config.
    assert not any(c[0] == "/usr/bin/openssl" for c in calls)
    assert not any(c[0] == "/usr/bin/security" and c[1] == "import" for c in calls)


def test_setup_tcc_identity_non_macos_skips(tmp_path, monkeypatch, capsys):
    """On non-macOS the setup is a no-op failure (not a crash)."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")

    assert cli_main._desktop_macos_setup_tcc_identity() is False
    assert "macOS-only" in capsys.readouterr().out


def test_cmd_gui_setup_tcc_identity_exits_before_build(tmp_path, monkeypatch):
    """`hermes desktop --setup-tcc-identity` calls the setup and exits 0/1
    without building or launching the app."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch, platform="darwin")

    with patch("hermes_cli.main._desktop_macos_setup_tcc_identity", return_value=True) as mock_setup, \
         patch("hermes_cli.main._run_npm_install_deterministic") as mock_install, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(setup_tcc_identity=True, identity="Hermes Local Signing"))

    assert exc.value.code == 0
    mock_setup.assert_called_once_with("Hermes Local Signing")
    mock_install.assert_not_called()


# --- desktop.* launch options (config.yaml) -------------------------------




# --- Linux launcher entry registration ------------------------------------


@pytest.mark.linux_only
def test_gui_registers_linux_desktop_entry_before_launch(tmp_path, monkeypatch):
    """`hermes desktop` gives the app a launcher presence on Linux."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    registered: list[Path] = []
    monkeypatch.setattr("hermes_cli.linux_desktop_entry.is_supported", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.linux_desktop_entry.install_desktop_entry",
        lambda project_root: registered.append(project_root) or (tmp_path / "hermes.desktop"),
    )

    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok), \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    assert registered == [root]


@pytest.mark.linux_only
def test_gui_launches_even_when_desktop_entry_install_fails(tmp_path, monkeypatch):
    """Launcher plumbing is a convenience — it must never block the app."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    def boom(_project_root):
        raise OSError("read-only /home")

    monkeypatch.setattr("hermes_cli.linux_desktop_entry.is_supported", lambda: True)
    monkeypatch.setattr("hermes_cli.linux_desktop_entry.install_desktop_entry", boom)

    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    assert mock_run.call_args.args[0] == [str(packaged_exe)]


@pytest.mark.macos_only
def test_gui_skips_desktop_entry_off_linux(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    monkeypatch.setattr("hermes_cli.linux_desktop_entry.is_supported", lambda: False)

    def fail(_project_root):
        raise AssertionError("must not install a desktop entry off Linux")

    monkeypatch.setattr("hermes_cli.linux_desktop_entry.install_desktop_entry", fail)

    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok), \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("gnome-libsecret", "gnome-libsecret"),
        ("KWallet6", "kwallet6"),
        ("basic", "basic"),
        ("auto", "auto"),
        ("keychain-of-wonders", "auto"),
        (True, "auto"),
    ],
)
def test_desktop_launch_options_normalizes_password_store(raw, expected):
    cfg = {"desktop": {"password_store": raw}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        _, _, store, _ = cli_main._desktop_launch_options()
    assert store == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("x11", "x11"),
        ("WAYLAND", "wayland"),
        ("auto", "auto"),
        ("bogus", "auto"),
        (True, "auto"),
    ],
)
def test_desktop_launch_options_normalizes_ozone_hint(raw, expected):
    """``desktop.ozone_platform_hint`` normalizes to x11/wayland/auto."""
    cfg = {"desktop": {"ozone_platform_hint": raw}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        _, _, _, hint = cli_main._desktop_launch_options()
    assert hint == expected


def test_desktop_launch_options_ozone_hint_defaults_auto():
    with patch("hermes_cli.config.load_config", return_value={}):
        assert cli_main._desktop_launch_options()[3] == "auto"


def test_gui_bridges_ozone_hint_to_launch_env(tmp_path, monkeypatch):
    """COSMIC HUD: ``desktop.ozone_platform_hint: x11`` sets
    ``ELECTRON_OZONE_PLATFORM_HINT`` on the launched Electron process."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)
    cfg = {"desktop": {"ozone_platform_hint": "x11"}}

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    launch_env = mock_run.call_args_list[1].kwargs["env"]
    assert launch_env.get("ELECTRON_OZONE_PLATFORM_HINT") == "x11"

    monkeypatch.setenv("ELECTRON_OZONE_PLATFORM_HINT", "wayland")
    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run2, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    launch_env = mock_run2.call_args_list[1].kwargs["env"]
    assert launch_env.get("ELECTRON_OZONE_PLATFORM_HINT") == "wayland"


# --- desktop.password_store detection & bridging (linux) ------------------


def _clear_keychain_env(monkeypatch):
    for var in (
        "KDE_SESSION_VERSION",
        "KDE_FULL_SESSION",
        "GNOME_KEYRING_CONTROL",
        "HERMES_DESKTOP_PASSWORD_STORE",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    "kde_version,expected",
    [
        ("6", "kwallet6"),
        ("5", "kwallet5"),
        ("4", "kwallet"),
    ],
)
def test_detect_linux_password_store_prefers_kde_session(monkeypatch, kde_version, expected):
    _clear_keychain_env(monkeypatch)
    monkeypatch.setenv("KDE_SESSION_VERSION", kde_version)
    assert cli_main._detect_linux_password_store() == expected


def test_detect_linux_password_store_kde_full_session(monkeypatch):
    _clear_keychain_env(monkeypatch)
    monkeypatch.setenv("KDE_FULL_SESSION", "true")
    assert cli_main._detect_linux_password_store() == "kwallet"


def test_detect_linux_password_store_gnome_keyring(monkeypatch):
    _clear_keychain_env(monkeypatch)
    monkeypatch.setenv("GNOME_KEYRING_CONTROL", "/run/user/1000/keyring")
    assert cli_main._detect_linux_password_store() == "gnome-libsecret"


def test_detect_linux_password_store_via_dbus_secret_service(monkeypatch):
    _clear_keychain_env(monkeypatch)
    ping_ok = subprocess.CompletedProcess(["dbus-send"], 0)
    with patch("hermes_cli.main.subprocess.run", return_value=ping_ok) as mock_run:
        assert cli_main._detect_linux_password_store() == "gnome-libsecret"
    assert "--dest=org.freedesktop.secrets" in mock_run.call_args.args[0]


def test_detect_linux_password_store_none_when_no_keychain(monkeypatch):
    _clear_keychain_env(monkeypatch)
    ping_fail = subprocess.CompletedProcess(["dbus-send"], 1)
    with patch("hermes_cli.main.subprocess.run", return_value=ping_fail):
        assert cli_main._detect_linux_password_store() is None
    with patch("hermes_cli.main.subprocess.run", side_effect=FileNotFoundError):
        assert cli_main._detect_linux_password_store() is None


@pytest.mark.linux_only
def test_gui_linux_packaged_launch_bridges_detected_password_store(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main._detect_linux_password_store", return_value="gnome-libsecret"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    launch_env = mock_run.call_args_list[1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_PASSWORD_STORE"] == "gnome-libsecret"


@pytest.mark.linux_only
def test_gui_linux_source_launch_bridges_detected_password_store(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main._detect_linux_password_store", return_value="kwallet6"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns(source=True))

    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/npm", "exec", "--", "electron", "."]
    launch_env = mock_run.call_args_list[1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_PASSWORD_STORE"] == "kwallet6"


@pytest.mark.linux_only
def test_gui_config_password_store_skips_detection(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)
    cfg = {"desktop": {"password_store": "kwallet6"}}

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main._detect_linux_password_store") as mock_detect, \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    mock_detect.assert_not_called()
    launch_env = mock_run.call_args_list[1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_PASSWORD_STORE"] == "kwallet6"


@pytest.mark.linux_only
def test_gui_explicit_password_store_env_wins_over_config_and_detection(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    monkeypatch.setenv("HERMES_DESKTOP_PASSWORD_STORE", "basic")
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)
    cfg = {"desktop": {"password_store": "kwallet6"}}

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main._detect_linux_password_store") as mock_detect, \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    mock_detect.assert_not_called()
    launch_env = mock_run.call_args_list[1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_PASSWORD_STORE"] == "basic"


@pytest.mark.macos_only
def test_gui_password_store_bridge_is_linux_only(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main._detect_linux_password_store") as mock_detect, \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    mock_detect.assert_not_called()
    launch_env = mock_run.call_args_list[1].kwargs["env"]
    assert "HERMES_DESKTOP_PASSWORD_STORE" not in launch_env
