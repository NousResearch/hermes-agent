"""Tests for ``hermes gui`` desktop launcher wiring."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_desktop
from hermes_cli import main_install_repair
from hermes_cli import main_web_build


@pytest.fixture(autouse=True)
def _prepared_build_environment(monkeypatch):
    from hermes_cli import source_build
    monkeypatch.setattr(source_build, "source_build_env", lambda env=None, **kw: {**os.environ, **(env or {})})


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


def _staging_dir_from(cmd) -> Path:
    """Extract the ``-c.directories.output=<dir>`` electron-builder override
    ``cmd_gui`` appends to ``npm run pack`` (stage-and-swap, #86443)."""
    for arg in cmd:
        if isinstance(arg, str) and arg.startswith("-c.directories.output="):
            return Path(arg.split("=", 1)[1])
    raise AssertionError(f"no staging output override in {cmd!r}")


def _packaged_exe_rel() -> Path:
    """Packaged-exe path relative to electron-builder's output dir on THIS host."""
    if sys.platform == "darwin":
        return Path("mac-arm64") / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    if sys.platform == "win32":
        return Path("win-unpacked") / "Hermes.exe"
    return Path("linux-unpacked") / "hermes"


def _pack_into_staging(root: Path, content: str = "", returncode: int = 0):
    """``subprocess.run`` side effect mimicking a real ``npm run pack``: lays
    the packaged app down inside the STAGING dir named on the command line
    (never in release/), then returns *returncode*. Non-pack commands (the
    launch) return success."""
    def _run(cmd, **kwargs):
        if len(cmd) >= 3 and cmd[1:3] == ["run", "builder"]:
            exe = _staging_dir_from(cmd) / _packaged_exe_rel()
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.write_text(content, encoding="utf-8")
            if sys.platform not in ("darwin", "win32"):
                (exe.parent / "chrome-sandbox").write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode)
        return subprocess.CompletedProcess(cmd, 0)
    return _run


@pytest.mark.parametrize("local", [False, True])
def test_source_launch_reads_bom_electron_path_without_provisioning(tmp_path, monkeypatch, local):
    root = _make_desktop_tree(tmp_path)
    desktop = root / "apps" / "desktop"
    (desktop / "dist").mkdir()
    (desktop / "dist" / "index.html").write_text("prepared renderer", encoding="utf-8")
    electron = root / "node_modules" / "electron"
    (electron / "dist").mkdir(parents=True)
    (electron / "package.json").write_text("{}", encoding="utf-8")
    executable = electron / "dist" / "électron"
    executable.touch()
    (electron / "path.txt").write_text(executable.name + "\n", encoding="utf-8-sig")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(main_desktop, "_desktop_launch_env", lambda args: ({}, []))
    monkeypatch.setattr(main_desktop, "_register_linux_desktop_entry", lambda **kw: None)
    calls = []
    monkeypatch.setattr(main_desktop.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    args = _ns(source=True, skip_build=True, local=local)
    with pytest.raises(SystemExit) as exit_info:
        main_desktop.cmd_gui(args)
    assert exit_info.value.code == 0
    assert calls == [[str(executable), ".", *(["--local"] if local else [])]]
    executable.unlink()
    with pytest.raises(SystemExit) as exit_info:
        main_desktop.cmd_gui(args)
    assert exit_info.value.code == 1
    assert len(calls) == 1


def test_packaged_renderer_bom_does_not_bypass_entry_validation(tmp_path):
    import json
    import struct
    from hermes_cli.desktop_update_verify import _verify_packaged_entry

    resources = tmp_path / "resources"
    dist = resources / "app.asar.unpacked" / "dist"
    dist.mkdir(parents=True)
    entry = b"export {};"
    (dist / "main.mjs").write_bytes(entry)
    package = json.dumps({"main": "dist/main.mjs"}).encode("utf-8")
    header = json.dumps({"files": {
        "package.json": {"size": len(package), "offset": "0"},
        "dist": {"files": {"main.mjs": {"size": len(entry), "unpacked": True}}},
    }}).encode("utf-8")
    padded = header + b"\0" * (-len(header) % 4)
    (resources / "app.asar").write_bytes(
        struct.pack("<4I", 4, 8 + len(padded), 4 + len(padded), len(header)) + padded + package)
    index = dist / "index.html"
    index.write_text('<title>café</title><script type="module" src="./main.mjs"></script>',
                     encoding="utf-8-sig")
    _verify_packaged_entry(resources)
    index.write_text("<title>café</title>", encoding="utf-8-sig")
    with pytest.raises(RuntimeError, match="renderer has no local module entry"):
        _verify_packaged_entry(resources)
    index.write_bytes(b"\xef\xbb\xbf\xff")
    with pytest.raises(RuntimeError, match="renderer entry is invalid"):
        _verify_packaged_entry(resources)


# Dependency admission and staging are exercised by test_desktop_source_build.py.


# --- package-manager (Homebrew/pip) installs: no desktop source tree -------
# (#61056: `hermes desktop` under Homebrew looked for apps/desktop inside the
# Cellar site-packages, which the formula does not ship.)


def test_site_packages_install_kind_detects_brew_and_pip():
    brew_root = Path("/opt/homebrew/Cellar/hermes-agent/2026.7.7.2/libexec/site-packages")
    pip_root = Path("/usr/lib/python3/dist-packages")
    venv_root = Path("/home/u/proj/venv/lib/python3.12/site-packages")
    assert main_desktop._site_packages_install_kind(brew_root) == "homebrew"
    assert main_desktop._site_packages_install_kind(pip_root) == "pip"
    assert main_desktop._site_packages_install_kind(venv_root) == "pip"
    assert main_desktop._site_packages_install_kind(Path("/home/u/hermes-agent")) is None


def test_gui_brew_install_prints_brew_guidance_not_venv_hint(tmp_path, monkeypatch, capsys):
    """A Homebrew install has no apps/desktop tree and can never build one — the
    error must say so instead of implying a broken checkout."""
    root = tmp_path / "Cellar" / "hermes-agent" / "2026.7.7.2" / "libexec" / "site-packages"
    root.mkdir(parents=True)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(main_desktop, "_launch_installed_macos_desktop_app", lambda: False)

    with pytest.raises(SystemExit) as exc:
        main_desktop.cmd_gui(_ns())

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Desktop GUI source not found" in out
    assert "Homebrew" in out


def test_gui_brew_install_launches_installed_app_when_present(tmp_path, monkeypatch):
    """Prefer the separately installed /Applications/Hermes.app over failing."""
    root = tmp_path / "Cellar" / "hermes-agent" / "2026.7.7.2" / "libexec" / "site-packages"
    root.mkdir(parents=True)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    launched = []
    monkeypatch.setattr(main_desktop, "_launch_installed_macos_desktop_app", lambda: launched.append(1) or True)

    with pytest.raises(SystemExit) as exc:
        main_desktop.cmd_gui(_ns())

    assert exc.value.code == 0
    assert launched == [1]


@pytest.mark.parametrize("exists,platform", [(True, "darwin"), (False, "darwin"), (True, "linux")])
def test_launch_installed_macos_desktop_app_gates_on_bundle_and_platform(tmp_path, monkeypatch, exists, platform):
    monkeypatch.setattr(main_desktop.sys, "platform", platform)
    exe = Path("/Applications/Hermes.app/Contents/MacOS/Hermes")
    monkeypatch.setattr(main_desktop.Path, "is_file", lambda self: exists if self == exe else Path.is_file(self))
    calls = []
    if exists and platform == "darwin":
        import hermes_cli.bundled_app as bundled_app
        monkeypatch.setattr(bundled_app, "launch_detached", lambda argv, **kw: calls.append(argv) or 4321)

    assert main_desktop._launch_installed_macos_desktop_app() is (exists and platform == "darwin")
    if exists and platform == "darwin":
        assert calls == [[str(exe)]]


# ── Content-hash stamp tests ──────────────────────────────────────────


# ── Electron build-cache recovery tests ───────────────────────────────


# ── electronDist (re)download helper tests (#47266) ───────────────────





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

    assert main_desktop._desktop_macos_local_codesign(app, desktop_dir=desktop_dir) is True

    signed = [c[-1] for c in calls if c[:3] == ["/usr/bin/codesign", "--force", "--sign"]]
    assert str(app / "Contents" / "Resources" / "app.asar.unpacked" / "node_modules" / "pty" / "pty.node") in signed
    assert str(app / "Contents" / "Frameworks" / "chrome_crashpad_handler") in signed




# --- desktop --setup-tcc-identity ------------------------------------------


def _fake_proc(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


@pytest.mark.platforms("macos")
def test_setup_tcc_identity_creates_cert_imports_trusts_and_configures(tmp_path, monkeypatch, capsys):
    """Fresh identity: openssl generates, security imports + trusts, config is written."""
    monkeypatch.setattr(
        cli_main.shutil,
        "which",
        lambda name: {"openssl": "/usr/bin/openssl", "security": "/usr/bin/security", "codesign": "/usr/bin/codesign"}.get(name),
    )
    monkeypatch.setattr(cli_main.Path, "home", classmethod(lambda cls: tmp_path))

    identity = "Hermes Local Signing"
    calls = []
    state = {"trusted": False}

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:4] == ["/usr/bin/security", "find-identity", "-v", "-p"]:
            # Valid only after import AND trust have both happened — mirrors
            # real macOS, where an untrusted self-signed cert is invisible to
            # the -v listing (the #77189 review finding).
            if state["trusted"]:
                return _fake_proc(cmd, stdout=f'  1) ABCD "{identity}"\n     1 valid identities found')
            return _fake_proc(cmd, stdout="     0 valid identities found")
        if cmd[0] == "/usr/bin/security" and cmd[1] == "add-trusted-cert":
            state["trusted"] = True
            return _fake_proc(cmd)
        return _fake_proc(cmd)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable", lambda d: None)
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda d: True)
    # Avoid writing the real user config.
    monkeypatch.setattr("hermes_cli.config.set_config_value", lambda key, value: None)

    assert main_desktop._desktop_macos_setup_tcc_identity(identity) is True

    # openssl cert generation + pkcs12 export + security import + trust all ran.
    assert any(c[0] == "/usr/bin/openssl" and "req" in c for c in calls)
    assert any(c[0] == "/usr/bin/openssl" and "pkcs12" in c for c in calls)
    assert any(c[0] == "/usr/bin/security" and c[1] == "import" for c in calls)
    assert any(c[0] == "/usr/bin/security" and c[1] == "add-trusted-cert" for c in calls)
    # The trust step targets the codeSign policy specifically.
    trust_call = next(c for c in calls if c[1:2] == ["add-trusted-cert"])
    assert "codeSign" in trust_call and "trustRoot" in trust_call
    # Temp files cleaned up.
    assert not list(tmp_path.glob("hermes-tcc-*"))


@pytest.mark.platforms("macos")
def test_setup_tcc_identity_retries_pkcs12_with_legacy_on_mac_verification_failure(tmp_path, monkeypatch, capsys):
    """OpenSSL 3: first import fails with the MAC-verification signature, the
    -legacy re-export imports cleanly (the exact failure @ctaylor86 hit live)."""
    monkeypatch.setattr(
        cli_main.shutil,
        "which",
        lambda name: {"openssl": "/usr/bin/openssl", "security": "/usr/bin/security", "codesign": "/usr/bin/codesign"}.get(name),
    )
    monkeypatch.setattr(cli_main.Path, "home", classmethod(lambda cls: tmp_path))

    identity = "Hermes Local Signing"
    calls = []
    state = {"legacy_exported": False, "trusted": False}

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:4] == ["/usr/bin/security", "find-identity", "-v", "-p"]:
            if state["trusted"]:
                return _fake_proc(cmd, stdout=f'  1) ABCD "{identity}"\n     1 valid identities found')
            return _fake_proc(cmd, stdout="     0 valid identities found")
        if cmd[0] == "/usr/bin/openssl" and "pkcs12" in cmd:
            state["legacy_exported"] = "-legacy" in cmd
            return _fake_proc(cmd)
        if cmd[0] == "/usr/bin/security" and cmd[1] == "import":
            if not state["legacy_exported"]:
                return _fake_proc(
                    cmd, returncode=1,
                    stderr="security: SecKeychainItemImport: MAC verification failed during PKCS12 import (wrong password?)",
                )
            return _fake_proc(cmd)
        if cmd[0] == "/usr/bin/security" and cmd[1] == "add-trusted-cert":
            state["trusted"] = True
            return _fake_proc(cmd)
        return _fake_proc(cmd)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable", lambda d: None)
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda d: True)
    monkeypatch.setattr("hermes_cli.config.set_config_value", lambda key, value: None)

    assert main_desktop._desktop_macos_setup_tcc_identity(identity) is True

    # Two pkcs12 exports (plain then -legacy) and two import attempts.
    pkcs12_calls = [c for c in calls if c[0] == "/usr/bin/openssl" and "pkcs12" in c]
    assert len(pkcs12_calls) == 2
    assert "-legacy" not in pkcs12_calls[0] and "-legacy" in pkcs12_calls[1]
    assert len([c for c in calls if c[0] == "/usr/bin/security" and c[1] == "import"]) == 2


@pytest.mark.platforms("macos")
def test_setup_tcc_identity_fails_when_trust_step_fails(tmp_path, monkeypatch, capsys):
    """A cert that imports but cannot be trusted for codeSign is a failure,
    not a silent success."""
    monkeypatch.setattr(
        cli_main.shutil,
        "which",
        lambda name: {"openssl": "/usr/bin/openssl", "security": "/usr/bin/security", "codesign": "/usr/bin/codesign"}.get(name),
    )
    monkeypatch.setattr(cli_main.Path, "home", classmethod(lambda cls: tmp_path))

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["/usr/bin/security", "find-identity", "-v", "-p"]:
            return _fake_proc(cmd, stdout="     0 valid identities found")
        if cmd[0] == "/usr/bin/security" and cmd[1] == "add-trusted-cert":
            return _fake_proc(cmd, returncode=1, stderr="SecTrustSettingsSetTrustSettings: authorization denied")
        return _fake_proc(cmd)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)

    assert main_desktop._desktop_macos_setup_tcc_identity("Hermes Local Signing") is False


@pytest.mark.platforms("macos")
def test_setup_tcc_identity_fails_when_identity_never_becomes_valid(tmp_path, monkeypatch, capsys):
    """Postcondition gate: import + trust both 'succeed' but find-identity -v
    still lists nothing → report failure with guidance (the silent-success bug
    from the original PR)."""
    monkeypatch.setattr(
        cli_main.shutil,
        "which",
        lambda name: {"openssl": "/usr/bin/openssl", "security": "/usr/bin/security", "codesign": "/usr/bin/codesign"}.get(name),
    )
    monkeypatch.setattr(cli_main.Path, "home", classmethod(lambda cls: tmp_path))

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["/usr/bin/security", "find-identity", "-v", "-p"]:
            return _fake_proc(cmd, stdout="     0 valid identities found")
        return _fake_proc(cmd)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)

    assert main_desktop._desktop_macos_setup_tcc_identity("Hermes Local Signing") is False


@pytest.mark.platforms("macos")
def test_setup_tcc_identity_skips_generation_when_already_valid(tmp_path, monkeypatch, capsys):
    """Idempotent: an existing VALID identity is reused, not regenerated."""
    monkeypatch.setattr(
        cli_main.shutil,
        "which",
        lambda name: {"openssl": "/usr/bin/openssl", "security": "/usr/bin/security", "codesign": "/usr/bin/codesign"}.get(name),
    )
    monkeypatch.setattr(cli_main.Path, "home", classmethod(lambda cls: tmp_path))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:4] == ["/usr/bin/security", "find-identity", "-v", "-p"]:
            return _fake_proc(cmd, stdout='  1) ABCD "Hermes Local Signing"\n     1 valid identities found')
        return _fake_proc(cmd)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable", lambda d: None)
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda d: True)
    monkeypatch.setattr("hermes_cli.config.set_config_value", lambda key, value: None)

    assert main_desktop._desktop_macos_setup_tcc_identity("Hermes Local Signing") is True

    # No openssl generation, no security import — only find-identity + config.
    assert not any(c[0] == "/usr/bin/openssl" for c in calls)
    assert not any(c[0] == "/usr/bin/security" and c[1] == "import" for c in calls)


@pytest.mark.platforms("macos")
def test_setup_tcc_identity_untrusted_existing_cert_is_repaired(tmp_path, monkeypatch, capsys):
    """A cert that EXISTS but is not valid (CSSMERR_TP_NOT_TRUSTED) is repaired
    — regenerated/trusted — instead of being reported as already done. The
    original name-in-output probe treated this state as success."""
    monkeypatch.setattr(
        cli_main.shutil,
        "which",
        lambda name: {"openssl": "/usr/bin/openssl", "security": "/usr/bin/security", "codesign": "/usr/bin/codesign"}.get(name),
    )
    monkeypatch.setattr(cli_main.Path, "home", classmethod(lambda cls: tmp_path))

    calls = []
    state = {"trusted": False}

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:4] == ["/usr/bin/security", "find-identity", "-v", "-p"]:
            # -v never lists the untrusted cert; it only appears once the
            # repair path has run add-trusted-cert.
            if state["trusted"]:
                return _fake_proc(cmd, stdout='  1) ABCD "Hermes Local Signing"\n     1 valid identities found')
            return _fake_proc(cmd, stdout="     0 valid identities found")
        if cmd[0] == "/usr/bin/security" and cmd[1] == "add-trusted-cert":
            state["trusted"] = True
            return _fake_proc(cmd)
        return _fake_proc(cmd)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable", lambda d: None)
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda d: True)
    monkeypatch.setattr("hermes_cli.config.set_config_value", lambda key, value: None)

    assert main_desktop._desktop_macos_setup_tcc_identity("Hermes Local Signing") is True
    assert any(c[0] == "/usr/bin/security" and c[1] == "add-trusted-cert" for c in calls)




def test_cmd_gui_setup_tcc_identity_exits_before_build(tmp_path, monkeypatch):
    """`hermes desktop --setup-tcc-identity` calls the setup and exits 0/1
    without building or launching the app."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    with patch("hermes_cli.main_desktop._desktop_macos_setup_tcc_identity", return_value=True) as mock_setup, \
         patch("hermes_cli.source_build.prepare_source_dependencies") as mock_install, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(setup_tcc_identity=True, identity="Hermes Local Signing"))

    assert exc.value.code == 0
    mock_setup.assert_called_once_with("Hermes Local Signing")
    mock_install.assert_not_called()


@pytest.mark.platforms("macos")
def test_relaunchable_fixup_stable_identity_never_touches_keychain(tmp_path, monkeypatch):
    """A successful stable-identity re-sign must NOT delete the safeStorage item.

    Regression for review feedback on #90961: deleting the keychain item
    permanently orphans every safeStorage-backed credential (gateway token,
    native OAuth access/refresh tokens — see electron/main.ts). On the stable
    path the cert-anchored designated requirement is stable across rebuilds,
    so after the first launch the keychain ACL already matches and deleting
    the item would destroy working credentials on every update.

    ``platforms("macos")``: the fixup no-ops on non-macOS (sys.platform guard), and
    the subject is codesign against a real ``.app`` bundle layout.
    """
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.delenv("CSC_LINK", raising=False)
    monkeypatch.delenv("APPLE_SIGNING_IDENTITY", raising=False)
    _make_packaged_executable(root, monkeypatch)

    calls: list[list[str]] = []
    monkeypatch.setattr(main_desktop, "_desktop_macos_has_valid_real_signature", lambda a: False)
    monkeypatch.setattr(main_desktop, "_desktop_macos_local_signing_identity", lambda: "Developer ID Application: Example"
    )
    monkeypatch.setattr(main_desktop, "_desktop_macos_local_codesign", lambda app, **kw: True)
    monkeypatch.setattr(
        cli_main.subprocess, "run",
        lambda cmd, **kw: calls.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
    )

    assert cli_main._desktop_macos_relaunchable_fixup(desktop_dir) is True
    assert not any("delete-generic-password" in c for c in calls)




@pytest.mark.platforms("macos")
def test_relaunchable_fixup_legacy_adhoc_failure_never_touches_keychain(tmp_path, monkeypatch):
    """A failed fallback re-sign must preserve the keychain item (no deletion).

    Regression for review feedback on #90961: the fallback previously deleted
    the safeStorage item unconditionally, even when ``codesign`` failed
    (``check=False`` result was ignored). A failed recovery can permanently
    orphan gateway and native OAuth credentials without producing a verified
    successor app/key identity. The fixup must check the codesign result,
    run strict verification, and leave the keychain untouched on failure.

    ``platforms("macos")``: the fixup no-ops on non-macOS (sys.platform guard), and
    the subject is codesign against a real ``.app`` bundle layout.
    """
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.delenv("CSC_LINK", raising=False)
    monkeypatch.delenv("APPLE_SIGNING_IDENTITY", raising=False)
    exe = _make_packaged_executable(root, monkeypatch)
    app = exe.parents[2]

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        # First subprocess call is the xattr clear (exit 0); the deep sign
        # fails with a non-zero exit.
        if cmd[:2] == ["/usr/bin/codesign", "--force"]:
            return subprocess.CompletedProcess(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        cli_main.shutil, "which", lambda name: "/usr/bin/codesign" if name == "codesign" else None
    )
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(main_desktop, "_desktop_macos_has_valid_real_signature", lambda a: False)
    monkeypatch.setattr(main_desktop, "_desktop_macos_local_signing_identity", lambda: None)

    def boom(*a, **kw):
        raise subprocess.CalledProcessError(1, ["codesign"])

    monkeypatch.setattr(main_desktop, "_desktop_macos_local_codesign", boom)

    assert cli_main._desktop_macos_relaunchable_fixup(desktop_dir) is False
    assert ["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app)] in calls
    assert not any("--verify" in c for c in calls)
    assert not any("delete-generic-password" in c for c in calls)


@pytest.mark.platforms("macos")
def test_relaunchable_fixup_legacy_adhoc_success_still_verifies_and_never_deletes(tmp_path, monkeypatch):
    """A successful fallback re-sign runs strict verification, no deletion.

    The legacy ad-hoc fallback signs, verifies with
    ``codesign --verify --deep --strict``, and leaves the safeStorage keychain
    item untouched. The keychain prompt macOS shows instead is recoverable
    ("Always Allow" updates the ACL partition list and preserves the key);
    deletion is not.

    ``platforms("macos")``: the fixup no-ops on non-macOS (sys.platform guard), and
    the subject is codesign against a real ``.app`` bundle layout.
    """
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.delenv("CSC_LINK", raising=False)
    monkeypatch.delenv("APPLE_SIGNING_IDENTITY", raising=False)
    exe = _make_packaged_executable(root, monkeypatch)
    app = exe.parents[2]

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        cli_main.shutil, "which", lambda name: "/usr/bin/codesign" if name == "codesign" else None
    )
    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(main_desktop, "_desktop_macos_has_valid_real_signature", lambda a: False)
    monkeypatch.setattr(main_desktop, "_desktop_macos_local_signing_identity", lambda: None)

    def boom(*a, **kw):
        raise subprocess.CalledProcessError(1, ["codesign"])

    monkeypatch.setattr(main_desktop, "_desktop_macos_local_codesign", boom)

    assert cli_main._desktop_macos_relaunchable_fixup(desktop_dir) is True
    assert ["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app)] in calls
    assert ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)] in calls
    assert not any("delete-generic-password" in c for c in calls)


# --- desktop.* launch options (config.yaml) -------------------------------


# --- Linux launcher entry registration ------------------------------------


@pytest.mark.platforms("linux")
def test_gui_registers_linux_desktop_entry_before_launch(tmp_path, monkeypatch):
    """A terminal launch (no DESKTOP_STARTUP_ID) still installs the entry before spawning Electron."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.delenv("DESKTOP_STARTUP_ID", raising=False)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    registered: list[Path] = []
    monkeypatch.setattr("hermes_cli.linux_desktop_entry.is_supported", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.linux_desktop_entry.install_desktop_entry",
        lambda project_root: registered.append(project_root) or (tmp_path / "hermes.desktop"),
    )

    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main_desktop._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main_desktop._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok), \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    assert registered == [root]


@pytest.mark.platforms("linux")
def test_gui_shell_launch_defers_desktop_entry_until_window_reveal(tmp_path, monkeypatch):
    """An app-grid launch (DESKTOP_STARTUP_ID set) writes the entry only after Electron reports
    its window on screen — never before the spawn, while gnome-shell has the app in STARTING
    (#111906). Electron gets the pipe's write end via HERMES_DESKTOP_READY_FD."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setenv("DESKTOP_STARTUP_ID", "gnome-shell/Hermes/1-0_TIME1")
    monkeypatch.setattr("hermes_cli.linux_desktop_entry.time.sleep", lambda _s: None)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    events: list[str] = []
    monkeypatch.setattr("hermes_cli.linux_desktop_entry.is_supported", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.linux_desktop_entry.install_desktop_entry",
        lambda project_root: events.append(f"install:{project_root}") or (tmp_path / "hermes.desktop"),
    )

    def fake_electron(cmd, **kwargs):
        events.append("spawn")
        fd = int(kwargs["env"]["HERMES_DESKTOP_READY_FD"])
        assert fd in kwargs["pass_fds"]
        os.write(fd, b"r")  # main window revealed
        deadline = time.monotonic() + 10
        while not any(e.startswith("install:") for e in events) and time.monotonic() < deadline:
            time.sleep(0.01)
        return subprocess.CompletedProcess(cmd, 0)

    with patch("hermes_cli.main_desktop._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main_desktop._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main.subprocess.run", side_effect=fake_electron), \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    assert events == ["spawn", f"install:{root}"]
    assert packaged_exe.exists()


@pytest.mark.platforms("linux")
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

    with patch("hermes_cli.main_desktop._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main_desktop._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    launched = mock_run.call_args.args[0]
    if sys.platform.startswith("linux"):
        assert launched == [str(packaged_exe), "--disable-setuid-sandbox"]
    else:
        assert launched == [str(packaged_exe)]



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
        _, _, store, _ = main_desktop._desktop_launch_options()
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
        _, _, _, hint = main_desktop._desktop_launch_options()
    assert hint == expected




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
    assert main_desktop._detect_linux_password_store() == expected


def test_detect_linux_password_store_kde_full_session(monkeypatch):
    _clear_keychain_env(monkeypatch)
    monkeypatch.setenv("KDE_FULL_SESSION", "true")
    assert main_desktop._detect_linux_password_store() == "kwallet"


def test_detect_linux_password_store_gnome_keyring(monkeypatch):
    _clear_keychain_env(monkeypatch)
    monkeypatch.setenv("GNOME_KEYRING_CONTROL", "/run/user/1000/keyring")
    assert main_desktop._detect_linux_password_store() == "gnome-libsecret"


def test_detect_linux_password_store_via_dbus_secret_service(monkeypatch):
    _clear_keychain_env(monkeypatch)
    ping_ok = subprocess.CompletedProcess(["dbus-send"], 0)
    with patch("hermes_cli.main.subprocess.run", return_value=ping_ok) as mock_run:
        assert main_desktop._detect_linux_password_store() == "gnome-libsecret"
    assert "--dest=org.freedesktop.secrets" in mock_run.call_args.args[0]


def test_detect_linux_password_store_none_when_no_keychain(monkeypatch):
    _clear_keychain_env(monkeypatch)
    ping_fail = subprocess.CompletedProcess(["dbus-send"], 1)
    with patch("hermes_cli.main.subprocess.run", return_value=ping_fail):
        assert main_desktop._detect_linux_password_store() is None
    with patch("hermes_cli.main.subprocess.run", side_effect=FileNotFoundError):
        assert main_desktop._detect_linux_password_store() is None


@pytest.mark.platforms("linux")
def test_gui_linux_packaged_launch_bridges_detected_password_store(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.source_build.prepare_source_dependencies", return_value=ok), \
         patch("hermes_cli.main_desktop._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main_desktop._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main_desktop._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main_desktop._detect_linux_password_store", return_value="gnome-libsecret"), \
         patch("hermes_cli.main.subprocess.run", side_effect=_pack_into_staging(root)) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    launch_env = mock_run.call_args_list[-1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_PASSWORD_STORE"] == "gnome-libsecret"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("explicit,configured,detected,expected", [
    (None, "auto", "gnome-libsecret", "gnome-libsecret"),
    (None, "kwallet6", None, "kwallet6"),
    ("basic", "kwallet6", None, "basic"),
])
def test_desktop_environment_precedence(monkeypatch, explicit, configured, detected, expected):
    _clear_keychain_env(monkeypatch)
    monkeypatch.delenv('ELECTRON_OZONE_PLATFORM_HINT', raising=False)
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: {
        'desktop': {'password_store': configured, 'ozone_platform_hint': 'x11'}})
    def detect():
        assert configured == 'auto' and explicit is None
        return detected
    monkeypatch.setattr(main_desktop, '_detect_linux_password_store', detect)
    if explicit:
        monkeypatch.setenv('HERMES_DESKTOP_PASSWORD_STORE', explicit)
    env, _ = main_desktop._desktop_launch_env(_ns())
    assert env['HERMES_DESKTOP_PASSWORD_STORE'] == expected
    assert env['ELECTRON_OZONE_PLATFORM_HINT'] == 'x11'
    monkeypatch.setenv('ELECTRON_OZONE_PLATFORM_HINT', 'wayland')
    assert main_desktop._desktop_launch_env(_ns())[0]['ELECTRON_OZONE_PLATFORM_HINT'] == 'wayland'


@pytest.mark.platforms("linux")
def test_gui_linux_source_launch_bridges_detected_password_store(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    electron = root / "node_modules/electron"
    (electron / "dist").mkdir(parents=True)
    (electron / "path.txt").write_text("electron")
    (electron / "dist/electron").touch()

    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.source_build.prepare_source_dependencies", return_value=ok), \
         patch("hermes_cli.main_desktop._desktop_build_needed", return_value=True), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main_desktop._detect_linux_password_store", return_value="kwallet6"), \
         patch("hermes_cli.main.subprocess.run", side_effect=_pack_into_staging(root)) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns(source=True))

    assert mock_run.call_args_list[-1].args[0] == [str(electron / "dist/electron"), "."]
    launch_env = mock_run.call_args_list[-1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_PASSWORD_STORE"] == "kwallet6"


@pytest.mark.platforms("macos")
def test_gui_password_store_bridge_is_linux_only(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.source_build.prepare_source_dependencies", return_value=ok), \
         patch("hermes_cli.main_desktop._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main_desktop._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main_desktop._detect_linux_password_store") as mock_detect, \
         patch("hermes_cli.main.subprocess.run", side_effect=_pack_into_staging(root)) as mock_run, \
         pytest.raises(SystemExit):
            cli_main.cmd_gui(_ns())

    mock_detect.assert_not_called()
    launch_env = mock_run.call_args_list[-1].kwargs["env"]
    assert "HERMES_DESKTOP_PASSWORD_STORE" not in launch_env


# ---------------------------------------------------------------------------
# #86443: stage-and-swap — a failed Desktop rebuild must never remove the
# working app. electron-builder packs IN PLACE (before-pack.mjs wipes
# release/<unpacked> first), so cmd_gui now packs into a staging dir and only
# renames it over release/ after the staged result verifies.
# ---------------------------------------------------------------------------


def _gui_build_patches(root: Path, run_side_effect):
    return [
        patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"),
        # Staging doubles are text; the PE-validation suite owns real binaries.
        patch("hermes_cli.main_desktop._desktop_exe_integrity_error", return_value=None),
        patch("hermes_cli.source_build.prepare_source_dependencies",
              return_value=subprocess.CompletedProcess(["npm", "ci"], 0)),
        patch("hermes_cli.main_desktop._desktop_build_needed", return_value=True),
        patch("hermes_cli.main_desktop._desktop_macos_relaunchable_fixup"),
        patch("hermes_cli.main_desktop._register_linux_desktop_entry"),
        patch("hermes_cli.main_desktop._stop_desktop_processes_locking_build", return_value=[]),
        patch("hermes_cli.main.subprocess.run", side_effect=run_side_effect),
    ]


def test_swap_staged_desktop_app_promotes_staged_tree_and_drops_previous(tmp_path):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    staged_exe = staging / _packaged_exe_rel()
    staged_exe.parent.mkdir(parents=True)
    staged_exe.write_text("new", encoding="utf-8")

    promoted = main_desktop._swap_staged_desktop_app(desktop_dir, staging)

    assert promoted == live_exe
    assert live_exe.read_text(encoding="utf-8") == "new"
    assert not staging.exists()
    assert sorted(p.name for p in (desktop_dir / "release").iterdir()) == [_packaged_exe_rel().parts[0]]


def test_swap_staged_desktop_app_without_staged_exe_keeps_live_app(tmp_path):
    """Zero-exit pack that produced nothing: live app untouched, staging gone."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    (staging / "linux-unpacked" / "resources").mkdir(parents=True)  # partial tree, no exe

    assert main_desktop._swap_staged_desktop_app(desktop_dir, staging) is None
    assert live_exe.read_text(encoding="utf-8") == "old"
    assert not staging.exists()


def test_swap_staged_desktop_app_rolls_back_when_second_rename_fails(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    staged_exe = staging / _packaged_exe_rel()
    staged_exe.parent.mkdir(parents=True)
    staged_exe.write_text("new", encoding="utf-8")

    real_rename = cli_main.os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # staged → live
            raise OSError("EXDEV simulated")
        return real_rename(src, dst)

    monkeypatch.setattr(cli_main.os, "rename", flaky_rename)
    assert main_desktop._swap_staged_desktop_app(desktop_dir, staging) is None
    assert live_exe.read_text(encoding="utf-8") == "old"
    assert not (live_exe.parent.parent / (live_exe.parent.name + ".previous")).exists()


def test_swap_staged_desktop_app_stops_live_renderer_before_rename(tmp_path):
    """#109643: a renderer alive through the promotion rename keeps fetching its
    old hashed chunks from disk and dies on the next lazy import — the swap must
    ask for running desktop processes to stop on EVERY platform."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    staged_exe = staging / _packaged_exe_rel()
    staged_exe.parent.mkdir(parents=True)
    staged_exe.write_text("new", encoding="utf-8")

    with patch("hermes_cli.main_desktop._stop_desktop_processes_locking_build",
               return_value=[4321]) as stop:
        promoted = main_desktop._swap_staged_desktop_app(desktop_dir, staging)

    assert promoted == live_exe
    stop.assert_called_once_with(desktop_dir, also_posix=True)


def test_stop_desktop_processes_locking_build_posix_swap_bypasses_early_return(tmp_path, monkeypatch):
    """#109643: also_posix=True must run the scan on POSIX (the default pack-time
    call stays Windows-only — the staging pack never touches the live tree)."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")

    class _FakeProc:
        def __init__(self, pid, exe):
            self.info = {"pid": pid, "exe": exe}
            self.pid = pid

        def terminate(self):
            return None

    target = _FakeProc(100, str(live_exe))
    outsider = _FakeProc(200, "/usr/bin/unrelated")

    class _FakePsutil:
        @staticmethod
        def process_iter(attrs):
            return [target, outsider]

        @staticmethod
        def wait_procs(victims, timeout=5):
            return [], []

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)

    assert main_desktop._stop_desktop_processes_locking_build(desktop_dir) == []
    assert main_desktop._stop_desktop_processes_locking_build(desktop_dir, also_posix=True) == [100]


@pytest.mark.platforms("posix")  # Windows must stop the exe-locking ancestor too
def test_posix_swap_spares_the_desktop_driving_this_update(tmp_path, monkeypatch):
    """A historical Desktop runs `hermes update` as a piped child and relaunches
    itself afterwards; stopping it breaks the update's stdout (EPIPE). Its
    renderer/GPU/zygote helpers run the same exe but are not our ancestors;
    stopping them leaves a main process that can neither draw nor quit. Only an
    unrelated Desktop from the same release tree is stopped."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")

    class _FakeProc:
        def __init__(self, pid, exe, children=()):
            self.info = {"pid": pid, "exe": exe}
            self.pid = pid
            self._children = list(children)

        def exe(self):
            return self.info["exe"]

        def children(self, recursive=False):
            assert recursive
            return self._children

        def terminate(self):
            return None

    renderer = _FakeProc(101, str(live_exe))
    gpu = _FakeProc(102, str(live_exe))
    driver = _FakeProc(100, str(live_exe), children=[renderer, gpu])
    # A non-Desktop ancestor's tree (think init) must not spare everything.
    shell = _FakeProc(1, "/usr/bin/bash", children=[driver, renderer, gpu, _FakeProc(300, str(live_exe))])
    other = shell._children[-1]

    class _FakePsutil:
        @staticmethod
        def Process(pid):
            assert pid == os.getpid()
            return types.SimpleNamespace(parents=lambda: [driver, shell])

        @staticmethod
        def process_iter(attrs):
            return [driver, renderer, gpu, other]

        @staticmethod
        def wait_procs(victims, timeout=5):
            return [], []

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)

    assert main_desktop._stop_desktop_processes_locking_build(desktop_dir, also_posix=True) == [300]


def test_gui_failed_pack_leaves_previous_app_untouched(tmp_path, monkeypatch, capsys):
    """Every pack attempt fails → the pre-existing app is exactly as it was,
    no staging dir remains, exit is non-zero."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    live_exe = _make_packaged_executable(root, monkeypatch)
    live_exe.write_text("good build", encoding="utf-8")
    monkeypatch.setenv("ELECTRON_MIRROR", "https://example.test/electron/")

    def failing_pack(cmd, **kwargs):
        if cmd[1:3] != ["run", "builder"]:
            return subprocess.CompletedProcess(cmd, 0)
        # Mimic before-pack.mjs wiping appOutDir inside the OUTPUT dir it was
        # given, then dying (corrupt Electron zip → ENOENT on rename).
        out = _staging_dir_from(cmd) / _packaged_exe_rel().parts[0]
        out.mkdir(parents=True, exist_ok=True)
        (out / "resources").mkdir(exist_ok=True)
        raise subprocess.CalledProcessError(1, cmd)

    patches = _gui_build_patches(root, failing_pack)
    for p in patches:
        p.start()
    try:
        with pytest.raises(SystemExit) as exc:
            cli_main.cmd_gui(_ns(build_only=True))
    finally:
        for p in patches:
            p.stop()

    assert exc.value.code == 1
    assert live_exe.read_text(encoding="utf-8") == "good build"
    assert not list(desktop_dir.glob(".staging-*"))
    assert not list((desktop_dir / "release").glob("*.previous"))


def test_gui_successful_pack_swaps_new_app_into_release(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(main_desktop, "_desktop_exe_integrity_error", lambda exe: None)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    live_exe = _make_packaged_executable(root, monkeypatch)
    live_exe.write_text("old build", encoding="utf-8")

    patches = _gui_build_patches(root, _pack_into_staging(root, content="new build"))
    for p in patches:
        p.start()
    try:
        cli_main.cmd_gui(_ns(build_only=True))
    finally:
        for p in patches:
            p.stop()

    assert live_exe.read_text(encoding="utf-8") == "new build"
    assert not list(desktop_dir.glob(".staging-*"))
    assert not list((desktop_dir / "release").glob("*.previous"))


def test_gui_zero_exit_pack_without_artifact_keeps_previous_app(tmp_path, monkeypatch, capsys):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    live_exe = _make_packaged_executable(root, monkeypatch)
    live_exe.write_text("good build", encoding="utf-8")

    def empty_pack(cmd, **kwargs):
        if cmd[1:3] == ["run", "builder"]:
            _staging_dir_from(cmd).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0)

    patches = _gui_build_patches(root, empty_pack)
    for p in patches:
        p.start()
    try:
        with pytest.raises(SystemExit) as exc:
            cli_main.cmd_gui(_ns(build_only=True))
    finally:
        for p in patches:
            p.stop()

    assert exc.value.code == 1
    assert live_exe.read_text(encoding="utf-8") == "good build"
    assert not list(desktop_dir.glob(".staging-*"))
