"""Tests for ``hermes gui`` desktop launcher wiring."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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



# Dependency admission and staging are exercised by test_desktop_source_build.py.






# ── Content-hash stamp tests ──────────────────────────────────────────








# ── Electron build-cache recovery tests ───────────────────────────────












# ── electronDist (re)download helper tests (#47266) ───────────────────


















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

    assert main_desktop._desktop_macos_local_codesign(app, desktop_dir=desktop_dir) is True

    signed = [c[-1] for c in calls if c[:3] == ["/usr/bin/codesign", "--force", "--sign"]]
    assert str(app / "Contents" / "Resources" / "app.asar.unpacked" / "node_modules" / "pty" / "pty.node") in signed
    assert str(app / "Contents" / "Frameworks" / "chrome_crashpad_handler") in signed




@pytest.mark.platforms("macos")
def test_relaunchable_fixup_falls_back_to_legacy_adhoc_on_failure(tmp_path, monkeypatch, capsys):
    """A failing stable sign must still leave a launchable (deep ad-hoc) bundle.

    The stable signer raising routes into the legacy deep ad-hoc fallback;
    with the fallback sign and strict verification succeeding, the fixup
    reports ``True`` per its documented contract.

    ``platforms("macos")``: the subject is ``codesign`` against a real ``.app`` bundle
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
    monkeypatch.setattr(main_desktop, "_desktop_macos_has_valid_real_signature", lambda a: False)
    monkeypatch.setattr(main_desktop, "_desktop_macos_local_signing_identity", lambda: None)

    def boom(*a, **kw):
        raise subprocess.CalledProcessError(1, ["codesign"])

    monkeypatch.setattr(main_desktop, "_desktop_macos_local_codesign", boom)

    assert cli_main._desktop_macos_relaunchable_fixup(desktop_dir) is True
    assert ["xattr", "-cr", str(app)] in calls
    assert ["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app)] in calls


# --- desktop --setup-tcc-identity ------------------------------------------


def _fake_proc(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def test_setup_tcc_identity_creates_cert_imports_trusts_and_configures(tmp_path, monkeypatch, capsys):
    """Fresh identity: openssl generates, security imports + trusts, config is written."""
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
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

    out = capsys.readouterr().out
    assert "created, imported, and trusted self-signed identity" in out
    assert "set desktop.macos_signing_identity" in out
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


def test_setup_tcc_identity_retries_pkcs12_with_legacy_on_mac_verification_failure(tmp_path, monkeypatch, capsys):
    """OpenSSL 3: first import fails with the MAC-verification signature, the
    -legacy re-export imports cleanly (the exact failure @ctaylor86 hit live)."""
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
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


def test_setup_tcc_identity_fails_when_trust_step_fails(tmp_path, monkeypatch, capsys):
    """A cert that imports but cannot be trusted for codeSign is a failure,
    not a silent success."""
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
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
    assert "could not trust the certificate" in capsys.readouterr().out


def test_setup_tcc_identity_fails_when_identity_never_becomes_valid(tmp_path, monkeypatch, capsys):
    """Postcondition gate: import + trust both 'succeed' but find-identity -v
    still lists nothing → report failure with guidance (the silent-success bug
    from the original PR)."""
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
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
    assert "not a VALID code-signing identity" in capsys.readouterr().out


def test_setup_tcc_identity_skips_generation_when_already_valid(tmp_path, monkeypatch, capsys):
    """Idempotent: an existing VALID identity is reused, not regenerated."""
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
        if cmd[:4] == ["/usr/bin/security", "find-identity", "-v", "-p"]:
            return _fake_proc(cmd, stdout='  1) ABCD "Hermes Local Signing"\n     1 valid identities found')
        return _fake_proc(cmd)

    monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable", lambda d: None)
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda d: True)
    monkeypatch.setattr("hermes_cli.config.set_config_value", lambda key, value: None)

    assert main_desktop._desktop_macos_setup_tcc_identity("Hermes Local Signing") is True

    out = capsys.readouterr().out
    assert "already valid in keychain" in out
    # No openssl generation, no security import — only find-identity + config.
    assert not any(c[0] == "/usr/bin/openssl" for c in calls)
    assert not any(c[0] == "/usr/bin/security" and c[1] == "import" for c in calls)


def test_setup_tcc_identity_untrusted_existing_cert_is_repaired(tmp_path, monkeypatch, capsys):
    """A cert that EXISTS but is not valid (CSSMERR_TP_NOT_TRUSTED) is repaired
    — regenerated/trusted — instead of being reported as already done. The
    original name-in-output probe treated this state as success."""
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
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


def test_setup_tcc_identity_non_macos_skips(tmp_path, monkeypatch, capsys):
    """On non-macOS the setup is a no-op failure (not a crash)."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")

    assert main_desktop._desktop_macos_setup_tcc_identity() is False
    assert "macOS-only" in capsys.readouterr().out


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
    exe = _make_packaged_executable(root, monkeypatch)
    app = exe.parents[2]

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
def test_relaunchable_fixup_default_noconfig_success_never_touches_keychain(tmp_path, monkeypatch):
    """Default no-config path (identity == '-') must not delete the keychain item.

    Witness for the default ad-hoc success path: with no
    ``desktop.macos_signing_identity`` configured, the fixup signs ad-hoc with
    identifier-pinned requirements and must leave the safeStorage item alone.

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
    monkeypatch.setattr(main_desktop, "_desktop_macos_has_valid_real_signature", lambda a: False)
    monkeypatch.setattr(main_desktop, "_desktop_macos_local_signing_identity", lambda: None)
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

    with patch("hermes_cli.main_desktop._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main_install_repair._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main_desktop._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok), \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    assert registered == [root]


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
         patch("hermes_cli.main_install_repair._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
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


@pytest.mark.platforms("macos")
def test_gui_skips_desktop_entry_off_linux(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    monkeypatch.setattr("hermes_cli.linux_desktop_entry.is_supported", lambda: False)

    def fail(_project_root):
        raise AssertionError("must not install a desktop entry off Linux")

    monkeypatch.setattr("hermes_cli.linux_desktop_entry.install_desktop_entry", fail)

    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main_desktop._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main_install_repair._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main_desktop._desktop_macos_relaunchable_fixup"), \
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


def test_desktop_launch_options_ozone_hint_defaults_auto():
    with patch("hermes_cli.config.load_config", return_value={}):
        assert main_desktop._desktop_launch_options()[3] == "auto"


@pytest.mark.platforms("linux")
def test_gui_bridges_ozone_hint_to_launch_env(tmp_path, monkeypatch):
    """COSMIC HUD: ``desktop.ozone_platform_hint: x11`` sets
    ``ELECTRON_OZONE_PLATFORM_HINT`` on the launched Electron process."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)
    cfg = {"desktop": {"ozone_platform_hint": "x11"}}

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.source_build.prepare_source_dependencies", return_value=ok), \
         patch("hermes_cli.main_desktop._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main_desktop._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main_desktop._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main_desktop._detect_linux_password_store", return_value=None), \
         patch("hermes_cli.main.subprocess.run", side_effect=_pack_into_staging(root)) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    launch_env = mock_run.call_args_list[-1].kwargs["env"]
    assert launch_env.get("ELECTRON_OZONE_PLATFORM_HINT") == "x11"

    monkeypatch.setenv("ELECTRON_OZONE_PLATFORM_HINT", "wayland")
    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.source_build.prepare_source_dependencies", return_value=ok), \
         patch("hermes_cli.main_desktop._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main_desktop._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main_desktop._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main_desktop._detect_linux_password_store", return_value=None), \
         patch("hermes_cli.main.subprocess.run", side_effect=_pack_into_staging(root)) as mock_run2, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    launch_env = mock_run2.call_args_list[-1].kwargs["env"]
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


@pytest.mark.platforms("linux")
def test_gui_config_password_store_skips_detection(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)
    cfg = {"desktop": {"password_store": "kwallet6"}}

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.source_build.prepare_source_dependencies", return_value=ok), \
         patch("hermes_cli.main_desktop._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main_desktop._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main_desktop._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main_desktop._detect_linux_password_store") as mock_detect, \
         patch("hermes_cli.main.subprocess.run", side_effect=_pack_into_staging(root)) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    mock_detect.assert_not_called()
    launch_env = mock_run.call_args_list[-1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_PASSWORD_STORE"] == "kwallet6"


@pytest.mark.platforms("linux")
def test_gui_explicit_password_store_env_wins_over_config_and_detection(tmp_path, monkeypatch):
    _clear_keychain_env(monkeypatch)
    monkeypatch.setenv("HERMES_DESKTOP_PASSWORD_STORE", "basic")
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)
    cfg = {"desktop": {"password_store": "kwallet6"}}

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.source_build.prepare_source_dependencies", return_value=ok), \
         patch("hermes_cli.main_desktop._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main_desktop._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main_desktop._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.linux_desktop_entry.install_desktop_entry", return_value=None), \
         patch("hermes_cli.main_desktop._detect_linux_password_store") as mock_detect, \
         patch("hermes_cli.main.subprocess.run", side_effect=_pack_into_staging(root)) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns())

    mock_detect.assert_not_called()
    launch_env = mock_run.call_args_list[-1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_PASSWORD_STORE"] == "basic"


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
    out = capsys.readouterr().out
    assert "Desktop GUI build failed" in out


def test_gui_successful_pack_swaps_new_app_into_release(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(main_desktop, "_ensure_desktop_exe_launchable", lambda _root, exe: (exe, False))
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
    assert "produced no launchable app" in capsys.readouterr().out
