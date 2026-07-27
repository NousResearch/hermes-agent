"""Behavior tests for the Windows desktop-exe integrity gate (#69179).

The desktop self-update chain (Desktop → hermes-setup --update →
``hermes update`` → ``hermes desktop --build-only`` → relaunch) rebuilds
Hermes.exe on the end user's machine. Before this gate, "build succeeded" was
just "the file exists", so a truncated PE (corrupt cached Electron zip), a
non-PE file, or a wrong-architecture tree shipped as the new app — Windows
then refuses to launch it with "This app can't run on your computer"
(此应用无法在你的电脑上运行) and the user has no working install left.

These tests exercise the behavior contract only: synthetic PE files go in,
verdicts/rollbacks come out.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main

PE_AMD64 = 0x8664
PE_ARM64 = 0xAA64
PE_I386 = 0x014C


def make_pe(path: Path, machine: int = PE_AMD64, *, truncate_to: int | None = None) -> Path:
    """Write a minimal, structurally-complete PE file with one section.

    Layout: DOS header (e_lfanew=0x80) → PE signature + COFF header at 0x80 →
    one 40-byte section entry whose raw data spans [0x200, 0x400). Total file
    size 0x400 unless ``truncate_to`` cuts it short (the corrupt-download
    shape).
    """
    buf = bytearray(0x400)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x80)
    buf[0x80:0x84] = b"PE\x00\x00"
    # COFF: machine, nsections=1, timestamp, symtab ptr, nsyms, opt-hdr size=0, chars
    struct.pack_into("<HHIIIHH", buf, 0x84, machine, 1, 0, 0, 0, 0, 0x0002)
    # Section table entry at 0x80 + 24 = 0x98; SizeOfRawData @+16, PointerToRawData @+20
    section_off = 0x98
    struct.pack_into("<II", buf, section_off + 16, 0x200, 0x200)
    data = bytes(buf if truncate_to is None else buf[:truncate_to])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ─── _parse_pe_machine ──────────────────────────────────────────────────────


def test_parse_pe_machine_reads_machine_field(tmp_path):
    exe = make_pe(tmp_path / "Hermes.exe", PE_AMD64)
    assert cli_main._parse_pe_machine(exe) == PE_AMD64

    exe_arm = make_pe(tmp_path / "arm.exe", PE_ARM64)
    assert cli_main._parse_pe_machine(exe_arm) == PE_ARM64


def test_parse_pe_machine_rejects_non_pe_file(tmp_path):
    """An HTML error page saved as .exe — the classic corrupted-download body."""
    fake = tmp_path / "Hermes.exe"
    fake.write_bytes(b"<html><body>404 Not Found</body></html>" + b" " * 600)
    with pytest.raises(ValueError, match="MZ"):
        cli_main._parse_pe_machine(fake)


def test_parse_pe_machine_rejects_tiny_file(tmp_path):
    stub = tmp_path / "Hermes.exe"
    stub.write_bytes(b"MZ")
    with pytest.raises(ValueError, match="too small"):
        cli_main._parse_pe_machine(stub)


def test_parse_pe_machine_rejects_truncated_sections(tmp_path):
    """File cut mid-download: headers parse but section data extends past EOF."""
    exe = make_pe(tmp_path / "Hermes.exe", PE_AMD64, truncate_to=0x300)
    with pytest.raises(ValueError, match="truncated"):
        cli_main._parse_pe_machine(exe)


def test_parse_pe_machine_rejects_bad_pe_signature(tmp_path):
    exe = make_pe(tmp_path / "Hermes.exe", PE_AMD64)
    data = bytearray(exe.read_bytes())
    data[0x80:0x84] = b"XX\x00\x00"
    exe.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="PE signature"):
        cli_main._parse_pe_machine(exe)


# ─── _expected_windows_pe_machines ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("host", "loadable", "not_loadable"),
    [
        ("AMD64", {PE_AMD64, PE_I386}, {PE_ARM64}),
        ("ARM64", {PE_ARM64, PE_AMD64}, {PE_I386}),
        ("x86", {PE_I386}, {PE_AMD64, PE_ARM64}),
    ],
)
def test_expected_machines_per_host(host, loadable, not_loadable):
    with patch("hermes_cli.main._windows_native_machine", return_value=host):
        expected = cli_main._expected_windows_pe_machines()
    assert loadable <= expected
    assert not (not_loadable & expected)


def test_expected_machines_unknown_host_is_permissive():
    """The gate must never brick launch on hosts we don't recognize."""
    with patch("hermes_cli.main._windows_native_machine", return_value="RISCV64"):
        expected = cli_main._expected_windows_pe_machines()
    assert {PE_AMD64, PE_ARM64, PE_I386} <= expected


# ─── _windows_native_machine ────────────────────────────────────────────────

# MACHINE_ATTRIBUTES.UserEnabled — mirrors hermes_cli.main.
_USER_ENABLED = 0x00000001


class _SettableFunc:
    """Callable that accepts ctypes ``.restype`` / ``.argtypes`` assignment."""

    def __init__(self, fn):
        self._fn = fn
        self.restype = None
        self.argtypes = None

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeKernel32:
    """Stands in for kernel32 so the native-arch probes run on any CI OS."""

    def __init__(
        self,
        native_pe_machine: int,
        *,
        wow64_ok: bool = True,
        user_runnable: set | None = None,
    ):
        self._native_pe = native_pe_machine
        self._wow64_ok = wow64_ok
        self._user_runnable = user_runnable
        self.GetCurrentProcess = _SettableFunc(lambda: -1)
        self.IsWow64Process2 = _SettableFunc(self._is_wow64_process2)
        self.GetMachineTypeAttributes = _SettableFunc(
            self._get_machine_type_attributes
        )

    def _is_wow64_process2(self, _handle, process_ref, native_ref):
        if not self._wow64_ok:
            return 0
        process_ref._obj.value = PE_AMD64  # emulated x64 process
        native_ref._obj.value = self._native_pe
        return 1

    def _get_machine_type_attributes(self, machine, attributes_ref):
        if self._user_runnable is None:
            raise AttributeError("GetMachineTypeAttributes unavailable in this fake")
        if machine not in self._user_runnable:
            attributes_ref._obj.value = 0
            return 0
        attributes_ref._obj.value = _USER_ENABLED
        return 0


def _fake_windll(
    native_pe_machine: int,
    *,
    wow64_ok: bool = True,
    user_runnable: set | None = None,
):
    kernel32 = _FakeKernel32(
        native_pe_machine,
        wow64_ok=wow64_ok,
        user_runnable=user_runnable,
    )

    def _windll(name, *args, **kwargs):
        assert "kernel32" in str(name).lower()
        return kernel32

    return _windll


def test_native_machine_reports_os_arch_not_process_arch(monkeypatch):
    """The #69179 WoA regression: x64 Python under ARM64 emulation must report
    ARM64 (the OS), not AMD64 (the process) — otherwise the integrity gate
    rejects the correct ARM64 rebuild."""
    import ctypes

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    # WinDLL only exists on Windows; create=True so Linux/macOS CI can stub it.
    with patch.object(ctypes, "WinDLL", _fake_windll(PE_ARM64), create=True), \
         patch("platform.machine", return_value="AMD64"):
        assert cli_main._windows_native_machine() == "ARM64"


def test_native_machine_binds_current_process_handle_restype(monkeypatch):
    """ctypes must type GetCurrentProcess as HANDLE — default c_int truncates
    the pseudo-handle on Win64 and makes IsWow64Process2 return
    ERROR_INVALID_HANDLE, re-breaking the #71218 gate on WoA."""
    import ctypes
    from ctypes import wintypes

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    windll = _fake_windll(PE_ARM64)
    with patch.object(ctypes, "WinDLL", windll, create=True), \
         patch("platform.machine", return_value="AMD64"):
        assert cli_main._windows_native_machine() == "ARM64"
    kernel32 = windll("kernel32")
    assert kernel32.GetCurrentProcess.restype is wintypes.HANDLE
    assert kernel32.IsWow64Process2.argtypes is not None


def test_expected_machines_prefers_user_runnable_api_over_arch_name(monkeypatch):
    """GetMachineTypeAttributes answers "can this host load PE machine X?"
    directly, so a WoA host that reports AMD64 everywhere else still accepts an
    ARM64 exe."""
    import ctypes

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
    monkeypatch.delenv("PROCESSOR_ARCHITEW6432", raising=False)
    with patch.object(
        ctypes,
        "WinDLL",
        _fake_windll(
            PE_ARM64, wow64_ok=False, user_runnable={PE_ARM64, PE_AMD64}
        ),
        create=True,
    ), patch("platform.machine", return_value="AMD64"):
        assert cli_main._expected_windows_pe_machines() == {PE_ARM64, PE_AMD64}


def test_expected_machines_falls_back_when_attributes_api_missing(monkeypatch):
    """Pre-Windows-11 hosts have no GetMachineTypeAttributes — the name-based
    mapping must still drive the gate."""
    import ctypes

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    with patch.object(
        ctypes, "WinDLL", _fake_windll(PE_ARM64, user_runnable=None), create=True
    ), patch("platform.machine", return_value="AMD64"):
        assert cli_main._expected_windows_pe_machines() == {PE_ARM64, PE_AMD64}


def test_native_machine_env_fallback_without_api(monkeypatch):
    """Pre-1511 Windows 10: no IsWow64Process2 → env."""
    import ctypes

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    monkeypatch.setenv("PROCESSOR_ARCHITEW6432", "AMD64")

    def _no_kernel32(name, *args, **kwargs):
        raise OSError(f"no {name} in this fake pre-1511 host")

    with patch.object(ctypes, "WinDLL", _no_kernel32, create=True), \
         patch("platform.machine", return_value="x86"):
        assert cli_main._windows_native_machine() == "AMD64"


def test_native_machine_platform_fallback(monkeypatch):
    """No API, no env vars → the historical platform.machine() answer."""
    import ctypes

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    monkeypatch.delenv("PROCESSOR_ARCHITEW6432", raising=False)
    monkeypatch.delenv("PROCESSOR_ARCHITECTURE", raising=False)

    def _no_kernel32(name, *args, **kwargs):
        raise OSError(f"no {name} in this fake host")

    with patch.object(ctypes, "WinDLL", _no_kernel32, create=True), \
         patch("platform.machine", return_value="AMD64"):
        assert cli_main._windows_native_machine() == "AMD64"


def test_native_machine_non_windows_uses_platform(monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    with patch("platform.machine", return_value="aarch64"):
        assert cli_main._windows_native_machine() == "AARCH64"


def test_integrity_gate_accepts_arm64_exe_from_emulated_x64_process(monkeypatch, tmp_path):
    """End-to-end shape of the reporter's failure: ARM64 host, x64 updater
    process, correctly-built ARM64 Hermes.exe. The gate must pass it."""
    import ctypes

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    exe = make_pe(tmp_path / "Hermes.exe", PE_ARM64)
    with patch.object(ctypes, "WinDLL", _fake_windll(PE_ARM64), create=True), \
         patch("platform.machine", return_value="AMD64"):
        assert cli_main._desktop_exe_integrity_error(exe) is None


def test_integrity_gate_accepts_arm64_when_iswow64_fails_but_attributes_ok(
    monkeypatch, tmp_path
):
    """End-to-end residual WoA shape: IsWow64Process2 fails and env lies as
    AMD64, but GetMachineTypeAttributes reports ARM64 as user-runnable, so the
    ARM64 Hermes.exe must pass the gate."""
    import ctypes

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
    monkeypatch.delenv("PROCESSOR_ARCHITEW6432", raising=False)
    exe = make_pe(tmp_path / "Hermes.exe", PE_ARM64)
    with patch.object(
        ctypes,
        "WinDLL",
        _fake_windll(
            PE_ARM64, wow64_ok=False, user_runnable={PE_ARM64, PE_AMD64}
        ),
        create=True,
    ), patch("platform.machine", return_value="AMD64"):
        assert cli_main._desktop_exe_integrity_error(exe) is None


# ─── _desktop_exe_integrity_error ───────────────────────────────────────────


def test_integrity_error_none_for_matching_arch(tmp_path):
    exe = make_pe(tmp_path / "Hermes.exe", PE_AMD64)
    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"):
        assert cli_main._desktop_exe_integrity_error(exe) is None


def test_integrity_error_reports_arch_mismatch(tmp_path):
    """ARM64 exe on the reporter's 'Windows 10 AMD64' host — the wrong-arch
    flavor of 'This app can't run on your computer'."""
    exe = make_pe(tmp_path / "Hermes.exe", PE_ARM64)
    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"):
        error = cli_main._desktop_exe_integrity_error(exe)
    assert error is not None and "architecture mismatch" in error
    assert "ARM64" in error


def test_integrity_error_reports_corruption_reason(tmp_path):
    exe = make_pe(tmp_path / "Hermes.exe", PE_AMD64, truncate_to=0x300)
    error = cli_main._desktop_exe_integrity_error(exe)
    assert error is not None and "truncated" in error


# ─── _desktop_packaged_executable arch preference (win32) ───────────────────


def test_packaged_executable_prefers_host_arch_over_mtime(tmp_path, monkeypatch):
    """A newer wrong-arch tree must not shadow the loadable one (#69179)."""
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    desktop_dir = tmp_path / "apps" / "desktop"
    good = make_pe(desktop_dir / "release" / "win-unpacked" / "Hermes.exe", PE_AMD64)
    bad = make_pe(desktop_dir / "release" / "win-arm64-unpacked" / "Hermes.exe", PE_ARM64)
    # Make the wrong-arch tree the newest, which the pure-mtime pick would take.
    import os

    os.utime(bad, (bad.stat().st_atime + 1000, bad.stat().st_mtime + 1000))

    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"):
        assert cli_main._desktop_packaged_executable(desktop_dir) == good


def test_packaged_executable_falls_back_to_mtime_when_unparseable(tmp_path, monkeypatch):
    """Non-PE stubs (dev trees, tests) keep the historical newest-wins pick."""
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    desktop_dir = tmp_path / "apps" / "desktop"
    a = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    b = desktop_dir / "release" / "win-arm64-unpacked" / "Hermes.exe"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("", encoding="utf-8")
    import os

    os.utime(b, (b.stat().st_atime + 1000, b.stat().st_mtime + 1000))
    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"):
        assert cli_main._desktop_packaged_executable(desktop_dir) == b


# ─── rollback ───────────────────────────────────────────────────────────────


def _win_tree(tmp_path: Path) -> tuple[Path, Path]:
    desktop_dir = tmp_path / "apps" / "desktop"
    exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    return desktop_dir, exe


def test_rollback_restores_backup_and_keeps_corrupt_copy(tmp_path):
    desktop_dir, exe = _win_tree(tmp_path)
    make_pe(exe, PE_AMD64, truncate_to=0x300)  # corrupt new build
    backup_exe = desktop_dir / "release" / "win-unpacked.bak" / "Hermes.exe"
    make_pe(backup_exe, PE_AMD64)  # valid old build

    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"):
        restored = cli_main._rollback_desktop_from_backup(exe)

    assert restored == exe
    # The restored exe is the old, valid build.
    assert cli_main._parse_pe_machine(exe) == PE_AMD64
    assert exe.stat().st_size == 0x400
    # Corrupt tree preserved for diagnostics; backup consumed.
    assert (desktop_dir / "release" / "win-unpacked.corrupt" / "Hermes.exe").exists()
    assert not backup_exe.exists()


def test_rollback_returns_none_without_backup(tmp_path):
    _, exe = _win_tree(tmp_path)
    make_pe(exe, PE_AMD64, truncate_to=0x300)
    assert cli_main._rollback_desktop_from_backup(exe) is None
    # The corrupt tree is left in place (nothing to restore over it).
    assert exe.exists()


def test_rollback_refuses_corrupt_backup(tmp_path):
    """Never 'restore' a backup that would also fail to launch."""
    desktop_dir, exe = _win_tree(tmp_path)
    make_pe(exe, PE_AMD64, truncate_to=0x300)
    backup_exe = desktop_dir / "release" / "win-unpacked.bak" / "Hermes.exe"
    make_pe(backup_exe, PE_AMD64, truncate_to=0x280)  # backup also corrupt

    assert cli_main._rollback_desktop_from_backup(exe) is None
    assert backup_exe.exists()  # untouched


# ─── _ensure_desktop_exe_launchable (the gate) ──────────────────────────────


def test_gate_passes_valid_exe(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    desktop_dir, exe = _win_tree(tmp_path)
    make_pe(exe, PE_AMD64)
    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"):
        verified, rolled_back = cli_main._ensure_desktop_exe_launchable(desktop_dir, exe)
    assert verified == exe
    assert rolled_back is False


def test_gate_noop_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    desktop_dir, exe = _win_tree(tmp_path)
    exe.parent.mkdir(parents=True)
    exe.write_text("not a pe at all", encoding="utf-8")
    verified, rolled_back = cli_main._ensure_desktop_exe_launchable(desktop_dir, exe)
    assert verified == exe
    assert rolled_back is False


def test_gate_rolls_back_corrupt_exe_and_purges_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    desktop_dir, exe = _win_tree(tmp_path)
    make_pe(exe, PE_AMD64, truncate_to=0x300)
    make_pe(desktop_dir / "release" / "win-unpacked.bak" / "Hermes.exe", PE_AMD64)

    stamp = tmp_path / "home" / "desktop-build-stamp.json"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("{}", encoding="utf-8")

    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[]) as mock_purge, \
         patch("hermes_cli.main._desktop_stamp_path", return_value=stamp):
        verified, rolled_back = cli_main._ensure_desktop_exe_launchable(desktop_dir, exe)

    assert rolled_back is True
    assert verified == exe
    assert cli_main._parse_pe_machine(exe) == PE_AMD64  # restored old build
    mock_purge.assert_called_once()
    assert not stamp.exists()  # stamp invalidated so retry genuinely rebuilds
    out = capsys.readouterr().out
    assert "integrity check" in out
    assert "Update aborted" in out
    assert "existing version was kept" in out


def test_gate_fails_clearly_without_backup(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    desktop_dir, exe = _win_tree(tmp_path)
    fake = exe
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"<html>proxy error</html>" + b" " * 600)

    with patch("hermes_cli.main._purge_electron_build_cache", return_value=[]), \
         patch("hermes_cli.main._desktop_stamp_path", return_value=tmp_path / "stamp.json"):
        verified, rolled_back = cli_main._ensure_desktop_exe_launchable(desktop_dir, exe)

    assert verified is None
    assert rolled_back is False
    out = capsys.readouterr().out
    assert "integrity check" in out
    assert "No usable backup" in out


# ─── end-to-end: `hermes desktop --build-only` exits nonzero on corrupt exe ─


def _ns(**kw):
    defaults = dict(
        skip_build=False,
        build_only=True,
        force_build=False,
        source=False,
        fake_boot=False,
        ignore_existing=False,
        hermes_root=None,
        cwd=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_build_only_fails_when_pack_produces_corrupt_exe(tmp_path, monkeypatch, capsys):
    """The updater chain's contract: a rebuild whose Hermes.exe cannot launch
    must exit nonzero (so hermes-setup's retry-once kicks in) and must restore
    the previous working build instead of leaving the corrupt one."""
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "win32")

    exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    make_pe(exe, PE_AMD64, truncate_to=0x300)  # what the failed pack produced
    make_pe(desktop_dir / "release" / "win-unpacked.bak" / "Hermes.exe", PE_AMD64)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._stop_desktop_processes_locking_build", return_value=[]), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[]), \
         patch("hermes_cli.main._desktop_stamp_path", return_value=tmp_path / "stamp.json"), \
         patch("hermes_cli.main._write_desktop_build_stamp") as mock_stamp, \
         patch("hermes_cli.main._windows_native_machine", return_value="AMD64"), \
         patch("hermes_cli.main.subprocess.run", return_value=pack_ok), \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    # The previous working exe was restored...
    assert cli_main._parse_pe_machine(exe) == PE_AMD64
    # ...and the poisoned build was never stamped as good.
    mock_stamp.assert_not_called()
    out = capsys.readouterr().out
    assert "integrity check" in out


def test_build_only_succeeds_with_valid_exe(tmp_path, monkeypatch, capsys):
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "win32")

    make_pe(desktop_dir / "release" / "win-unpacked" / "Hermes.exe", PE_AMD64)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._stop_desktop_processes_locking_build", return_value=[]), \
         patch("hermes_cli.main._write_desktop_build_stamp") as mock_stamp, \
         patch("hermes_cli.main._windows_native_machine", return_value="AMD64"), \
         patch("hermes_cli.main.subprocess.run", return_value=pack_ok):
        cli_main.cmd_gui(_ns())

    mock_stamp.assert_called_once()
    assert "Desktop packaged app ready" in capsys.readouterr().out
