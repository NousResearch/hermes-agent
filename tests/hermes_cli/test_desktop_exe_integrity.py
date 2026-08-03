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






# ─── _expected_windows_pe_machines ──────────────────────────────────────────




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








# ─── _desktop_exe_integrity_error ───────────────────────────────────────────




# ─── _desktop_packaged_executable arch preference (win32) ───────────────────






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




# ─── _ensure_desktop_exe_launchable (the gate) ──────────────────────────────






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


