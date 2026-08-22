"""Behavior tests for the macOS desktop arch guard (#75612).

The desktop self-update chain (Desktop → ``hermes update`` →
``hermes desktop --build-only`` → relaunch) can rebuild the Electron app for
the wrong architecture when the build runs under an x86_64 Python on an Apple
Silicon host (``stage-native-deps.mjs`` defaults ``arch = process.arch``).
That leaves a stale x86_64 ``release/mac/`` tree beside the working
``release/mac-arm64/`` one. ``_desktop_packaged_executable`` used to pick the
candidate purely by mtime, so the freshly-built wrong-arch tree always won and
the launcher hung with no window.

These tests exercise the behavior contract only: synthetic Mach-O files go in,
selection verdicts come out. The host-arch probes (``sysctl`` subprocess,
``platform.machine``) are mocked so the suite runs on any CI host.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main

CPU_X86_64 = 0x01000007
CPU_ARM64 = 0x0100000C

MH_CIGAM_64 = 0xCFFAEDFE
FAT_MAGIC = 0xCAFEBABE


def make_thin_macho(path: Path, cputype: int) -> Path:
    """Write a minimal little-endian 64-bit thin Mach-O header.

    Layout: magic (4) + cputype (4) + padding. Every real macOS desktop
    binary is little-endian on disk, so the header fields are written "<I".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<II", MH_CIGAM_64, cputype) + b"\x00" * 56)
    return path


def make_fat_macho(path: Path, *cputypes: int) -> Path:
    """Write a minimal fat/Universal header listing ``cputypes`` slices."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(">II", FAT_MAGIC, len(cputypes))
    records = b"".join(struct.pack(">IIIII", ct, 0, 0, 0, 0) for ct in cputypes)
    path.write_bytes(header + records)
    return path


def _fake_sysctl(values: dict[str, str]):
    """Return a ``subprocess.check_output`` fake keyed by sysctl name."""

    def _check_output(cmd, *args, **kwargs):
        # cmd is ["/usr/sbin/sysctl", "-n", <name>]
        name = cmd[-1]
        if name in values:
            return (values[name] + "\n").encode()
        raise subprocess.CalledProcessError(1, cmd)

    return _check_output


# ─── _darwin_native_machine ─────────────────────────────────────────────────


def test_native_machine_translated_process_means_arm64_host():
    """The #75612 regression: x86_64 Python under Rosetta reports
    ``platform.machine() == "x86_64"`` on an arm64 host. The
    ``sysctl.proc_translated == 1`` probe must override that lie.
    """
    with patch("subprocess.check_output", _fake_sysctl({"sysctl.proc_translated": "1"})), \
         patch("platform.machine", return_value="x86_64"):
        assert cli_main._darwin_native_machine() == "arm64"


def test_native_machine_native_arm64_process():
    with patch("subprocess.check_output", _fake_sysctl({"sysctl.proc_translated": "0"})), \
         patch("platform.machine", return_value="arm64"):
        assert cli_main._darwin_native_machine() == "arm64"


def test_native_machine_native_x86_64_process():
    with patch("subprocess.check_output", _fake_sysctl({"sysctl.proc_translated": "0"})), \
         patch("platform.machine", return_value="x86_64"):
        assert cli_main._darwin_native_machine() == "x86_64"


def test_native_machine_falls_back_to_hw_machine_on_exotic_arch():
    """Unknown ``platform.machine()`` values fall through to ``hw.machine``."""
    with patch(
        "subprocess.check_output",
        _fake_sysctl({"sysctl.proc_translated": "0", "hw.machine": "arm64"}),
    ), patch("platform.machine", return_value="riscv64"):
        assert cli_main._darwin_native_machine() == "arm64"


# ─── _darwin_host_cpu_types / _darwin_native_cpu_types ──────────────────────


def test_host_cpu_types_arm64_host_runs_both():
    with patch.object(cli_main, "_darwin_native_machine", return_value="arm64"):
        assert cli_main._darwin_host_cpu_types() == frozenset({CPU_ARM64, CPU_X86_64})


def test_host_cpu_types_x86_64_host_runs_only_x64():
    with patch.object(cli_main, "_darwin_native_machine", return_value="x86_64"):
        assert cli_main._darwin_host_cpu_types() == frozenset({CPU_X86_64})


def test_native_cpu_types_is_single_arch():
    with patch.object(cli_main, "_darwin_native_machine", return_value="arm64"):
        assert cli_main._darwin_native_cpu_types() == frozenset({CPU_ARM64})
    with patch.object(cli_main, "_darwin_native_machine", return_value="x86_64"):
        assert cli_main._darwin_native_cpu_types() == frozenset({CPU_X86_64})


# ─── _macho_cpu_types ───────────────────────────────────────────────────────


def test_macho_cpu_types_thin_arm64(tmp_path):
    exe = make_thin_macho(tmp_path / "Hermes", CPU_ARM64)
    assert cli_main._macho_cpu_types(exe) == frozenset({CPU_ARM64})


def test_macho_cpu_types_thin_x86_64(tmp_path):
    exe = make_thin_macho(tmp_path / "Hermes", CPU_X86_64)
    assert cli_main._macho_cpu_types(exe) == frozenset({CPU_X86_64})


def test_macho_cpu_types_fat_universal(tmp_path):
    exe = make_fat_macho(tmp_path / "Hermes", CPU_ARM64, CPU_X86_64)
    assert cli_main._macho_cpu_types(exe) == frozenset({CPU_ARM64, CPU_X86_64})


def test_macho_cpu_types_unreadable_returns_empty(tmp_path):
    assert cli_main._macho_cpu_types(tmp_path / "missing") == frozenset()


def test_macho_cpu_types_non_macho_returns_empty(tmp_path):
    junk = tmp_path / "junk"
    junk.write_bytes(b"not a mach-o binary at all")
    assert cli_main._macho_cpu_types(junk) == frozenset()


# ─── _desktop_packaged_executable (darwin selection) ────────────────────────


def _make_app(release: Path, tree: str, cputype: int, *, mtime: float) -> Path:
    exe = release / tree / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    make_thin_macho(exe, cputype)
    import os

    os.utime(exe, (mtime, mtime))
    return exe


@pytest.mark.macos_only
def test_selection_prefers_native_over_newer_wrong_arch(tmp_path, monkeypatch):
    """The core #75612 scenario: a stale x86_64 ``mac/`` tree with a NEWER
    mtime than the working ``mac-arm64/`` tree must not win the selection.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    release = tmp_path / "release"
    arm_exe = _make_app(release, "mac-arm64", CPU_ARM64, mtime=1_000_000)
    _make_app(release, "mac", CPU_X86_64, mtime=9_999_999)  # newer, wrong arch

    with patch.object(cli_main, "_darwin_native_machine", return_value="arm64"):
        picked = cli_main._desktop_packaged_executable(tmp_path)
    assert picked == arm_exe


@pytest.mark.macos_only
def test_selection_single_candidate_untouched(tmp_path, monkeypatch):
    """A lone candidate is returned regardless of arch (no regression)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    release = tmp_path / "release"
    x64_exe = _make_app(release, "mac", CPU_X86_64, mtime=1_000_000)

    with patch.object(cli_main, "_darwin_native_machine", return_value="arm64"):
        picked = cli_main._desktop_packaged_executable(tmp_path)
    assert picked == x64_exe


@pytest.mark.macos_only
def test_selection_falls_back_to_runnable_when_no_native(tmp_path, monkeypatch):
    """On an x86_64 host with only an x86_64 tree, that tree is picked."""
    monkeypatch.setattr(sys, "platform", "darwin")
    release = tmp_path / "release"
    x64_exe = _make_app(release, "mac", CPU_X86_64, mtime=1_000_000)
    _make_app(release, "mac-arm64", CPU_ARM64, mtime=9_999_999)  # unrunnable here

    with patch.object(cli_main, "_darwin_native_machine", return_value="x86_64"):
        picked = cli_main._desktop_packaged_executable(tmp_path)
    assert picked == x64_exe
