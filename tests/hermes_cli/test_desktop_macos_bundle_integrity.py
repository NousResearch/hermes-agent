"""Behavior tests for the macOS desktop-bundle integrity gate.

Companion to ``test_desktop_exe_integrity.py`` (the Windows PE gate, #69179).
macOS had no equivalent, and the gap shipped a broken app three times:

``npm run pack`` pins no target arch, so electron-builder defaults to x64 on
macOS and names its output dir ``release/mac``. ``run-electron-builder.mjs``
forces ``-c.electronDist=<local dist>`` — the **arm64** Electron — while
``before-pack.mjs`` stages node-pty for the *declared* (x64) target. The result
is one internally-inconsistent bundle: an arm64 Electron binary carrying only
``prebuilds/darwin-x64``. It dies at launch with

    Failed to load native module: pty.node
    Cannot find module './prebuilds/darwin-arm64//pty.node'

Two properties make this invisible to every check that existed:

1. **The main executable is arm64 in both the good and the bad bundle**, so an
   arch probe of ``Contents/MacOS/Hermes`` (the natural mirror of the Windows PE
   check) passes the broken bundle. Only the node-pty payload differs.
2. **Electron resolves through the asar header, not the filesystem.** A bundle
   whose header lists ``prebuilds/darwin-x64`` alone cannot load a
   ``darwin-arm64`` binary dropped into ``app.asar.unpacked`` — ``require``
   never looks. So the gate must read the header, not just stat files.

These tests exercise the behavior contract only: synthetic Mach-O files and
hand-rolled asar headers go in, verdicts / selections / rollbacks come out.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_desktop

CPU_ARM64 = 0x0100000C
CPU_X86_64 = 0x01000007

# node-pty's staged payload, relative to the app bundle's Resources dir.
_PREBUILDS = "dist/node_modules/node-pty/prebuilds"


def make_macho(path: Path, cputype: int = CPU_ARM64, *, truncate: bool = False) -> Path:
    """Write a minimal Mach-O 64-bit little-endian header (MH_CIGAM_64).

    Only the first 8 bytes carry meaning for the gate: the magic identifies the
    container and byte order, the cputype identifies the architecture. Mirrors
    ``makeFakeNode`` in ``apps/desktop/scripts/stage-native-deps.test.mjs``.
    """
    data = b"\xcf\xfa\xed\xfe" + struct.pack("<I", cputype)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data[:3] if truncate else data)
    return path


def make_fat_macho(path: Path, cputypes: tuple[int, ...] = (CPU_X86_64, CPU_ARM64)) -> Path:
    """Write a universal (fat) Mach-O header advertising *cputypes*."""
    buf = bytearray(struct.pack(">II", 0xCAFEBABE, len(cputypes)))
    for cputype in cputypes:
        # fat_arch: cputype, cpusubtype, offset, size, align — all big-endian.
        buf += struct.pack(">5I", cputype, 0, 0x1000, 0x10, 12)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return path


def _asar(archive: Path, files: dict) -> None:
    """Write an asar whose header is *files*, with no packed payload.

    The Pickle framing matches ``_verify_packaged_entry``'s validation exactly
    (see the ``bundle`` fixture in ``test_desktop_update_verify.py``).
    """
    header = json.dumps({"files": files}).encode()
    padded = header + b"\0" * (-len(header) % 4)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(
        struct.pack("<4I", 4, 8 + len(padded), 4 + len(padded), len(header)) + padded
    )


def _nest(path: str, leaf: dict) -> dict:
    """Wrap *leaf* in the nested ``{"files": {...}}`` chain naming *path*."""
    node = leaf
    for part in reversed(path.split("/")):
        node = {part: {"files": node}}
    return node


def make_app_bundle(
    release_dir: Path,
    target: str,
    *,
    prebuild_arch: str = "arm64",
    exe_cputype: int = CPU_ARM64,
    unpacked: bool = True,
    omit: str = "",
    spawn_helper_mode: int = 0o755,
) -> Path:
    """Lay down ``<release_dir>/<target>/Hermes.app`` and return its main executable.

    *prebuild_arch* names the ONE node-pty prebuild this bundle carries — the
    single knob that separates a working arm64 pack from the broken generic
    ``release/mac`` pack. *omit* drops one file name from the payload.
    """
    app = release_dir / target / "Hermes.app"
    exe = make_macho(app / "Contents" / "MacOS" / "Hermes", exe_cputype)

    resources = app / "Contents" / "Resources"
    prebuild_dir = f"{_PREBUILDS}/darwin-{prebuild_arch}"
    cputype = CPU_ARM64 if prebuild_arch == "arm64" else CPU_X86_64

    entries: dict = {}
    for name in ("pty.node", "spawn-helper"):
        if name == omit:
            continue
        entries[name] = {"size": 8, "unpacked": True} if unpacked else {"size": 8, "offset": "0"}
        if unpacked:
            binary = make_macho(resources / "app.asar.unpacked" / prebuild_dir / name, cputype)
            if name == "spawn-helper":
                os.chmod(binary, spawn_helper_mode)

    _asar(resources / "app.asar", _nest(prebuild_dir, entries))
    return exe


# ─── the gate: _desktop_exe_integrity_error on darwin ───────────────────────


@pytest.mark.macos_only
def test_generic_mac_bundle_with_only_x64_prebuilds_is_rejected(tmp_path):
    """The exact artifact `npm run pack` produces unpinned: arm64 Electron, x64 node-pty.

    This is the regression. The bundle looks healthy by every pre-existing
    measure — the file exists, the main executable is native arm64 — and it
    cannot start. The verdict must name the payload, not the executable.
    """
    exe = make_app_bundle(tmp_path / "release", "mac", prebuild_arch="x64")

    error = main_desktop._desktop_exe_integrity_error(exe)

    assert error is not None, "a bundle that cannot load pty.node must not pass the gate"
    assert "darwin-arm64" in error
    assert "darwin-x64" in error, "the verdict must report what the bundle actually carries"


@pytest.mark.macos_only
def test_arm64_bundle_passes_the_gate(tmp_path):
    exe = make_app_bundle(tmp_path / "release", "mac-arm64", prebuild_arch="arm64")

    assert main_desktop._desktop_exe_integrity_error(exe) is None


@pytest.mark.macos_only
def test_universal_prebuild_passes_the_gate(tmp_path):
    """A lipo-merged fat binary carrying an arm64 slice is loadable."""
    exe = make_app_bundle(tmp_path / "release", "mac-arm64")
    resources = exe.parents[1] / "Resources"
    make_fat_macho(resources / "app.asar.unpacked" / _PREBUILDS / "darwin-arm64" / "pty.node")

    assert main_desktop._desktop_exe_integrity_error(exe) is None


@pytest.mark.macos_only
def test_missing_spawn_helper_is_rejected(tmp_path):
    """pty.node alone is not enough — node-pty execs spawn-helper to fork the pty."""
    exe = make_app_bundle(tmp_path / "release", "mac-arm64", omit="spawn-helper")

    error = main_desktop._desktop_exe_integrity_error(exe)

    assert error is not None
    assert "spawn-helper" in error


@pytest.mark.macos_only
def test_non_executable_spawn_helper_is_rejected(tmp_path):
    exe = make_app_bundle(tmp_path / "release", "mac-arm64", spawn_helper_mode=0o644)

    error = main_desktop._desktop_exe_integrity_error(exe)

    assert error is not None
    assert "executable" in error


@pytest.mark.macos_only
def test_prebuilds_packed_inside_the_asar_are_rejected(tmp_path):
    """Native code cannot be dlopen'd from inside an asar; it must be unpacked."""
    exe = make_app_bundle(tmp_path / "release", "mac-arm64", unpacked=False)

    error = main_desktop._desktop_exe_integrity_error(exe)

    assert error is not None
    assert "unpacked" in error


@pytest.mark.macos_only
def test_arm64_files_on_disk_do_not_rescue_an_x64_asar_header(tmp_path):
    """The 2026-08-23 retraction, pinned as a test.

    Copying darwin-arm64 binaries into ``app.asar.unpacked`` looks like a fix
    and passes a direct-path require, but Electron resolves through the asar
    header — which still lists darwin-x64 alone. The gate must read the header,
    so a bundle patched this way stays rejected.
    """
    exe = make_app_bundle(tmp_path / "release", "mac", prebuild_arch="x64")
    resources = exe.parents[1] / "Resources"
    for name in ("pty.node", "spawn-helper"):
        make_macho(resources / "app.asar.unpacked" / _PREBUILDS / "darwin-arm64" / name, CPU_ARM64)

    assert main_desktop._desktop_exe_integrity_error(exe) is not None


@pytest.mark.macos_only
def test_wrong_arch_main_executable_is_rejected(tmp_path):
    exe = make_app_bundle(
        tmp_path / "release", "mac", prebuild_arch="x64", exe_cputype=CPU_X86_64
    )

    error = main_desktop._desktop_exe_integrity_error(exe)

    assert error is not None
    assert "x86_64" in error


@pytest.mark.macos_only
def test_unreadable_asar_is_rejected(tmp_path):
    exe = make_app_bundle(tmp_path / "release", "mac-arm64")
    (exe.parents[1] / "Resources" / "app.asar").write_bytes(b"not an asar")

    assert main_desktop._desktop_exe_integrity_error(exe) is not None


# ─── selection: _desktop_packaged_executable_in ─────────────────────────────


@pytest.mark.macos_only
def test_selection_prefers_arm64_bundle_over_a_newer_generic_mac_bundle(tmp_path):
    """The trace this whole gate exists for.

    ``release/mac`` and ``release/mac-arm64`` coexist because the promote step
    names the live dir after electron-builder's staged output dir, so an
    unpinned (x64-declared) pack lands beside the last good arm64 one. Selection
    globbed ``mac*`` and broke the tie by mtime, so the freshly built broken
    bundle won every time. Windows has had an arch preference since #69179;
    macOS never got one.
    """
    release = tmp_path / "release"
    good = make_app_bundle(release, "mac-arm64", prebuild_arch="arm64")
    broken = make_app_bundle(release, "mac", prebuild_arch="x64")
    os.utime(broken, (10**9, 10**9))  # the broken bundle is the NEWER one
    os.utime(good, (10**8, 10**8))

    assert main_desktop._desktop_packaged_executable_in(release) == good


@pytest.mark.macos_only
def test_selection_falls_back_to_an_unloadable_bundle_when_it_is_the_only_one(tmp_path):
    """No silent "no app installed": a lone broken bundle is still returned so
    the caller's gate can report *why* it cannot run, not merely that nothing
    was found."""
    release = tmp_path / "release"
    broken = make_app_bundle(release, "mac", prebuild_arch="x64")

    assert main_desktop._desktop_packaged_executable_in(release) == broken


# ─── swap refusal: the previous app is retained ─────────────────────────────


@pytest.mark.macos_only
def test_promote_refuses_an_unloadable_staged_bundle_and_keeps_the_live_app(
    tmp_path, capsys
):
    """A staged pack that cannot load pty.node must never reach ``release/``."""
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    live = make_app_bundle(desktop_dir / "release", "mac-arm64", prebuild_arch="arm64")
    live_marker = live.read_bytes()

    staging = desktop_dir / ".staging-1-1"
    make_app_bundle(staging, "mac", prebuild_arch="x64")

    with pytest.raises(SystemExit) as exit_info:
        main_desktop._promote_staged_desktop_app(desktop_dir, staging)

    assert exit_info.value.code == 1
    assert live.exists() and live.read_bytes() == live_marker, "live app must be untouched"
    assert not staging.exists(), "the rejected staging tree must be discarded"
    assert not (desktop_dir / "release" / "mac").exists(), "the bad bundle must not be promoted"
    out = capsys.readouterr().out
    assert "darwin-arm64" in out
    assert "previous desktop app" in out


# ─── launch refusal ─────────────────────────────────────────────────────────


def _gui_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        source=False, skip_build=True, force_build=False, build_only=False,
        local=False, setup_tcc_identity=False, identity=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.macos_only
def test_gui_refuses_to_launch_an_unloadable_bundle(tmp_path, monkeypatch, capsys):
    """``hermes desktop --skip-build`` must not exec a bundle that will crash."""
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    make_app_bundle(desktop_dir / "release", "mac", prebuild_arch="x64")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    with patch("hermes_cli.main.subprocess.run") as run:
        with pytest.raises(SystemExit) as exit_info:
            main_desktop.cmd_gui(_gui_args())

    assert exit_info.value.code == 1
    run.assert_not_called()
    assert "darwin-arm64" in capsys.readouterr().out


@pytest.mark.macos_only
def test_gui_launches_the_arm64_bundle(tmp_path, monkeypatch):
    """The positive half: a loadable arm64 bundle is selected and launched."""
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    exe = make_app_bundle(desktop_dir / "release", "mac-arm64", prebuild_arch="arm64")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    launched = subprocess.CompletedProcess([str(exe)], 0)
    with patch("hermes_cli.main.subprocess.run", return_value=launched) as run:
        with pytest.raises(SystemExit) as exit_info:
            main_desktop.cmd_gui(_gui_args())

    assert exit_info.value.code == 0
    run.assert_called_once()
    assert run.call_args[0][0][0] == str(exe)


# ─── other platforms are unaffected ─────────────────────────────────────────


@pytest.mark.linux_only
def test_linux_bundles_are_not_subject_to_the_macos_gate(tmp_path):
    exe = tmp_path / "release" / "linux-unpacked" / "hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")

    assert main_desktop._desktop_exe_integrity_error(exe) is None
