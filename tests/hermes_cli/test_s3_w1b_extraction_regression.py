"""Wave-1 extraction regression tests (shard s3, implementer w1b).

The s3 god-file decomposition moved two clusters out of
``hermes_cli/main.py`` into new modules:

* ``hermes_cli/web_ui_build.py`` — web-UI / desktop build staleness, content
  hashing and build stamps (cluster c2, 16 methods incl. nested).
* ``hermes_cli/pe_integrity_mixin.py`` — Windows desktop-exe PE-integrity
  gate (cluster c5, 10 methods) plus its module-level PE constants.

These tests pin the extraction contract:

1. **Re-export surface** — every moved name still resolves on
   ``hermes_cli.main`` and is the *same object* as the new module's function
   (existing callers and ``from hermes_cli.main import ...`` imports keep
   working).
2. **Patch surface** — monkeypatches targeting ``hermes_cli.main.<name>``
   (the historical test convention) still reach the moved code, which routes
   shared/moved-but-patched references through the lazy ``_m()`` reference
   (the same convention ``hermes_cli/update_cmd.py`` established).
3. **Behavior** — the moved pure functions behave identically.

The module is deliberately imported before ``hermes_cli.main`` in this file to
prove the new modules import standalone without an import cycle.
"""

import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import pe_integrity_mixin as pe
from hermes_cli import web_ui_build as web

# Imported last on purpose — see module docstring point 3.
from hermes_cli import main as cli_main  # noqa: E402

MOVED_WEB = [
    "_web_ui_build_needed",
    "_compute_web_ui_content_hash",
    "_web_ui_stamp_path",
    "_write_web_ui_build_stamp",
    "_missing_web_build_tool",
    "_build_web_ui",
    "_do_build_web_ui",
    "_desktop_dist_exists",
    "_compute_desktop_content_hash",
    "_desktop_stamp_path",
    "_desktop_build_needed",
    "_write_desktop_build_stamp",
]

MOVED_PE = [
    "_windows_native_machine_from_iswow64",
    "_windows_user_runnable_pe_machines",
    "_windows_native_machine",
    "_expected_windows_pe_machines",
    "_parse_pe_machine",
    "_pe_machine_or_none",
    "_desktop_exe_integrity_error",
    "_desktop_backup_unpacked_dir",
    "_rollback_desktop_from_backup",
    "_ensure_desktop_exe_launchable",
]

PE_AMD64 = 0x8664


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
    struct.pack_into("<HHIIIHH", buf, 0x84, machine, 1, 0, 0, 0, 0, 0x0002)
    section_off = 0x98
    struct.pack_into("<II", buf, section_off + 16, 0x200, 0x200)
    data = bytes(buf if truncate_to is None else buf[:truncate_to])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ─── 1. Re-export surface ───────────────────────────────────────────────────


def test_main_re_exports_every_moved_name_as_same_object():
    for name in MOVED_WEB:
        assert getattr(cli_main, name) is getattr(web, name), name
    for name in MOVED_PE:
        assert getattr(cli_main, name) is getattr(pe, name), name


def test_moved_functions_defined_in_new_modules():
    assert web._do_build_web_ui.__module__ == "hermes_cli.web_ui_build"
    assert pe._parse_pe_machine.__module__ == "hermes_cli.pe_integrity_mixin"


def test_pe_constants_moved_with_cluster():
    assert pe._PE_MACHINE_I386 == 0x014C
    assert pe._PE_MACHINE_AMD64 == 0x8664
    assert pe._PE_MACHINE_ARM64 == 0xAA64
    assert pe._PE_MACHINE_NAMES[0x8664] == "x64 (AMD64)"
    assert pe._PE_MACHINE_TO_NAME[0xAA64] == "ARM64"
    assert pe._MACHINE_ATTRIBUTE_USER_ENABLED == 0x00000001
    # The constants are used only by the moved cluster; they are not re-exported.
    assert not hasattr(cli_main, "_PE_MACHINE_I386")


# ─── 2. Patch surface (hermes_cli.main.<name> must keep working) ────────────


def test_web_routing_patch_hermes_cli_main_desktop_stamp_path(tmp_path):
    """_m()-routed call sees a patch on hermes_cli.main._desktop_stamp_path."""
    stamp = tmp_path / "stamp.json"
    with patch("hermes_cli.main._desktop_stamp_path", return_value=stamp):
        web._write_desktop_build_stamp(tmp_path, source_mode=True)
    assert stamp.exists()
    assert '"sourceMode": true' in stamp.read_text(encoding="utf-8")


def test_web_routing_patch_hermes_cli_main_desktop_packaged_executable(tmp_path):
    """_desktop_build_needed routes _desktop_packaged_executable via _m()."""
    with patch("hermes_cli.main._desktop_packaged_executable", return_value=None):
        assert web._desktop_build_needed(
            tmp_path / "apps" / "desktop", tmp_path, source_mode=False
        ) is True


def test_pe_routing_patch_hermes_cli_main_windows_native_machine(monkeypatch):
    """_expected_windows_pe_machines routes _windows_native_machine via _m()."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"):
        assert pe._expected_windows_pe_machines() == {0x8664, 0x014C}


def test_pe_routing_patch_hermes_cli_main_end_to_end(tmp_path, monkeypatch):
    """_ensure_desktop_exe_launchable sees main-side patches end to end."""
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    desktop_dir = tmp_path / "apps" / "desktop"
    exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    make_pe(exe, PE_AMD64)
    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[]), \
         patch("hermes_cli.main._desktop_stamp_path", return_value=tmp_path / "stamp.json"):
        verified, rolled_back = cli_main._ensure_desktop_exe_launchable(desktop_dir, exe)
    assert verified == exe
    assert rolled_back is False


# ─── 3. Behavior ────────────────────────────────────────────────────────────


def test_missing_web_build_tool_dialects():
    assert web._missing_web_build_tool("sh: 1: tsc: not found") == "tsc"
    assert web._missing_web_build_tool("vite: command not found") == "vite"
    assert web._missing_web_build_tool(
        "'tsc' is not recognized as an internal or external command"
    ) == "tsc"
    assert web._missing_web_build_tool("npm ERR! code ENOENT") is None


def test_parse_pe_machine_roundtrip(tmp_path):
    exe = make_pe(tmp_path / "Hermes.exe", PE_AMD64)
    assert cli_main._parse_pe_machine(exe) == PE_AMD64


def test_pe_machine_or_none_on_truncated(tmp_path):
    exe = make_pe(tmp_path / "broken.exe", PE_AMD64, truncate_to=0x300)
    with pytest.raises(ValueError):
        cli_main._parse_pe_machine(exe)
    assert cli_main._pe_machine_or_none(exe) is None


def test_desktop_exe_integrity_error_machine_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    exe = make_pe(tmp_path / "arm64.exe", 0xAA64)
    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"):
        err = cli_main._desktop_exe_integrity_error(exe)
    assert err is not None
    assert "architecture mismatch" in err


def test_desktop_backup_unpacked_dir():
    exe = Path("C:/apps/desktop/release/win-unpacked/Hermes.exe")
    assert pe._desktop_backup_unpacked_dir(exe) == Path(
        "C:/apps/desktop/release/win-unpacked.bak"
    )


def test_rollback_restores_backup(tmp_path):
    """Rollback picks the .bak tree when the live exe fails the probe."""
    unpacked = tmp_path / "win-unpacked"
    exe = unpacked / "Hermes.exe"
    make_pe(exe, PE_AMD64, truncate_to=0x300)  # corrupt live exe
    backup_exe = tmp_path / "win-unpacked.bak" / "Hermes.exe"
    make_pe(backup_exe, PE_AMD64)  # good backup
    good_bytes = backup_exe.read_bytes()
    with patch("hermes_cli.main._windows_native_machine", return_value="AMD64"):
        restored = cli_main._rollback_desktop_from_backup(exe)
    assert restored == exe
    # The .bak tree was renamed into place; the corrupt tree is kept alongside.
    assert exe.read_bytes() == good_bytes
    assert (tmp_path / "win-unpacked.corrupt" / "Hermes.exe").exists()
    assert not backup_exe.exists()
