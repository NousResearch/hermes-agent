"""Windows desktop-exe integrity gate — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition, shard s3 cluster c5): the PE-header
integrity helpers (``#69179``) and their module-level constants. Function
bodies are lifted verbatim; the only mechanical change is that references to
helpers that STAY in ``hermes_cli.main`` (``_purge_electron_build_cache``) and
to moved-but-test-patched siblings (``_desktop_stamp_path``,
``_windows_native_machine``) are routed through ``_m()`` — a lazy
``hermes_cli.main`` reference — so existing call sites and test monkeypatches
that target ``hermes_cli.main.<name>`` keep working unchanged. ``main.py``
re-imports every moved name from here (``# noqa: E402``) so the call surface
still resolves on ``hermes_cli.main``.

Imports are one-way: ``hermes_cli.main`` imports this module, never the
reverse at import time (``_m()`` resolves lazily at call time, when main.py is
fully loaded, so there is no import cycle).
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional


def _m():
    """Lazy ``hermes_cli.main`` reference.

    Lets callers keep patching ``hermes_cli.main.<helper>`` (the historical
    test surface) and have those patches reach this code path, and defers the
    import so ``hermes_cli.main`` -> ``hermes_cli.pe_integrity_mixin`` stays
    one-way at import time.
    """
    from hermes_cli import main

    return main


# ─── Desktop exe integrity gate (#69179) ────────────────────────────────────
#
# The desktop self-update chain (Desktop → hermes-setup --update →
# `hermes update` → `hermes desktop --build-only` → relaunch) rebuilds
# Hermes.exe on the end user's machine and used to verify only that the file
# EXISTS before declaring success. A corrupt cached Electron zip whose
# extraction produced a truncated electron.exe, an interrupted rcedit resource
# rewrite, a disk-full pack, or a wrong-arch unpacked tree therefore shipped a
# broken binary that Windows refuses to load ("This app can't run on your
# computer" / 此应用无法在你的电脑上运行). These helpers parse the PE header —
# no signature infrastructure required — so a structurally broken or
# wrong-architecture Hermes.exe is caught BEFORE the updater replaces the
# working app, and the previous build can be restored from the .bak tree that
# apps/desktop/scripts/before-pack.mjs now preserves.


_PE_MACHINE_I386 = 0x014C
_PE_MACHINE_AMD64 = 0x8664
_PE_MACHINE_ARM64 = 0xAA64

_PE_MACHINE_NAMES = {
    _PE_MACHINE_I386: "x86 (32-bit)",
    _PE_MACHINE_AMD64: "x64 (AMD64)",
    _PE_MACHINE_ARM64: "ARM64",
}

_PE_MACHINE_TO_NAME = {
    _PE_MACHINE_ARM64: "ARM64",
    _PE_MACHINE_AMD64: "AMD64",
    _PE_MACHINE_I386: "X86",
}

# MACHINE_ATTRIBUTES bits (processthreadsapi.h). UserEnabled means the host
# can run user-mode code of that machine type — natively or under emulation.
_MACHINE_ATTRIBUTE_USER_ENABLED = 0x00000001


def _windows_native_machine_from_iswow64() -> Optional[str]:
    """Ask IsWow64Process2 for the OS-native machine (None if unavailable/fail).

    ctypes defaults ``GetCurrentProcess``'s restype to ``c_int``, so the
    current-process pseudo-handle ``(HANDLE)-1`` is truncated to
    ``0xFFFFFFFF`` and zero-extended into a 64-bit invalid handle. On Win64
    that makes ``IsWow64Process2`` fail with ``ERROR_INVALID_HANDLE`` (6),
    which is exactly the residual Windows-on-ARM failure after #71218: the
    gate fell through to ``PROCESSOR_ARCHITECTURE=AMD64`` (the emulated
    process arch) and rejected a correctly-built ARM64 ``Hermes.exe``.
    Binding ``restype``/``argtypes`` to ``wintypes.HANDLE`` keeps the full
    ``0xFFFFFFFFFFFFFFFF`` pseudo-handle.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.IsWow64Process2.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.USHORT),
        ctypes.POINTER(wintypes.USHORT),
    ]
    kernel32.IsWow64Process2.restype = wintypes.BOOL

    process_machine = wintypes.USHORT(0)
    native_machine = wintypes.USHORT(0)
    if not kernel32.IsWow64Process2(
        kernel32.GetCurrentProcess(),
        ctypes.byref(process_machine),
        ctypes.byref(native_machine),
    ):
        return None
    return _PE_MACHINE_TO_NAME.get(native_machine.value)


def _windows_user_runnable_pe_machines() -> Optional[set]:
    """PE machines this host can run in user mode, via GetMachineTypeAttributes.

    This asks the question the integrity gate actually cares about — "can this
    Windows host load a PE of machine X?" — instead of inferring it from a
    host-architecture name. It is also the only documented API that reports
    AMD64-on-ARM64 emulation support; ``IsWow64GuestMachineSupported`` only
    answers for 32-bit guests.

    Returns None when the API is unavailable (pre-Windows-11 build 22000) or
    reports nothing runnable, so callers fall back to name-based detection.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetMachineTypeAttributes.argtypes = [
        wintypes.USHORT,
        ctypes.POINTER(ctypes.c_int),
    ]
    kernel32.GetMachineTypeAttributes.restype = ctypes.c_long

    runnable = set()
    for machine in (_PE_MACHINE_ARM64, _PE_MACHINE_AMD64, _PE_MACHINE_I386):
        attributes = ctypes.c_int(0)
        # HRESULT: zero is success, any nonzero value is a failure.
        if kernel32.GetMachineTypeAttributes(machine, ctypes.byref(attributes)):
            continue
        if attributes.value & _MACHINE_ATTRIBUTE_USER_ENABLED:
            runnable.add(machine)
    return runnable or None


def _windows_native_machine() -> str:
    """The Windows host OS's NATIVE machine architecture, normalized upper.

    ``platform.machine()`` reports the PROCESS architecture, which lies under
    emulation: the desktop update chain runs an x64 hermes-setup.exe (and thus
    x64 Python) on Windows-on-ARM devices, where ``platform.machine()``
    returns ``AMD64`` even though the OS is ARM64. The #71119 integrity gate
    then rejected the CORRECT ARM64 rebuild as an "architecture mismatch"
    (#69179 follow-up report). Probe order:

    1. ``IsWow64Process2`` with a correctly-typed current-process HANDLE
       (#71218 + HANDLE-truncation fix). This is the only API that tells the
       truth from an x64 process emulated on ARM64.
    2. ``PROCESSOR_ARCHITEW6432`` / ``PROCESSOR_ARCHITECTURE`` — WOW64
       (32-bit) hosts and pre-1511 Windows 10 without the newer API.
    3. ``platform.machine()``.

    Note ``GetNativeSystemInfo`` is deliberately NOT used: Microsoft documents
    that it "also returns emulated processor details when run from an app
    under emulation", so on the very WoA hosts this function exists to serve
    it reports AMD64 — no better than the env-var rung below it.
    """
    if sys.platform == "win32":
        try:
            name = _windows_native_machine_from_iswow64()
        except (OSError, AttributeError, TypeError, ValueError):
            # API missing (pre-1511), DLL load failure in tests, or a
            # mistyped ctypes binding — fall through to the env vars.
            name = None
        if name:
            return name
        env_arch = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get(
            "PROCESSOR_ARCHITECTURE"
        )
        if env_arch:
            return env_arch.upper()
    import platform as _platform

    return (_platform.machine() or "").upper()


def _expected_windows_pe_machines() -> set:
    """PE machine values the current Windows host can natively load.

    Preferred source is ``GetMachineTypeAttributes``, which answers this
    question directly (including AMD64-on-ARM64 emulation) instead of
    inferring it from an architecture name.

    Fallback is name-based: AMD64 hosts run x64 and (via WOW64) x86. ARM64
    hosts run ARM64 and (Windows 11 emulation) x64. 32-bit x86 hosts run only
    x86. Unknown machines return the permissive full set so the integrity gate
    can never brick launch on exotic hosts. Host detection uses the OS-native
    machine (see ``_windows_native_machine``), not the process architecture.
    """
    if sys.platform == "win32":
        try:
            runnable = _windows_user_runnable_pe_machines()
        except (OSError, AttributeError, TypeError, ValueError):
            runnable = None
        if runnable:
            return runnable
    machine = _m()._windows_native_machine().upper()
    if machine in ("AMD64", "X86_64", "X64"):
        return {_PE_MACHINE_AMD64, _PE_MACHINE_I386}
    if machine in ("ARM64", "AARCH64"):
        return {_PE_MACHINE_ARM64, _PE_MACHINE_AMD64}
    if machine in ("X86", "I386", "I486", "I586", "I686"):
        return {_PE_MACHINE_I386}
    return {_PE_MACHINE_AMD64, _PE_MACHINE_ARM64, _PE_MACHINE_I386}


def _parse_pe_machine(path: Path) -> int:
    """Parse ``path`` as a PE executable and return its COFF machine field.

    Raises ``ValueError`` with a human-readable reason when the file is not a
    structurally complete PE: missing MZ/PE magic (an HTML error page or JSON
    body saved as .exe), header truncation, or raw section data extending past
    the end of the file (the truncated-download / interrupted-extraction
    shape). Purely a header walk — cheap even on a 200 MB Electron exe.
    """
    import struct

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"unreadable: {exc}")
    if file_size < 512:
        raise ValueError(
            f"file is only {file_size} bytes — far too small to be a Windows executable"
        )
    with path.open("rb") as fh:
        head = fh.read(64)
        if len(head) < 64 or head[:2] != b"MZ":
            raise ValueError(
                "missing MZ header — not a Windows executable "
                "(a truncated or non-binary file saved as .exe?)"
            )
        e_lfanew = struct.unpack_from("<I", head, 0x3C)[0]
        if e_lfanew <= 0 or e_lfanew + 24 > file_size:
            raise ValueError("corrupt DOS header: PE header offset points past end of file")
        fh.seek(e_lfanew)
        pe_head = fh.read(24)
        if len(pe_head) < 24 or pe_head[:4] != b"PE\x00\x00":
            raise ValueError("missing PE signature — corrupt executable header")
        machine, n_sections = struct.unpack_from("<HH", pe_head, 4)
        size_of_optional = struct.unpack_from("<H", pe_head, 20)[0]
        fh.seek(e_lfanew + 24 + size_of_optional)
        max_section_end = 0
        for _ in range(n_sections):
            section = fh.read(40)
            if len(section) < 40:
                raise ValueError("truncated PE section table")
            size_of_raw, pointer_to_raw = struct.unpack_from("<II", section, 16)
            max_section_end = max(max_section_end, pointer_to_raw + size_of_raw)
        if file_size < max_section_end:
            raise ValueError(
                f"truncated executable: file is {file_size} bytes but its PE "
                f"sections extend to {max_section_end} bytes"
            )
    return machine


def _pe_machine_or_none(path: Path) -> Optional[int]:
    try:
        return _parse_pe_machine(path)
    except ValueError:
        return None


def _desktop_exe_integrity_error(path: Path) -> Optional[str]:
    """Return a human-readable reason ``path`` cannot run on this Windows host,
    or ``None`` when the exe parses as a complete PE of a loadable architecture.
    """
    try:
        machine = _parse_pe_machine(path)
    except ValueError as exc:
        return str(exc)
    expected = _expected_windows_pe_machines()
    if machine not in expected:
        got = _PE_MACHINE_NAMES.get(machine, f"unknown machine 0x{machine:04X}")
        return (
            f"architecture mismatch: built a {got} executable but this is a "
            f"{_m()._windows_native_machine()} Windows host"
        )
    return None


def _desktop_backup_unpacked_dir(packaged_executable: Path) -> Path:
    """The rollback tree before-pack.mjs preserves: ``<unpacked-dir>.bak``."""
    unpacked = packaged_executable.parent
    return unpacked.parent / (unpacked.name + ".bak")


def _rollback_desktop_from_backup(packaged_executable: Path) -> Optional[Path]:
    """Restore the previous unpacked desktop app from its ``.bak`` tree.

    Returns the restored executable path, or ``None`` when no usable backup
    exists (missing, or its exe fails the same integrity probe). The corrupt
    tree is kept alongside as ``<unpacked-dir>.corrupt`` for diagnostics.
    Best-effort: never raises.
    """
    unpacked = packaged_executable.parent
    backup_dir = _desktop_backup_unpacked_dir(packaged_executable)
    backup_exe = backup_dir / packaged_executable.name
    if not backup_exe.exists():
        return None
    if _desktop_exe_integrity_error(backup_exe) is not None:
        return None
    corrupt_dir = unpacked.parent / (unpacked.name + ".corrupt")
    try:
        shutil.rmtree(corrupt_dir, ignore_errors=True)
        try:
            unpacked.rename(corrupt_dir)
        except OSError:
            shutil.rmtree(unpacked, ignore_errors=True)
        backup_dir.rename(unpacked)
    except OSError:
        return None
    restored = unpacked / packaged_executable.name
    return restored if restored.exists() else None


def _ensure_desktop_exe_launchable(
    desktop_dir: Path, packaged_executable: Optional[Path]
) -> tuple:
    """Windows post-build integrity gate for the self-update rebuild (#69179).

    Returns ``(verified_exe_or_None, rolled_back)``:

    - exe passed the probe → ``(exe, False)``
    - exe corrupt/wrong-arch, previous build restored → ``(old_exe, True)``
    - exe corrupt and nothing restorable → ``(None, False)``

    On any integrity failure the corrupt cached Electron zip is purged and the
    desktop build stamp invalidated, so the updater's retry-once rebuild pulls
    a fresh, SHASUM-verified Electron download instead of re-staging the same
    corrupt bytes. No-op off Windows and when there is no executable to check.
    """
    if packaged_executable is None or sys.platform != "win32":
        return packaged_executable, False

    error = _desktop_exe_integrity_error(packaged_executable)
    if error is None:
        return packaged_executable, False

    print(f"✗ The built Hermes.exe failed its integrity check: {error}")
    print(f"    at: {packaged_executable}")

    # Self-heal setup for the retry: drop the (likely corrupt) cached Electron
    # zip and the content stamp so the next rebuild is a genuine re-download +
    # re-stage rather than a replay of the same broken extraction.
    _m()._purge_electron_build_cache(desktop_dir)
    try:
        _m()._desktop_stamp_path().unlink()
    except OSError:
        pass

    restored = _rollback_desktop_from_backup(packaged_executable)
    if restored is not None:
        print("  ↩ Update aborted — restored the previous working Hermes.exe from backup.")
        print("    Your existing version was kept and still works. Run `hermes desktop`")
        print("    (or the in-app update) again to retry with a fresh Electron download.")
        return restored, True

    print("  ✗ No usable backup was found to restore.")
    print("    Run `hermes desktop --force-build` to rebuild, or re-run the Hermes")
    print("    installer to repair the install.")
    return None, False
