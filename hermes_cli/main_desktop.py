"""Desktop (Electron) app: build/stamp, stage-and-swap pack, exe integrity gate, macOS signing/TCC, Linux sandbox, launch (hermes gui/desktop).

Split out of ``hermes_cli/main.py``. Names that still live in main (``PROJECT_ROOT``, ...)
are imported lazily inside the functions that use them (avoids an import cycle).
"""

import logging
import contextlib
import argparse
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time as _time_mod

from pathlib import Path
from typing import Optional

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.main")

_PREVIOUS_APP_KEPT = "  ↩ The previous desktop app was left untouched and still works."


def _desktop_dist_exists(desktop_dir: Path) -> bool:
    """Return True when a local desktop renderer build is present."""
    return (desktop_dir / "dist" / "index.html").exists()


def _renderer_bundle_dir(desktop_dir: Path, *, source_mode: bool) -> Optional[Path]:
    """The renderer ``dist`` a launch loads: ``apps/desktop/dist`` in source mode, else the
    ``app.asar.unpacked/dist`` copy (the only real directory, and the one an interrupted replace tears)."""
    if source_mode:
        return desktop_dir / "dist"

    executable = _desktop_packaged_executable(desktop_dir)
    if executable is None:
        return None

    # macOS: …/Hermes.app/Contents/MacOS/Hermes → …/Contents/Resources
    resources = (
        executable.parent.parent / "Resources" if sys.platform == "darwin" else executable.parent / "resources"
    )
    return resources / "app.asar.unpacked" / "dist"


# The module files the renderer fetches before any app code runs: Vite emits
# them as `<script type="module" src>` plus `<link rel="modulepreload" href>`.
_HTML_TAG_WITH_URL = re.compile(r"""<(?:script|link)\b[^>]*\b(?:src|href)=["']([^"']+)["'][^>]*>""", re.IGNORECASE)

_MODULE_TAG = re.compile(r"""\btype=["']module["']|\brel=["']modulepreload["']""", re.IGNORECASE)


def _renderer_bundle_torn(dist_dir: Path) -> bool:
    """True when ``index.html`` names hashed module chunks that aren't there.

    A replace interrupted by locked files leaves index and ``assets/`` from
    different generations; the app dies on its first lazy import while the
    SOURCE-tree stamp still matches, so no rebuild fixes it. Conservative: an
    unreadable index or one naming nothing checkable is NOT torn.
    """
    try:
        html = (dist_dir / "index.html").read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False

    for match in _HTML_TAG_WITH_URL.finditer(html):
        href = match.group(1)
        # Absolute/CDN URLs aren't part of this bundle's generation.
        if not _MODULE_TAG.search(match.group(0)) or re.match(r"^[a-z]+:|^//", href, re.IGNORECASE):
            continue
        rel = href.split("?", 1)[0].split("#", 1)[0].lstrip("./")
        if rel and not (dist_dir / rel).exists():
            return True

    return False


def _desktop_build_needed(desktop_dir: Path, project_root: Path, *, source_mode: bool) -> bool:
    """True when the desktop build output is stale, missing, torn, or built in the other mode."""
    if source_mode:
        if not _desktop_dist_exists(desktop_dir):
            return True
    elif _desktop_packaged_executable(desktop_dir) is None:
        return True

    # A torn bundle is stale no matter what the stamp says: the hash describes
    # the intact SOURCE tree, not the half-replaced output.
    dist_dir = _renderer_bundle_dir(desktop_dir, source_mode=source_mode)
    if dist_dir is not None and _renderer_bundle_torn(dist_dir):
        print(f"  ⚠ A previous update left the desktop bundle incomplete ({dist_dir}); rebuilding it")
        return True

    from hermes_cli.source_build import source_product_current

    return dist_dir is None or not source_product_current(project_root, "desktop", dist_dir)


def _desktop_packaged_executable(desktop_dir: Path) -> Optional[Path]:
    """Return the current platform's unpacked Electron app executable."""
    return _desktop_packaged_executable_in(desktop_dir / "release")


def _desktop_packaged_executable_in(release_dir: Path) -> Optional[Path]:
    """The unpacked Electron app executable under *release_dir* (live ``release`` or a staging dir).

    *release_dir* is electron-builder's ``directories.output`` — the live ``apps/desktop/release`` or a
    stage-and-swap staging dir (#86443).
    """
    if sys.platform == "darwin":
        candidates = list(release_dir.glob("mac*/Hermes.app/Contents/MacOS/Hermes"))
    elif sys.platform == "win32":
        candidates = [
            release_dir / d / "Hermes.exe" for d in ("win-unpacked", "win-ia32-unpacked", "win-arm64-unpacked")
        ]
    else:
        candidates = [
            release_dir / d / n for d in ("linux-unpacked", "linux-arm64-unpacked") for n in ("hermes", "Hermes")
        ]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    if sys.platform == "win32" and len(existing) > 1:
        # A stale win-arm64-unpacked next to the real win-unpacked: picking by
        # mtime can hand a wrong-architecture Hermes.exe to the launcher. Prefer
        # candidates whose PE machine matches the host; mtime when none parse.
        # Multiple unpacked trees can coexist (e.g. a stale win-arm64-unpacked left behind by a cross-arch
        # experiment next to the real win-unpacked). Picking purely by mtime can then hand a
        # wrong-architecture Hermes.exe to the launcher, which Windows rejects with "This app can't run on
        # your computer" (#69179).
        expected = _expected_windows_pe_machines()
        matching = [p for p in existing if _pe_machine_or_none(p) in expected]
        if matching:
            existing = matching
    return max(existing, key=lambda p: p.stat().st_mtime)


# ─── Desktop stage-and-swap pack (#86443) ─────────────────────────────────── electron-builder packs IN
# PLACE: before-pack.mjs wipes ``release/<platform>- unpacked`` (or the mac ``Hermes.app``) and the Electron
# unpack + asar + rename then rebuild it. Any failure after that wipe — corrupt cached zip, blocked
# download, missing dep, disk full — leaves the user with NO app, and ``hermes update`` used to report
# "partially complete" over an empty release/. Fix the class, not the predicate: build into a STAGING output
# dir next to release/, verify the staged result, and only then swap it over the live tree with renames. On
# any failure the live app is untouched.
_DESKTOP_STAGING_PREFIX = ".staging-"

_DESKTOP_PREVIOUS_SUFFIX = ".previous"


def _desktop_staging_dir(desktop_dir: Path) -> Path:
    """Fresh staging dir ``apps/desktop/.staging-<pid>-<ts>``: a sibling of ``release/`` (same fs → the
    swap is a rename) but not inside it, so ``release/*-unpacked`` globs never see it. Sweeps leftovers."""
    for stale in desktop_dir.glob(f"{_DESKTOP_STAGING_PREFIX}*"):
        shutil.rmtree(stale, ignore_errors=True)
    return desktop_dir / f"{_DESKTOP_STAGING_PREFIX}{os.getpid()}-{int(_time_mod.time())}"


def _desktop_unpacked_root(exe: Path, release_dir: Path) -> Path:
    """The dir directly under *release_dir* holding *exe* (electron-builder's ``appOutDir``, swapped whole)."""
    unpacked = exe
    while unpacked.parent != release_dir:
        if unpacked.parent == unpacked:
            raise ValueError(f"{exe} is not under {release_dir}")
        unpacked = unpacked.parent
    return unpacked


def _swap_staged_desktop_app(desktop_dir: Path, staging_dir: Path) -> Optional[Path]:
    """Promote a VERIFIED staged pack over ``release/`` by two renames (live → ``.previous``, staged →
    live); a failure between them rolls back. Returns the live exe or None (live app kept). Never raises."""
    staged_exe = _desktop_packaged_executable_in(staging_dir)
    if staged_exe is None:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return None
    release_dir = desktop_dir / "release"
    try:
        staged_root = _desktop_unpacked_root(staged_exe, staging_dir)
        live_root = release_dir / staged_root.name
        previous = release_dir / (staged_root.name + _DESKTOP_PREVIOUS_SUFFIX)
        release_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(previous, ignore_errors=True)
        moved_aside = live_root.exists()
        if moved_aside:
            # A Desktop may have reopened during the long packaging step.
            stopped = _stop_desktop_processes_locking_build(desktop_dir)
            if stopped:
                logger.info("stopped desktop processes before staged app promotion: %s", stopped)
            os.rename(live_root, previous)
        try:
            os.rename(staged_root, live_root)
        except OSError:
            if moved_aside:
                os.rename(previous, live_root)  # restore; live app back as it was
            raise
        if moved_aside:
            shutil.rmtree(previous, ignore_errors=True)
    except (OSError, ValueError) as exc:
        logger.warning("desktop stage-and-swap failed, live app kept: %s", exc)
        return None
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return live_root / staged_exe.relative_to(staged_root)


def _discard_desktop_staging(staging_dir: Path) -> None:
    shutil.rmtree(staging_dir, ignore_errors=True)


# ─── Desktop exe integrity gate (#69179) ──────────────────────────────────── The desktop self-update chain
# (Desktop → hermes-setup --update → `hermes update` → `hermes desktop --build-only` → relaunch) rebuilds
# Hermes.exe on the end user's machine and used to verify only that the file EXISTS before declaring
# success. A corrupt cached Electron zip whose extraction produced a truncated electron.exe, an interrupted
# rcedit resource rewrite, a disk-full pack, or a wrong-arch unpacked tree therefore shipped a broken binary
# that Windows refuses to load ("This app can't run on your computer" / 此应用无法在你的电脑上运行). These helpers parse
# the PE header — no signature infrastructure required — so a structurally broken or wrong-architecture
# Hermes.exe is caught BEFORE the updater replaces the working app, and the previous build can be restored
# from the .bak tree that apps/desktop/scripts/before-pack.mjs now preserves.
_PE_MACHINE_I386 = 0x014C
_PE_MACHINE_AMD64 = 0x8664
_PE_MACHINE_ARM64 = 0xAA64

_PE_MACHINE_NAMES = {
    _PE_MACHINE_I386: "x86 (32-bit)", _PE_MACHINE_AMD64: "x64 (AMD64)", _PE_MACHINE_ARM64: "ARM64",
}

_PE_MACHINE_TO_NAME = {_PE_MACHINE_ARM64: "ARM64", _PE_MACHINE_AMD64: "AMD64", _PE_MACHINE_I386: "X86"}

# MACHINE_ATTRIBUTES bits (processthreadsapi.h). UserEnabled means the host
# can run user-mode code of that machine type — natively or under emulation.
_MACHINE_ATTRIBUTE_USER_ENABLED = 0x00000001


def _kernel32():
    import ctypes
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _windows_native_machine_from_iswow64() -> Optional[str]:
    """IsWow64Process2's OS-native machine, or None. HANDLE types are bound explicitly: ctypes'
    default ``c_int`` truncates the ``(HANDLE)-1`` pseudo-handle → ``ERROR_INVALID_HANDLE`` on Win64.

    ctypes defaults ``GetCurrentProcess``'s restype to ``c_int``, so the current-process pseudo-handle
    ``(HANDLE)-1`` is truncated to ``0xFFFFFFFF`` and zero-extended into a 64-bit invalid handle. On Win64
    that makes ``IsWow64Process2`` fail with ``ERROR_INVALID_HANDLE`` (6), which is exactly the residual
    Windows-on-ARM failure after #71218: the gate fell through to ``PROCESSOR_ARCHITECTURE=AMD64`` (the
    emulated process arch) and rejected a correctly-built ARM64 ``Hermes.exe``. Binding
    ``restype``/``argtypes`` to ``wintypes.HANDLE`` keeps the full ``0xFFFFFFFFFFFFFFFF`` pseudo-handle.
    """
    import ctypes
    from ctypes import wintypes
    kernel32 = _kernel32()
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.IsWow64Process2.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT)]
    kernel32.IsWow64Process2.restype = wintypes.BOOL

    process_machine = wintypes.USHORT(0)
    native_machine = wintypes.USHORT(0)
    if not kernel32.IsWow64Process2(
        kernel32.GetCurrentProcess(), ctypes.byref(process_machine), ctypes.byref(native_machine)):
        return None
    return _PE_MACHINE_TO_NAME.get(native_machine.value)


def _windows_user_runnable_pe_machines() -> Optional[set]:
    """PE machines this host runs in user mode via GetMachineTypeAttributes (the only API reporting
    AMD64-on-ARM64 emulation); None when unavailable (pre-Win11 22000) so callers fall back."""
    import ctypes
    from ctypes import wintypes
    kernel32 = _kernel32()
    kernel32.GetMachineTypeAttributes.argtypes = [wintypes.USHORT, ctypes.POINTER(ctypes.c_int)]
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
    """The Windows host's NATIVE machine, upper-cased: ``IsWow64Process2`` (the only API that tells
    the truth from an emulated x64 process on ARM64), then ``PROCESSOR_ARCHITEW6432`` /
    ``PROCESSOR_ARCHITECTURE``, then ``platform.machine()`` (which lies under emulation).
    ``GetNativeSystemInfo`` is NOT used: it also returns emulated details.

    ``platform.machine()`` reports the PROCESS architecture, which lies under emulation: the desktop update
    chain runs an x64 hermes-setup.exe (and thus x64 Python) on Windows-on-ARM devices, where
    ``platform.machine()`` returns ``AMD64`` even though the OS is ARM64. The #71119 integrity gate then
    rejected the CORRECT ARM64 rebuild as an "architecture mismatch" (#69179 follow-up report). Probe order:
    1. ``IsWow64Process2`` with a correctly-typed current-process HANDLE (#71218 + HANDLE-truncation fix).
    2. 3.
    """
    if sys.platform == "win32":
        try:
            name = _windows_native_machine_from_iswow64()
        except (OSError, AttributeError, TypeError, ValueError):
            name = None  # API missing, DLL load failure in tests, mistyped binding
        if name:
            return name
        env_arch = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE")
        if env_arch:
            return env_arch.upper()
    import platform as _platform

    return (_platform.machine() or "").upper()


def _expected_windows_pe_machines() -> set:
    """PE machines this Windows host can load: ``GetMachineTypeAttributes``, else by name (AMD64 → x64+x86,
    ARM64 → ARM64+x64, x86 → x86). Unknown hosts get the full set so the gate can never brick launch."""
    if sys.platform == "win32":
        try:
            runnable = _windows_user_runnable_pe_machines()
        except (OSError, AttributeError, TypeError, ValueError):
            runnable = None
        if runnable:
            return runnable
    machine = _windows_native_machine().upper()
    if machine in ("AMD64", "X86_64", "X64"):
        return {_PE_MACHINE_AMD64, _PE_MACHINE_I386}
    if machine in ("ARM64", "AARCH64"):
        return {_PE_MACHINE_ARM64, _PE_MACHINE_AMD64}
    if machine in ("X86", "I386", "I486", "I586", "I686"):
        return {_PE_MACHINE_I386}
    return {_PE_MACHINE_AMD64, _PE_MACHINE_ARM64, _PE_MACHINE_I386}


def _parse_pe_machine(path: Path) -> int:
    """COFF machine field of the PE at ``path``; ``ValueError`` with a readable reason when it is not a
    structurally complete PE (bad magic, truncated header, section data past EOF). Header walk only."""
    import struct
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"unreadable: {exc}")
    if file_size < 512:
        raise ValueError(f"file is only {file_size} bytes — far too small to be a Windows executable")
    with path.open("rb") as fh:
        head = fh.read(64)
        if len(head) < 64 or head[:2] != b"MZ":
            raise ValueError(
                "missing MZ header — not a Windows executable (a truncated or non-binary file saved as .exe?)"
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
                f"truncated executable: file is {file_size} bytes but its PE sections extend to {max_section_end} bytes"
            )
    return machine


def _pe_machine_or_none(path: Path) -> Optional[int]:
    try:
        return _parse_pe_machine(path)
    except ValueError:
        return None


def _desktop_exe_integrity_error(path: Path) -> Optional[str]:
    """Why ``path`` cannot run on this Windows host, or None when it parses as a loadable PE."""
    try:
        machine = _parse_pe_machine(path)
    except ValueError as exc:
        return str(exc)
    if machine not in _expected_windows_pe_machines():
        got = _PE_MACHINE_NAMES.get(machine, f"unknown machine 0x{machine:04X}")
        return (
            f"architecture mismatch: built a {got} executable but this is a "
            f"{_windows_native_machine()} Windows host"
        )
    return None


def _desktop_backup_unpacked_dir(packaged_executable: Path) -> Path:
    """The rollback tree before-pack.mjs preserves: ``<unpacked-dir>.bak``."""
    unpacked = packaged_executable.parent
    return unpacked.parent / (unpacked.name + ".bak")


def _rollback_desktop_from_backup(packaged_executable: Path) -> Optional[Path]:
    """Restore the previous unpacked desktop app from its ``.bak`` tree.

    None when no usable backup exists (missing, or fails the same integrity
    probe). The corrupt tree is kept as ``<unpacked-dir>.corrupt``. Never raises.
    """
    unpacked = packaged_executable.parent
    backup_dir = _desktop_backup_unpacked_dir(packaged_executable)
    backup_exe = backup_dir / packaged_executable.name
    if not backup_exe.exists() or _desktop_exe_integrity_error(backup_exe) is not None:
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


def _ensure_desktop_exe_launchable(desktop_dir: Path, packaged_executable: Optional[Path]) -> tuple:
    """Windows post-build integrity gate → ``(verified_exe_or_None, rolled_back)``: pass →
    ``(exe, False)``; corrupt with backup restored → ``(old_exe, True)``; nothing restorable →
    ``(None, False)``. Corrupt staged output never replaces the running app.

    See #69179.
    """
    if packaged_executable is None or sys.platform != "win32":
        return packaged_executable, False

    error = _desktop_exe_integrity_error(packaged_executable)
    if error is None:
        return packaged_executable, False

    print(f"✗ The built Hermes.exe failed its integrity check: {error}\n    at: {packaged_executable}")


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


def _electron_dir(project_root: Path) -> Path:
    """The installed Electron package dir: workspace-local ``apps/desktop/node_modules/electron`` (where
    ``electronDist`` points) when present, else the root hoist npm sometimes uses instead."""
    desktop_local = project_root / "apps" / "desktop" / "node_modules" / "electron"
    if desktop_local.exists():
        return desktop_local
    return project_root / "node_modules" / "electron"


def _stop_desktop_processes_locking_build(desktop_dir: Path) -> list[int]:
    """Terminate a running desktop app whose exe lives INSIDE this build's ``release`` tree (Windows
    only — its lock makes the pack die with ``Access is denied``; POSIX can unlink a running
    binary). Never raises; returns the PIDs asked to stop."""
    if sys.platform != "win32":
        return []
    try:
        import psutil
        release_dir = (desktop_dir / "release").resolve()
    except Exception:
        return []
    if not release_dir.is_dir():
        return []

    me = os.getpid()
    victims = []
    try:
        proc_iter = psutil.process_iter(["pid", "exe"])
    except Exception:
        return []
    for proc in proc_iter:
        try:
            info = proc.info
            pid = info.get("pid")
            exe = info.get("exe")
            if not exe or pid is None or pid == me:
                continue
            exe_path = Path(exe).resolve()
        except Exception:
            continue
        if release_dir in exe_path.parents:
            victims.append(proc)

    stopped: list[int] = []
    for proc in victims:
        try:
            proc.terminate()
            stopped.append(int(proc.pid))
        except Exception:
            continue
    if stopped:
        # Wait for the handles (and thus the file locks) to actually release.
        with contextlib.suppress(Exception):
            _, alive = psutil.wait_procs(victims, timeout=5)
            killed = []
            for proc in alive:
                try:
                    proc.kill()
                    killed.append(proc)
                except Exception:
                    continue
            if killed:
                psutil.wait_procs(killed, timeout=5)
    return stopped


def _desktop_macos_bundle_id(bundle: Path) -> Optional[str]:
    """Return a bundle/framework CFBundleIdentifier for local macOS signing."""
    import plistlib
    info = bundle / "Contents" / "Info.plist"
    if not info.exists() and bundle.suffix == ".framework":
        candidates = list(bundle.glob("Versions/*/Resources/Info.plist")) + list(
            bundle.glob("Resources/Info.plist"))
        if candidates:
            info = candidates[0]
    if not info.exists():
        return None
    try:
        data = plistlib.loads(info.read_bytes())
    except Exception:
        return None
    ident = data.get("CFBundleIdentifier")
    return str(ident) if ident else None


def _desktop_macos_local_signing_identity() -> Optional[str]:
    """``desktop.macos_signing_identity`` — a persistent (even self-signed) code-signing cert anchors
    the Designated Requirement and keeps TCC grants stable across rebuilds. Unset → ad-hoc."""
    if sys.platform != "darwin":
        return None
    try:
        from hermes_cli.config import load_config
        desktop = load_config().get("desktop", {})
        if not isinstance(desktop, dict):
            return None
        identity = desktop.get("macos_signing_identity")
        if not isinstance(identity, str):
            return None
        return identity.strip() or None
    except Exception as exc:
        print(
            "  (warning: could not load desktop.macos_signing_identity: "
            f"{exc}; falling back to ad-hoc signing)"
        )
        return None


def _codesign_verify(codesign: str, app: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        [codesign, "--verify", "--deep", "--strict", str(app)], capture_output=True, **kwargs)


def _desktop_macos_has_valid_real_signature(app: Path) -> bool:
    """True when the bundle has an intact Team-ID signature, so the fixup never clobbers a notarized
    build with ad-hoc (resets TCC). A STALE real signature fails --verify → False → repairable."""
    codesign = shutil.which("codesign")
    if not codesign:
        return False
    try:
        info = subprocess.run(
            [codesign, "-dv", str(app)], check=False, capture_output=True, text=True)
        output = f"{info.stdout}\n{info.stderr}"
        if info.returncode != 0 or "TeamIdentifier=" not in output or "TeamIdentifier=not set" in output:
            return False
        return _codesign_verify(codesign, app, check=False).returncode == 0
    except Exception:
        return False


def _desktop_macos_local_codesign(app: Path, *, desktop_dir: Path, identity: str = "-") -> bool:
    """Sign a local build inside-out (Mach-O files, nested frameworks/helpers, main bundle) with the
    repo's entitlements and an identifier-pinned DR when ad-hoc — a plain ``--deep --sign -`` gives
    a cdhash-only DR (TCC re-prompts every rebuild) and strips the JIT/mic entitlements.
    Raises on signing failure; True after strict verification."""
    codesign = shutil.which("codesign")
    if not codesign:
        return False

    ent_main = desktop_dir / "electron" / "entitlements.mac.plist"
    ent_inherit = desktop_dir / "electron" / "entitlements.mac.inherit.plist"
    if not (ent_main.exists() and ent_inherit.exists()):
        # Hardened-runtime restrictions apply to ad-hoc signatures too; signing
        # with --options runtime but WITHOUT allow-jit would leave Electron/V8
        # crashing on launch. Bail so the caller falls back to the legacy sign.
        raise FileNotFoundError(f"desktop entitlement plists missing under {desktop_dir / 'electron'}")

    def sign_path(
        path: Path, *, entitlements: Optional[Path] = None, identifier: Optional[str] = None,
        runtime: bool = True) -> None:
        args = [codesign, "--force", "--sign", identity, "--timestamp=none"]
        if runtime:
            args += ["--options", "runtime"]
        if entitlements is not None and entitlements.exists():
            args += ["--entitlements", str(entitlements)]
        if identifier and identity == "-":
            # Ad-hoc signatures get a cdhash-only DR by default; pin an
            # identifier-based DR so TCC has something stable to persist.
            args += ["--requirements", f'=designated => identifier "{identifier}"']
        args.append(str(path))
        subprocess.run(args, check=True, capture_output=True)

    # 1) Standalone Mach-O files (native modules, dylibs, crashpad handler),
    #    compared relative to the app root — the absolute path always contains
    #    the outer Hermes.app component.
    contents = app / "Contents"
    standalone: list[Path] = []
    for root, _dirs, files in os.walk(contents):
        root_path = Path(root)
        if any(part.endswith(".app") for part in root_path.relative_to(app).parts):
            continue  # nested helper apps are signed as bundles below
        for name in files:
            fp = root_path / name
            if name in {"chrome_crashpad_handler", "spawn-helper"} or fp.suffix in {".node", ".dylib"}:
                standalone.append(fp)
    for fp in sorted(standalone, key=lambda p: len(p.parts), reverse=True):
        sign_path(fp, runtime=False)

    # 2) Nested frameworks and helper apps, deepest first.
    bundles: set[Path] = set()
    frameworks_dir = contents / "Frameworks"
    if frameworks_dir.exists():
        for root, _dirs, _files in os.walk(frameworks_dir):
            p = Path(root)
            if p.suffix in {".framework", ".app"}:
                bundles.add(p)
    for bundle in sorted(bundles, key=lambda p: len(p.parts), reverse=True):
        ent = ent_inherit if bundle.suffix == ".app" and "Helper" in bundle.name else None
        sign_path(bundle, entitlements=ent, identifier=_desktop_macos_bundle_id(bundle))

    # 3) The main bundle, with the app's own entitlements.
    sign_path(app, entitlements=ent_main, identifier=_desktop_macos_bundle_id(app))
    _codesign_verify(codesign, app, check=True)
    return True


def _macos_legacy_adhoc_resign(codesign: str, app: Path) -> bool:
    """Legacy deep ad-hoc re-sign; NEVER deletes the safeStorage keychain item (that would orphan every
    credential under it, and there is no verified successor identity here — the "Always Allow"
    prompt is recoverable, deletion is not)."""
    try:
        result = subprocess.run(
            [codesign, "--force", "--deep", "--sign", "-", str(app)], check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(
                f"  (warning: legacy ad-hoc re-sign failed (exit {result.returncode}); "
                "leaving safeStorage keychain item untouched)"
            )
            return False
        if _codesign_verify(codesign, app, check=False, text=True).returncode != 0:
            print(
                "  (warning: legacy ad-hoc re-sign did not pass strict verification; "
                "leaving safeStorage keychain item untouched)"
            )
            return False
        print("  → macOS desktop re-signed (legacy ad-hoc); safeStorage keychain item left untouched")
        return True
    except Exception as exc:
        print(f"  (warning: macOS relaunch fixup skipped: {exc})")
    return False


def _desktop_macos_relaunchable_fixup(
    desktop_dir: Path, *, publisher_signing_configured: Optional[bool] = None,
    release_dir: Optional[Path] = None) -> bool:
    """Re-sign a locally-built macOS app so in-place self-update doesn't reset TCC grants.

    A rebuilt ad-hoc bundle (new cdhash, no stable Designated Requirement) reports
    "Hermes is damaged" and loses every grant. Clear quarantine xattrs, then sign
    with ``desktop.macos_signing_identity`` or identifier-pinned ad-hoc, keeping
    entitlements; legacy deep ad-hoc as fallback. No-op with a publisher identity
    (CSC_LINK / APPLE_SIGNING_IDENTITY; callers may pass the decision so a later
    dotenv load can't reverse it) or an intact Developer ID signature.
    ``release_dir`` signs the STAGED bundle before promotion. Never raises.
    """
    if sys.platform != "darwin":
        return True
    if publisher_signing_configured is None:
        publisher_signing_configured = bool(
            os.environ.get("CSC_LINK") or os.environ.get("APPLE_SIGNING_IDENTITY"))
    if publisher_signing_configured:
        return True
    # ``release_dir`` (stage-and-swap, #86443): sign the STAGED bundle before it is promoted, so the live
    # app is never touched mid-sign.
    exe = _desktop_packaged_executable_in(release_dir or (desktop_dir / "release"))
    if exe is None:
        return True
    # exe = .../Hermes.app/Contents/MacOS/Hermes  ->  app bundle = .../Hermes.app
    app = exe.parents[2]
    if not str(app).endswith(".app") or not app.is_dir():
        return True
    codesign = shutil.which("codesign")
    if not codesign:
        return False
    if _desktop_macos_has_valid_real_signature(app):
        return True
    subprocess.run(["xattr", "-cr", str(app)], check=False)
    identity = _desktop_macos_local_signing_identity() or "-"
    try:
        if _desktop_macos_local_codesign(app, desktop_dir=desktop_dir, identity=identity):
            label = "keychain identity" if identity != "-" else "stable ad-hoc identity"
            print(f"  → macOS desktop signed with {label}; TCC grants persist across rebuilds")
            return True
    except Exception as exc:
        if identity != "-":
            print(
                f"  (warning: configured macOS signing identity failed: {identity!r}; "
                "falling back to ad-hoc — TCC grants may need to be re-granted)"
            )
        print(f"  (warning: stable macOS signing failed ({exc}); using legacy ad-hoc sign)")
    return _macos_legacy_adhoc_resign(codesign, app)


def _macos_codesigning_identity_valid(security: str, identity: str) -> bool:
    """True when `identity` is among VALID (``-v``) code-signing identities — the plain listing also
    shows untrusted certs codesign refuses. Idempotency probe + postcondition. Never raises."""
    try:
        result = subprocess.run(
            [security, "find-identity", "-v", "-p", "codesigning"], capture_output=True, text=True, check=False,
        )
    except Exception:
        return False

    return f'"{identity}"' in (result.stdout or "")


def _macos_create_signing_identity(
    openssl: str, security: str, codesign: str, keychain: str, identity: str) -> bool:
    """Create a self-signed code-signing cert (10 years), import it with codesign access, trust it for codeSign."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="hermes-tcc-"))
    try:
        key = tmp_dir / "sign.key"
        crt = tmp_dir / "sign.crt"
        p12 = tmp_dir / "sign.p12"
        subprocess.run(
            [
                openssl, "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key), "-out", str(crt),
                "-days", "3650", "-nodes",
                "-subj", f"/CN={identity}",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,digitalSignature,keyCertSign",
                "-addext", "extendedKeyUsage=codeSigning",
            ],
            capture_output=True, check=True)

        # OpenSSL 3 defaults to AES/SHA-2 PKCS#12 that `security import` rejects
        # with "MAC verification failed". `-legacy` restores the accepted
        # RC2/SHA-1 format but only exists on OpenSSL 3 — so try plain first and
        # fall back to `-legacy` when the IMPORT fails with that signature.
        # (Verified E2E on macOS 26.3.1 / OpenSSL 3.6.3 by @ctaylor86 on PR #77189.)
        def _export_p12(extra_args: list) -> None:
            subprocess.run(
                [
                    openssl, "pkcs12", "-export", *extra_args,
                    "-inkey", str(key), "-in", str(crt),
                    "-out", str(p12), "-passout", "pass:hermeslocal",
                ],
                capture_output=True, check=True)

        def _import_p12():
            return subprocess.run(
                [
                    security, "import", str(p12), "-k", keychain,
                    "-P", "hermeslocal",
                    "-T", codesign, "-T", "/usr/bin/codesign_allocate",
                ],
                capture_output=True, text=True, check=False)

        _export_p12([])
        imported = _import_p12()
        if imported.returncode != 0 and "MAC verification failed" in (imported.stderr or ""):
            # older OpenSSL without -legacy: keep the original failure
            with contextlib.suppress(subprocess.CalledProcessError):
                _export_p12(["-legacy"])
                imported = _import_p12()
        if imported.returncode != 0:
            print(f"  (could not import signing identity into keychain: {imported.stderr.strip()})")
            return False

        # Without explicit trust for the codeSign policy `find-identity -v`
        # reports 0 valid identities. This writes user trust settings, so macOS
        # may prompt for the login password ONCE — the one-time cost this
        # command exists to front-load.
        trusted = subprocess.run(
            [security, "add-trusted-cert", "-r", "trustRoot", "-p", "codeSign", "-k", keychain, str(crt)],
            capture_output=True, text=True, check=False)
        if trusted.returncode != 0:
            print(
                "  (could not trust the certificate for code signing: "
                f"{(trusted.stderr or trusted.stdout).strip()})"
            )
            return False
        print(f"  → created, imported, and trusted self-signed identity: {identity!r}")
        return True
    except Exception as exc:
        print(f"  (certificate creation failed: {exc})")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _desktop_macos_setup_tcc_identity(identity: str = "Hermes Local Signing") -> bool:
    """``--setup-tcc-identity``: create/import a self-signed code-signing cert, point
    ``desktop.macos_signing_identity`` at it and re-sign the packaged app. TCC grants follow the
    signing identity, so a certificate-anchored one is stable across rebuilds (the yabai/skhd
    mechanism). Idempotent; never raises."""
    from hermes_cli.main import PROJECT_ROOT
    if sys.platform != "darwin":
        print("  (--setup-tcc-identity is macOS-only; skipping)")
        return False

    openssl = shutil.which("openssl")
    security = shutil.which("security")
    codesign = shutil.which("codesign")
    if not (openssl and security and codesign):
        print(
            "  (--setup-tcc-identity requires openssl, security, and codesign; "
            f"found openssl={bool(openssl)} security={bool(security)} codesign={bool(codesign)})"
        )
        return False

    keychain = str(Path.home() / "Library" / "Keychains" / "login.keychain-db")
    # Probe with `-v` (valid identities only) so a previously imported-but-
    # untrusted cert is repaired rather than reported as done.
    if _macos_codesigning_identity_valid(security, identity):
        print(f"  → identity {identity!r} already valid in keychain")
    elif not _macos_create_signing_identity(openssl, security, codesign, keychain, identity):
        return False

    # Postcondition gate: name-in-output checks pass for invalid identities;
    # only macOS agreeing the identity is usable counts.
    if not _macos_codesigning_identity_valid(security, identity):
        print(
            f"  (identity {identity!r} was imported but is not a VALID code-signing identity; "
            "run `security find-identity -v -p codesigning` to inspect, and see the manual "
            "Keychain Access steps in the desktop docs)"
        )
        return False

    # config.yaml, not .env — it's not a secret.
    try:
        from hermes_cli.config import set_config_value
        set_config_value("desktop.macos_signing_identity", identity)
        print(f"  → set desktop.macos_signing_identity = {identity!r}")
    except Exception as exc:
        print(f"  (could not write desktop.macos_signing_identity: {exc})")
        return False

    desktop_dir = PROJECT_ROOT / "apps" / "desktop"
    if _desktop_packaged_executable(desktop_dir) is not None:
        try:
            if _desktop_macos_relaunchable_fixup(desktop_dir):
                print(
                    "  → packaged app re-signed with certificate-anchored identity; "
                    "TCC grants persist across rebuilds"
                )
        except Exception as exc:
            print(f"  (could not re-sign packaged app: {exc})")

    print(
        "\n  Note: macOS will re-prompt for permissions ONE final time (the identity "
        "changed). Grant them and they persist from then on. If a permission gets "
        "stuck, reset it with:  tccutil reset All com.nousresearch.hermes"
    )
    return True


def _force_adhoc_macos_signing(env: dict, *, source_mode: bool) -> bool:
    """Force ad-hoc signing for the local packaged rebuild: with ``CSC_IDENTITY_AUTO_DISCOVERY`` on,
    electron-builder grabs any personal keychain cert and stalls the sign step or clobbers a
    notarized signature. No-op for source runs, off-macOS, with a real identity, or when pinned."""
    if sys.platform != "darwin" or source_mode:
        return False
    if env.get("CSC_LINK") or env.get("APPLE_SIGNING_IDENTITY") or "CSC_IDENTITY_AUTO_DISCOVERY" in env:
        return False
    env["CSC_IDENTITY_AUTO_DISCOVERY"] = "false"
    return True


def _desktop_linux_needs_no_sandbox() -> bool:
    """True when Electron should run ``--no-sandbox``: Ubuntu 23.10+ ``apparmor_restrict_unprivileged_userns``
    breaks the userns sandbox without a root-owned 4755 helper. Deliberately NOT True for root —
    Electron as root without a sandbox must stay an explicit choice."""
    if os.environ.get("ELECTRON_DISABLE_SANDBOX", 0) == "1":
        return True

    if sys.platform != "linux":
        return False
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return False
    try:
        with open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", encoding="utf-8-sig") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def _desktop_linux_userns_sandbox_available() -> bool:
    """True when the unprivileged userns sandbox works (probed with ``unshare``, fails closed) — then
    the setuid ``chrome-sandbox`` helper is never consulted and no sudo prompt is needed."""
    if sys.platform != "linux":
        return False
    unshare = shutil.which("unshare")
    if not unshare:
        return False
    try:
        return (
            subprocess.run(
                [unshare, "--user", "--map-root-user", "true"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False,
            ).returncode
            == 0)
    except (OSError, subprocess.TimeoutExpired):
        return False


def _sandbox_helper_lstat(packaged_executable: Path) -> tuple[Path, Optional[os.stat_result]]:
    """``(chrome-sandbox path, lstat or None)`` — lstat so a symlink is inspected, not followed."""
    sandbox = packaged_executable.parent / "chrome-sandbox"
    try:
        return sandbox, sandbox.lstat()
    except OSError:
        return sandbox, None


def _sandbox_helper_is_setuid_root(st: os.stat_result) -> bool:
    return st.st_uid == 0 and stat.S_IMODE(st.st_mode) == 0o4755


def _desktop_linux_sandbox_helper_is_regular_file(packaged_executable: Path) -> bool:
    """Return True when ``chrome-sandbox`` exists as a regular file."""
    if sys.platform != "linux":
        return False
    _sandbox, st = _sandbox_helper_lstat(packaged_executable)
    return st is not None and stat.S_ISREG(st.st_mode)


def _desktop_linux_sandbox_fixup(packaged_executable: Path) -> bool:
    """Configure Electron's Linux SUID sandbox helper when required."""
    if sys.platform != "linux":
        return True

    sandbox, st = _sandbox_helper_lstat(packaged_executable)
    if not sandbox.exists():
        print(f"✗ Hermes Desktop is missing Electron's Linux sandbox helper: {sandbox}")
        return False
    # Reject symlinks — chown/chmod must not follow an attacker-controlled link.
    if st is None:
        print(f"✗ Cannot stat Electron's Linux sandbox helper: {sandbox}")
        return False
    if not stat.S_ISREG(st.st_mode):
        print(f"✗ Electron's Linux sandbox helper is not a regular file: {sandbox}")
        return False

    if _sandbox_helper_is_setuid_root(st):
        return True

    if _desktop_linux_userns_sandbox_available():
        print("✓ Using Chromium's user-namespace sandbox (setuid helper not needed).")
        return True

    sudo = shutil.which("sudo")
    if not sudo:
        print("✗ Hermes Desktop requires sudo to configure Electron's Linux sandbox helper.")
        return False

    print("→ Configuring Electron Linux sandbox helper (sudo required)...")
    for command in ([sudo, "chown", "root:root", str(sandbox)], [sudo, "chmod", "4755", str(sandbox)]):
        if subprocess.run(command, check=False).returncode != 0:
            print(f"✗ Failed to configure Electron's Linux sandbox helper: {sandbox}")
            return False
    return True


def _desktop_linux_needs_disable_setuid_sandbox(packaged_executable: Path) -> bool:
    """True when a present, non-setuid ``chrome-sandbox`` would make Chromium abort with
    ``setuid_sandbox_host`` despite a working userns sandbox (call after the fixup's userns path)."""
    if sys.platform != "linux":
        return False
    _sandbox, st = _sandbox_helper_lstat(packaged_executable)
    return st is not None and stat.S_ISREG(st.st_mode) and not _sandbox_helper_is_setuid_root(st)


_LINUX_PASSWORD_STORES = frozenset({"gnome-libsecret", "kwallet", "kwallet5", "kwallet6", "basic"})

_GPU_FLAG_WORDS = {**dict.fromkeys(("1", "true", "yes", "on"), "1"), **dict.fromkeys(("0", "false", "no", "off"), "0")}


def _detect_linux_password_store() -> str | None:
    """Chromium password-store backend for this Linux session (KDE env → GNOME Keyring socket → D-Bus
    ping of org.freedesktop.secrets), or None. Chromium's own detection fails under the launcher
    env, and safeStorage then reports encryption unavailable."""
    kde_version = os.environ.get("KDE_SESSION_VERSION", "").strip()
    if kde_version:
        return {"6": "kwallet6", "5": "kwallet5"}.get(kde_version, "kwallet")
    if os.environ.get("KDE_FULL_SESSION"):
        return "kwallet"
    if os.environ.get("GNOME_KEYRING_CONTROL"):
        return "gnome-libsecret"
    with contextlib.suppress(Exception):
        result = subprocess.run(
            [
                "dbus-send", "--session", "--print-reply", "--reply-timeout=2000",
                "--dest=org.freedesktop.secrets",
                "/org/freedesktop/secrets",
                "org.freedesktop.DBus.Peer.Ping",
            ],
            capture_output=True,
            timeout=5)
        if result.returncode == 0:
            return "gnome-libsecret"
    return None


def _desktop_launch_options() -> tuple[list[str], str, str, str]:
    """``desktop.*`` launch options: ``(electron_flags, disable_gpu "auto"/"1"/"0", password_store,
    ozone_hint "auto"/"x11"/"wayland")``; unknown values and config errors yield "auto"/[] so a
    malformed config never blocks the launch."""
    flags: list[str] = []
    disable_gpu = password_store = ozone_hint = "auto"
    try:
        from hermes_cli.config import load_config
        desktop_cfg = (load_config() or {}).get("desktop") or {}
    except Exception:
        return flags, disable_gpu, password_store, ozone_hint

    raw_flags = desktop_cfg.get("electron_flags")
    if isinstance(raw_flags, str):
        flags = shlex.split(raw_flags, posix=(os.name != "nt"))
    elif isinstance(raw_flags, (list, tuple)):
        flags = [str(f) for f in raw_flags if str(f).strip()]

    def _choice(key: str, allowed) -> str:
        raw = desktop_cfg.get(key, "auto")
        low = raw.strip().lower() if isinstance(raw, str) else ""
        return low if low in allowed else "auto"

    raw_gpu = desktop_cfg.get("disable_gpu", "auto")
    if isinstance(raw_gpu, bool):
        disable_gpu = "1" if raw_gpu else "0"
    elif isinstance(raw_gpu, str):
        disable_gpu = _GPU_FLAG_WORDS.get(raw_gpu.strip().lower(), "auto")
    password_store = _choice("password_store", _LINUX_PASSWORD_STORES)
    ozone_hint = _choice("ozone_platform_hint", ("auto", "x11", "wayland"))
    return flags, disable_gpu, password_store, ozone_hint


def _register_linux_desktop_entry() -> None:
    """Install the XDG desktop entry for Hermes Desktop (Linux only, best-effort).

    ``Exec`` and ``Icon`` are absolute so the entry works outside a login shell.
    ``hermes uninstall --gui`` removes it.
    """
    from hermes_cli.main import PROJECT_ROOT
    try:
        from hermes_cli.linux_desktop_entry import install_desktop_entry, is_supported
        if not is_supported():
            return
        entry = install_desktop_entry(PROJECT_ROOT)
        if entry:
            print(f"✓ Desktop launcher entry installed: {entry}")
    except Exception as exc:  # never block a launch on launcher plumbing
        print(f"⚠ Could not install the desktop launcher entry: {exc}")


def _promote_staged_desktop_app(desktop_dir: Path, staging_dir: Path) -> Path:
    """Sign and verify the staged pack before replacing the live app."""
    staged_executable = _desktop_packaged_executable_in(staging_dir)
    # Locally-built apps are ad-hoc signed; make them relaunchable after an
    # in-place self-update. Signs the STAGED bundle so the live app is never
    # half-signed. No-op on non-macOS and on real-identity builds.
    _desktop_macos_relaunchable_fixup(desktop_dir, release_dir=staging_dir)

    # Windows integrity gate: never declare the rebuild a success on a
    # Hermes.exe Windows cannot load. Verified on the STAGED exe, so a failure
    # fails the build without replacing the live app.
    verified_executable, rolled_back = _ensure_desktop_exe_launchable(desktop_dir, staged_executable)
    if staged_executable is None or rolled_back or verified_executable is None:
        _discard_desktop_staging(staging_dir)
        if staged_executable is None:
            print(f"✗ Desktop build produced no launchable app in {staging_dir}")
        raise RuntimeError(f"Desktop build produced no launchable app. {_PREVIOUS_APP_KEPT}")
    packaged_executable = _swap_staged_desktop_app(desktop_dir, staging_dir)
    if packaged_executable is None:
        print(f"✗ Could not install the rebuilt desktop app into {desktop_dir / 'release'}")
        raise RuntimeError(f"Could not publish the desktop build. {_PREVIOUS_APP_KEPT}")
    return packaged_executable


def build_prepared_desktop(desktop_dir: Path, *, source_mode: bool, npm: str, env: dict,
                           icons: Path | None = None, explicit: bool = False) -> Optional[Path]:
    """Build prepared desktop sources, then publish the verified staged app."""
    build_label = "source build" if source_mode else "packaged app"
    print(f"→ Building desktop {build_label}...")
    build_env = dict(env)
    if _force_adhoc_macos_signing(build_env, source_mode=source_mode):
        print("  → No Developer ID configured; ad-hoc signing this local rebuild "
              "(CSC_IDENTITY_AUTO_DISCOVERY=false)")
    build_args = (["--icons", str(icons)] if icons else []) + ([] if explicit else ["--on-demand"])
    build_cmd = [npm, "run", "build", "--", *build_args]
    staging_dir = None if source_mode else _desktop_staging_dir(desktop_dir)
    if staging_dir is not None:
        # electron-builder packs in place; only the verified staging tree may
        # replace the running app, never a failed or incomplete build.

        stopped = _stop_desktop_processes_locking_build(desktop_dir)
        if stopped:
            print(f"  ⚠ Stopped running desktop app to free the build output (pid {', '.join(map(str, stopped))})")
    try:
        subprocess.run(build_cmd, cwd=desktop_dir, env=build_env, check=True)
        if staging_dir is not None:
            subprocess.run([npm, "run", "builder", "--", "--dir", "--publish", "never",
                            f"-c.directories.output={staging_dir}"], cwd=desktop_dir, env=build_env, check=True)
        packaged_executable = (
            _promote_staged_desktop_app(desktop_dir, staging_dir) if staging_dir is not None else None
        )
        return packaged_executable
    finally:
        if staging_dir is not None:
            _discard_desktop_staging(staging_dir)


_WSL_DXG_DEVICE = Path("/dev/dxg")
_WSL_D3D12_DRIVERS = (
    Path("/usr/lib/x86_64-linux-gnu/dri/d3d12_dri.so"),
    Path("/usr/lib/aarch64-linux-gnu/dri/d3d12_dri.so"),
    Path("/usr/lib64/dri/d3d12_dri.so"),
    Path("/usr/lib/dri/d3d12_dri.so"),
)
_MESA_DRIVER_OVERRIDES = ("GALLIUM_DRIVER", "MESA_LOADER_DRIVER_OVERRIDE", "LIBGL_ALWAYS_SOFTWARE", "LIBGL_DRIVERS_PATH")


def _prefer_wsl_d3d12(env: dict) -> None:
    """Under WSLg, /dev/dxg alone does not make Mesa pick the GPU: Chromium still lands on
    llvmpipe unless GALLIUM_DRIVER selects d3d12, and it must be set before Electron spawns
    its GPU process (setting it from JS is too late). Explicit Mesa choices win; hosts without
    the driver are left alone."""
    from hermes_constants import is_wsl
    if any(key in env for key in _MESA_DRIVER_OVERRIDES):
        return
    if is_wsl() and _WSL_DXG_DEVICE.exists() and any(driver.is_file() for driver in _WSL_D3D12_DRIVERS):
        env["GALLIUM_DRIVER"] = "d3d12"


def _desktop_launch_env(args: argparse.Namespace) -> tuple[dict, list[str]]:
    """Electron child env + config-supplied extra flags. ``desktop.*`` config is bridged to env vars
    Electron already reads; an explicit env var wins over config (and over keychain detection)."""
    from hermes_constants import with_hermes_node_path
    # with_hermes_node_path() copies os.environ when called with no arg.
    env = with_hermes_node_path()
    _prefer_wsl_d3d12(env)
    for attr, key in (
        ("fake_boot", "HERMES_DESKTOP_BOOT_FAKE"), ("ignore_existing", "HERMES_DESKTOP_IGNORE_EXISTING")):
        if getattr(args, attr, False):
            env[key] = "1"
    if getattr(args, "hermes_root", None):
        env["HERMES_DESKTOP_HERMES_ROOT"] = str(Path(args.hermes_root).expanduser().resolve())
    cwd = getattr(args, "cwd", None)
    env["HERMES_DESKTOP_CWD"] = str(Path(cwd).expanduser().resolve()) if cwd else os.getcwd()

    config_electron_flags, config_disable_gpu, config_password_store, config_ozone_hint = (
        _desktop_launch_options())
    if config_disable_gpu != "auto" and "HERMES_DESKTOP_DISABLE_GPU" not in os.environ:
        env["HERMES_DESKTOP_DISABLE_GPU"] = config_disable_gpu
    if config_ozone_hint != "auto" and "ELECTRON_OZONE_PLATFORM_HINT" not in os.environ:
        env["ELECTRON_OZONE_PLATFORM_HINT"] = config_ozone_hint

    # Without --password-store safeStorage.isEncryptionAvailable() is often
    # false and the desktop app refuses to persist remote gateway tokens.
    if sys.platform == "linux" and "HERMES_DESKTOP_PASSWORD_STORE" not in os.environ:
        password_store = (
            config_password_store if config_password_store != "auto" else _detect_linux_password_store()
        )
        if password_store:
            env["HERMES_DESKTOP_PASSWORD_STORE"] = password_store
    return env, config_electron_flags


def _check_desktop_skip_build(
    desktop_dir: Path, project_root: Path, *, source_mode: bool, packaged_executable: Optional[Path]
) -> None:
    """Validate the pre-built artifact ``--skip-build`` promised; exit with a hint when it's missing."""
    if source_mode:
        if not _desktop_dist_exists(desktop_dir):
            print(f"✗ --skip-build --source was passed but no desktop dist found at: {desktop_dir / 'dist'}")
            print("  Pre-build first:  cd apps/desktop && npm run build")
            print("  Or drop --skip-build to install dependencies and build automatically.")
            sys.exit(1)
        if not (_electron_dir(project_root) / "package.json").exists():
            print("✗ --skip-build --source requires existing desktop workspace dependencies.")
            print(f"  Install first:  cd {project_root} && npm ci")
            print("  Or drop --skip-build to install dependencies and build automatically.")
            sys.exit(1)
        print(f"→ Skipping desktop source build (--skip-build --source); using dist at {desktop_dir / 'dist'}")
    elif packaged_executable is None:
        print(f"✗ --skip-build was passed but no packaged desktop app was found at: {desktop_dir / 'release'}")
        print("  Pre-build first:  cd apps/desktop && npm run pack")
        print("  Or drop --skip-build to package automatically.")
        sys.exit(1)
    else:
        print(f"→ Skipping desktop package build (--skip-build); using {packaged_executable}")


def _packaged_desktop_launch_command(packaged_executable: Path) -> list[str]:
    """``[exe, *sandbox flags]`` after the Linux sandbox fixup; exits when the sandbox can't be configured."""
    launch_command = [str(packaged_executable)]
    if not _desktop_linux_sandbox_fixup(packaged_executable):
        if _desktop_linux_needs_no_sandbox() and _desktop_linux_sandbox_helper_is_regular_file(packaged_executable):
            print("⚠ Falling back to --no-sandbox because this Linux host restricts unprivileged user namespaces and the Electron sandbox helper could not be configured.")
            launch_command.append("--no-sandbox")
        else:
            sys.exit(1)
    elif _desktop_linux_needs_disable_setuid_sandbox(packaged_executable):
        launch_command.append("--disable-setuid-sandbox")
    return launch_command


def cmd_gui(args: argparse.Namespace):
    """Build and launch the native Electron desktop GUI."""
    from hermes_cli.main import PROJECT_ROOT
    from hermes_cli.source_build import prepare_source_dependencies, source_build_env
    desktop_dir = PROJECT_ROOT / "apps" / "desktop"
    # A bundled install IS the app: no source tree, no build, and the
    # launcher is a sibling of this payload rather than something we
    # produce. Every rung below assembles a checkout build, so the sealed
    # shape leaves here with the env it just built.
    from hermes_cli.steward import is_bundled_payload

    bundled = is_bundled_payload(PROJECT_ROOT)
    if not bundled and not (desktop_dir / "package.json").exists():
        print(f"Desktop GUI source not found at: {desktop_dir}")
        sys.exit(1)

    with contextlib.suppress(Exception):
        from hermes_logging import setup_logging as _setup_logging_gui
        _setup_logging_gui(mode="gui")

    env, config_electron_flags = _desktop_launch_env(args)

    source_mode = getattr(args, "source", False)
    skip_build = getattr(args, "skip_build", False)
    force_build = getattr(args, "force_build", False)

    # macOS-only one-shot: create a self-signed code-signing identity so TCC
    # grants survive rebuilds, then exit without building/launching.
    if getattr(args, "setup_tcc_identity", False):
        identity = getattr(args, "identity", None) or "Hermes Local Signing"
        sys.exit(0 if _desktop_macos_setup_tcc_identity(identity) else 1)

    if bundled:
        _launch_bundled_desktop(args, env, config_electron_flags)

    packaged_executable = _desktop_packaged_executable(desktop_dir)

    needs_build = not skip_build and (
        force_build or _desktop_build_needed(desktop_dir, PROJECT_ROOT, source_mode=source_mode)
    )
    npm = None
    try:
        if needs_build:
            build_env = source_build_env(env, explicit=force_build or getattr(args, "build_only", False))
            npm = shutil.which("npm", path=build_env["PATH"])
            env["PATH"] = build_env["PATH"]
        if skip_build:
            _check_desktop_skip_build(
                desktop_dir, PROJECT_ROOT, source_mode=source_mode, packaged_executable=packaged_executable
            )
        elif needs_build:
            prepare_source_dependencies(PROJECT_ROOT, ("ui-tui", "web", "apps/desktop"), env=build_env,
                                        explicit=force_build or getattr(args, "build_only", False))
            built = build_prepared_desktop(desktop_dir, source_mode=source_mode, npm=npm, env=build_env,
                                           explicit=force_build or getattr(args, "build_only", False))
            if not source_mode:
                packaged_executable = built
        else:
            build_label = "source build" if source_mode else "packaged app"
            print(f"✓ Desktop {build_label} is up to date (content stamp matches)")
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"✗ Desktop GUI build failed: {exc}")
        raise SystemExit(1) from exc

    # Best-effort and idempotent; a failure must never stop the app from launching.
    _register_linux_desktop_entry()

    # --build-only: produce the artifact but do NOT launch. The installer's
    # --update flow drives the rebuild headlessly and launches the desktop
    # itself (detached, after the old exe has exited); launching here would
    # block the installer. Verify the artifact exists so a silent "built
    # nothing" can't slip past.
    if getattr(args, "build_only", False):
        if source_mode:
            if not _desktop_dist_exists(desktop_dir):
                print(f"✗ --build-only --source produced no dist at: {desktop_dir / 'dist'}")
                sys.exit(1)
            print(f"✓ Desktop source build ready at {desktop_dir / 'dist'} (not launching; --build-only)")
        elif packaged_executable is None:
            print(f"✗ --build-only produced no launchable app at: {desktop_dir / 'release'}")
            print("  Expected an unpacked Electron app for the current OS.")
            sys.exit(1)
        else:
            print(f"✓ Desktop packaged app ready: {packaged_executable} (not launching; --build-only)")
        return

    if source_mode:
        print("→ Launching Hermes Desktop from source build...")
        # Launch only the prepared runtime. npm exec can provision a missing
        # Electron package, including when --skip-build was requested.
        electron = _electron_dir(PROJECT_ROOT)
        try:
            executable = electron / "dist" / (electron / "path.txt").read_text(encoding="utf-8").strip()
            if not executable.is_file():
                raise FileNotFoundError(executable)
        except OSError as exc:
            print(f"✗ Prepared Electron runtime is missing: {exc}")
            raise SystemExit(1) from exc
        launch_command = [str(executable), "."]
    else:
        if packaged_executable is None:
            print(f"✗ Desktop package build completed but no launchable app was found at: {desktop_dir / 'release'}")
            print("  Expected an unpacked Electron app for the current OS.")
            sys.exit(1)
        launch_command = _packaged_desktop_launch_command(packaged_executable)
        launch_command.extend(config_electron_flags)
    if getattr(args, "local", False):
        launch_command.append("--local")
    if not source_mode:
        print(f"→ Launching packaged Hermes Desktop: {' '.join(launch_command)}")
    launch_result = subprocess.run(launch_command, cwd=desktop_dir, env=env, check=False)
    sys.exit(launch_result.returncode)


def _launch_bundled_desktop(
    args: argparse.Namespace, env: dict, electron_flags: list[str]
) -> None:
    """Start the desktop app this CLI ships inside, then exit.

    A bundled install has no source tree to build: the app is a signed,
    read-only artifact and this Python is a passenger in its resources.
    So the whole build ladder below is skipped and the launcher is started
    DETACHED — the user ran a CLI command, and the app must outlive the
    terminal it was typed into. The app's own single-instance lock turns a
    second run into "focus the running window".

    Never returns.
    """
    from hermes_cli.bundled_app import NotBundledApp, launch_detached, resolve_bundle_layout
    from hermes_cli.main import PROJECT_ROOT

    refused = [
        flag
        for flag, name in (
            ("--source", "source"),
            ("--build-only", "build_only"),
            ("--force-build", "force_build"),
        )
        if getattr(args, name, False)
    ]
    if refused:
        print(f"✗ {', '.join(refused)} cannot apply to a bundled Hermes install.")
        print("  This app ships prebuilt and has no desktop source tree to build.")
        sys.exit(2)

    try:
        layout = resolve_bundle_layout(PROJECT_ROOT)
    except NotBundledApp as exc:
        # The stamp says bundled, so a tree that is not one is a damaged or
        # mispackaged install. Report it — degrading to the build ladder
        # would run npm inside the app's own resources.
        print(f"✗ This Hermes is stamped as a bundled desktop install, but {exc}.")
        print("  The install is damaged — reinstall Hermes from the website.")
        sys.exit(1)

    if layout.launcher is None:
        print(f"✗ Found no Hermes Desktop launcher in {layout.app_root}.")
        print("  The install is damaged — reinstall Hermes from the website.")
        sys.exit(1)

    launch_command = [str(layout.launcher)]
    if not _desktop_linux_sandbox_fixup(layout.launcher):
        if _desktop_linux_needs_no_sandbox() and _desktop_linux_sandbox_helper_is_regular_file(layout.launcher):
            print("⚠ Falling back to --no-sandbox because this Linux host restricts unprivileged user namespaces and the Electron sandbox helper could not be configured.")
            launch_command.append("--no-sandbox")
        else:
            sys.exit(1)

    launch_command.extend(electron_flags)
    pid = launch_detached(launch_command, env=env, cwd=layout.app_root)
    print(f"→ Launched Hermes Desktop: {' '.join(launch_command)} (pid {pid})")
    sys.exit(0)


