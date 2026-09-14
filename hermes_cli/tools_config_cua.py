"""cua-driver installer, lock hygiene and pip-install helper for `hermes tools` /
`hermes computer-use install`."""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from hermes_cli.cli_output import (
    print_info as _print_info, print_success as _print_success, print_warning as _print_warning)

logger = logging.getLogger("hermes_cli.tools_config")

# One upstream-installer run must outlive the installer's own stale-lock recovery (_install-rust.sh
# force-releases a dead holder's lock only after LOCK_STALE_AFTER_SECONDS=600; a shorter timeout
# kills every run before that fires — a permanent wedge). 660s = 600s + 60s headroom.
# With a shorter Python-side timeout, a stale lock means every run gets killed before the installer's
# recovery can fire — a permanent "always times out" wedge (issue #58762). 660s = 600s lock window + 60s
# headroom for the actual download/swap.
_CUA_INSTALLER_TIMEOUT = 660
# Bounded pipe drain after a timeout kill: the kill is best-effort (_reap_after_timeout), and a
# surviving descendant holding the inherited stdout would otherwise block the read on an EOF that
# never comes. A successful kill closes the pipe at once, so this costs nothing.
# Grace period for draining the installer's pipes after a timeout kill. A successful kill closes the pipe
# immediately, so this costs nothing in the normal case; it only caps how long a failed one can stall the
# update. See #87703.
_CUA_INSTALLER_DRAIN_GRACE = 15
# Quiet ``hermes update`` refreshes stay bounded even when upstream waits on Read-Host / a consent
# prompt (explicit ``install --upgrade`` keeps the full ceiling); safe because the lock/network
# preflights make a legitimate long wait impossible here.
_CUA_BACKGROUND_UPDATE_TIMEOUT = 120
# Upstream's LOCK_STALE_AFTER_SECONDS: the pre-clear never yanks a lock a live install still holds.
_CUA_LOCK_STALE_AFTER = 600

_CUA_INSTALL_PS1_URL = (
    "https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.ps1")
_CUA_INSTALL_SH_URL = (
    "https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh")
_CUA_MANUAL_README = "https://github.com/trycua/cua/blob/main/libs/cua-driver/README.md"
_UPGRADE_CMD = "hermes computer-use install --upgrade"


def _run_text(cmd: list, *, timeout, capture_output: bool = True,
              **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` with the utf-8/replace text decoding every helper here uses."""
    return subprocess.run(cmd, capture_output=capture_output, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, **kwargs)


def _fail(message: str, *hints: str, warn: bool = True) -> bool:
    """Print ``message`` (warning, or info when ``warn=False``) plus info hints; returns False."""
    (_print_warning if warn else _print_info)(message)
    for hint in hints:
        _print_info(hint)
    return False


def _print_output_tail(result: subprocess.CompletedProcess, printer=None) -> None:
    """Echo the last three lines of a failed command's stderr (or stdout) as indented info.
    ``printer`` lets another module route through its own (test-patchable) ``_print_info``."""
    for line in (result.stderr or result.stdout or "").strip().splitlines()[-3:]:
        (printer or _print_info)(f"      {line[:200]}")


def _post_setup_no_window_flags(*, streams_to_console: bool = False) -> int:
    """Win32 creationflags that stop post-setup children flashing a console (0 on POSIX).
    CREATE_NO_WINDOW hides console grandchildren (npm, pip, powershell) while keeping stdio
    inheritable (unlike DETACHED_PROCESS). ``streams_to_console`` children are only hidden when our
    own stdout is not a console, so live installer output is never swallowed."""
    from hermes_cli._subprocess_compat import windows_hide_flags
    flags = windows_hide_flags()
    try:
        if flags and streams_to_console and sys.stdout is not None and sys.stdout.isatty():
            return 0
    except Exception:
        pass
    return flags or 0


def _cua_driver_cmd() -> str:
    """Return the configured cua-driver override, or the bare default name."""
    return os.environ.get("HERMES_CUA_DRIVER_CMD", "").strip() or "cua-driver"


def _cua_version_summary(raw: str, *, limit: int = 120) -> str:
    """First non-empty line of ``--version`` output, bounded (an override may print a banner)."""
    return next((line.strip()[:limit] for line in (raw or "").splitlines() if line.strip()), "")


def _resolved_cua_driver_cmd() -> Optional[str]:
    """Resolve cua-driver exactly as the runtime and Desktop status do."""
    from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd
    return resolve_cua_driver_cmd()


def _cua_driver_env() -> dict:
    """cua-driver child env with the Hermes telemetry policy applied; falls back to the current
    environment if the helper can't be imported, so install/status never break."""
    try:
        from tools.computer_use.cua_backend import cua_driver_child_env
        return cua_driver_child_env()
    except Exception:
        return dict(os.environ)


_CUA_DRIVER_CONTRACT_CACHE: dict = {}


def _cua_driver_contract_status(binary: Optional[str] = None) -> dict:
    """Inspect whether an installed driver supports Hermes' runtime contract (30s cache keyed on the
    binary's path/mtime/size fingerprint)."""
    from tools.computer_use.cua_backend_driver import cua_driver_runtime_contract_status
    resolved = binary or _resolved_cua_driver_cmd()
    if not resolved:
        return cua_driver_runtime_contract_status(None)
    try:
        stat = os.stat(resolved)
        fingerprint = (resolved, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return cua_driver_runtime_contract_status(resolved)
    now = time.monotonic()
    cache = _CUA_DRIVER_CONTRACT_CACHE
    if cache.get("fingerprint") == fingerprint and now - cache.get("checked_at", 0.0) < 30.0:
        return dict(cache["state"])
    state = cua_driver_runtime_contract_status(resolved)
    cache.update(fingerprint=fingerprint, checked_at=now, state=dict(state))
    return state


def _cua_driver_install_ready() -> bool:
    """Return whether an existing driver needs no install-time repair."""
    return bool(_cua_driver_contract_status().get("ready")) and (
        sys.platform != "win32" or _cua_driver_autostart_registered_windows())


def _pip_install(args: List[str], *, timeout: int = 300, capture_output: bool = True):
    """Install Python packages: ``uv pip install`` (needs no pip in the venv), then ``python -m
    pip``, then ``ensurepip --upgrade`` + retry — the Windows installer creates the venv via
    ``uv venv``, which does NOT seed pip, so bare ``-m pip`` failed on fresh installs."""
    venv_root = Path(sys.executable).parent.parent
    install_flags = _post_setup_no_window_flags(streams_to_console=not capture_output)

    # Managed uv first: $HERMES_HOME/bin is never on PATH, so a bare which() misses the uv Hermes
    # installed; ensure_uv() (not a pure lookup) because installing uv is in scope during setup.
    from hermes_cli.managed_uv import ensure_uv
    uv_bin = ensure_uv()
    try:  # a failed uv run falls through to pip — it may have failed for a reason pip can handle
        result = uv_bin and _run_text([uv_bin, "pip", "install", *args], timeout=timeout,
                                      capture_output=capture_output, creationflags=install_flags,
                                      env={**os.environ, "VIRTUAL_ENV": str(venv_root)})
        if result and result.returncode == 0:
            return result
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    pip_cmd = [sys.executable, "-m", "pip"]
    try:
        # Probe for pip; bootstrap via ensurepip if missing (uv venv lacks it).
        probe = _run_text(pip_cmd + ["--version"], timeout=15,
                          creationflags=_post_setup_no_window_flags())
        if probe.returncode != 0:
            raise FileNotFoundError("pip not in venv")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        try:
            _run_text([sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                      timeout=120, check=True, creationflags=_post_setup_no_window_flags())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Synthesize a result so callers see a clean failure path.
            return subprocess.CompletedProcess(
                pip_cmd, returncode=1, stdout="",
                stderr=f"pip not available and ensurepip failed: {e}")
    return _run_text(pip_cmd + ["install", *args], capture_output=capture_output, timeout=timeout,
                     creationflags=install_flags)


# No pre-install release/asset probe: cua-driver-rs releases are all prereleases, which GitHub's
# `/releases/latest` skips (zero binary assets → every non-arm64 host skipped the install), and
# re-implementing upstream's tag resolution here would drift. Fresh installs run install.sh directly
# (it errors clean on a missing-arch asset); upgrades ask the binary via cua_driver_update_check().


def _cua_install_target_writable() -> bool:
    """Return whether the upstream installer can write its app bundle target."""
    if sys.platform != "darwin":
        return True
    try:
        return not os.path.isdir("/Applications") or os.access("/Applications", os.W_OK)
    except Exception:
        return True


def _cua_driver_version(binary: str) -> Optional[str]:
    """``<binary> --version`` stdout (possibly ""), or None when the probe itself fails."""
    try:
        return _run_text([binary, "--version"], timeout=5, env=_cua_driver_env(),
                         creationflags=_post_setup_no_window_flags()).stdout.strip()
    except Exception:
        return None


def _confirmed_update_check(driver_cmd: str, require_confirmed_update: bool) -> tuple:
    """Ask the installed driver whether a newer release exists; returns ``(proceed, pin_version)``.
    ``proceed=False`` = stop with success (already latest, or indeterminate under
    ``require_confirmed_update``). An old driver (no check-update verb) or offline check yields
    None: `hermes update` then keeps the installed version — an indeterminate check must never
    cost a multi-minute silent reinstall on every update — while explicit `install --upgrade`
    falls through."""
    try:
        from tools.computer_use.cua_backend_driver import cua_driver_update_check
        _state = cua_driver_update_check()
    except Exception:
        _state = None
    if _state is None:
        if require_confirmed_update:
            _fail(f"    Could not confirm a newer {driver_cmd} release (offline, rate-limited, or "
                  "driver too old to check); keeping the installed version.",
                  f"    Force a refresh with: {_UPGRADE_CMD}", warn=False)
        return not require_confirmed_update, None
    if not _state.get("update_available"):
        _print_success(f"    {driver_cmd} is already on the latest release "
                       f"({_state.get('current_version') or 'unknown'}).")
        return False, None
    # Windows routine upgrades run unattended-safe (stdin closed, version pinned, ceiling
    # _CUA_BACKGROUND_UPDATE_TIMEOUT, preflights skip in seconds); only contract repairs and fresh
    # installs stay interactive-only, where upstream needs a human (autostart elevation/SmartScreen).
    # Pin to the release check-update confirmed: `latest_version` comes from the GitHub Releases API
    # so its assets exist, unlike the installer's baked version on `main` (bumped before assets are
    # published → 404s unpinned). Malformed values are ignored → unpinned fallback.
    _latest = str(_state.get("latest_version") or "").strip().lstrip("vV")
    return True, (_latest if re.fullmatch(r"\d+(\.\d+)*", _latest) else None)


def _report_repair_or_upgrade(ok: bool, *, repair_existing: bool, binary, before: str,
                              driver_cmd: str) -> bool:
    """Post-installer verdict: a repair must leave a usable contract; upgrades show before/after."""
    if ok and repair_existing:
        repaired = _cua_driver_contract_status()
        if not repaired.get("ready"):
            return _fail("    cua-driver was reinstalled, but its runtime contract is still "
                         f"unusable: {repaired.get('reason') or 'unknown error'}.",
                         "    Run: hermes computer-use doctor")
    if ok and before:
        after = _cua_driver_version(binary)
        if after and after != before:
            _print_success(f"    {driver_cmd} upgraded: {before} → {after}")
        elif after:
            _print_info(f"    {driver_cmd} up to date: {after}")
    return ok


def install_cua_driver(upgrade: bool = False, require_confirmed_update: bool = False,
                       show_installer_progress: bool = True) -> bool:
    """Install or refresh the cua-driver binary used by Computer Use.
    Re-running the upstream installer (always the latest release tag) is the canonical upgrade.
    ``upgrade=False`` (toolset enable flow) keeps a compatible installation, repairs an
    old/incomplete one and installs when missing; ``upgrade=True`` always refreshes."""
    system = platform.system()
    if system not in ("Darwin", "Windows", "Linux"):
        if not upgrade:  # silent under `hermes update`, which calls this for every user
            _print_warning(
                "    Computer Use (cua-driver) is unsupported on this platform; skipping.")
        return False
    is_windows, is_linux = system == "Windows", system == "Linux"
    # install.ps1 is fetched via PowerShell's `irm`; macOS/Linux use curl | bash.
    fetch_tool = "powershell" if is_windows else "curl"
    driver_cmd, binary = _cua_driver_cmd(), _resolved_cua_driver_cmd()
    # An explicit override is authoritative even when broken: installing the standard driver
    # cannot repair the configured path and would mutate an unrelated installation.
    override = os.environ.get("HERMES_CUA_DRIVER_CMD", "").strip()
    if override and not binary:
        return _fail(f"    HERMES_CUA_DRIVER_CMD does not resolve to an executable: {override}",
                     "    Fix or unset the override before running computer-use install.")

    # Not installed → fresh install path (only when caller asked for it).
    if not binary and not upgrade:
        if not _cua_install_target_writable():
            return _fail("    /Applications is not writable; skipping cua-driver install.",
                         "    Run from an admin account or install cua-driver manually.",
                         warn=False)
        if not shutil.which(fetch_tool):
            return _fail(f"    {fetch_tool} not found — install manually:",
                         f"      {_CUA_MANUAL_README}")
        return _run_cua_driver_installer(label="Installing")

    # A driver failing Hermes' runtime contract (version floor, missing manifest verbs) is repaired
    # regardless of mode. Hermes' minimum requirement IS the confirmation an upgrade is needed, so
    # this path must not defer to the driver's `check-update` verb — a cached/indeterminate "no
    # update" answer would pin users on an unusable driver forever.
    contract = _cua_driver_contract_status(binary) if binary else None
    repair_existing = bool(binary and contract and not contract.get("ready"))

    # Compatible existing install: no download, just the host-specific setup upstream normally owns.
    if binary and not upgrade and not repair_existing:
        version = _cua_driver_version(binary)
        suffix = "" if version is None else f": {version or 'unknown version'}"
        _print_success(f"    {driver_cmd} already installed{suffix}.")
        if is_windows and not _repair_cua_driver_autostart_windows(binary, verbose=False):
            return _fail("    cua-driver is compatible, but Windows autostart repair failed.")
        _print_cua_platform_notes(is_windows, is_linux, fresh_install=False)
        return True
    if repair_existing:
        _print_warning(f"    Found cua-driver {contract.get('version') or 'unknown version'}, but "
                       "Hermes cannot use its current runtime contract: "
                       f"{contract.get('reason') or 'required runtime features are missing'}.")
        if override:
            return _fail("    Update the binary selected by HERMES_CUA_DRIVER_CMD, or unset the "
                         f"override and run: {_UPGRADE_CMD}", warn=False)
        if is_windows and require_confirmed_update:
            return _fail("    Automatic Windows updates cannot safely run cua-driver's interactive "
                         "repair installer.",
                         f"    Repair it from an interactive terminal with: {_UPGRADE_CMD}",
                         warn=False)
        _print_info("    Repairing it with the current upstream installer.")

    # upgrade=True path — refresh to the latest upstream release.
    if not _cua_install_target_writable():
        _print_info("    /Applications is not writable; skipping cua-driver refresh.")
        _print_info(f"    Run `{_UPGRADE_CMD}` from an admin account to update it.")
        return bool(binary)
    if not shutil.which(fetch_tool):
        _print_warning(f"    {fetch_tool} not found — cannot refresh cua-driver.")
        return bool(binary)
    confirmed_version = None
    if binary and not repair_existing:
        proceed, confirmed_version = _confirmed_update_check(driver_cmd, require_confirmed_update)
        if not proceed:
            return True
    if is_windows and require_confirmed_update and not binary:
        # Missing binary (enabled but never installed, or wiped by a failed install): an automatic
        # Windows update must never launch install.ps1, which can demand console/UAC consent the
        # hidden updater cannot provide.
        return _fail("    cua-driver is not installed; automatic Windows updates cannot safely run "
                     "its interactive installer.",
                     f"    Install it from an interactive terminal with: {_UPGRADE_CMD}",
                     warn=False)
    before = (_cua_driver_version(binary) or "") if binary else ""  # best-effort before/after
    ok = _run_cua_driver_installer(
        label="Repairing" if repair_existing else "Refreshing", verbose=False,
        pin_version=confirmed_version, show_progress=show_installer_progress,
        installer_timeout=_CUA_BACKGROUND_UPDATE_TIMEOUT if require_confirmed_update else None)
    return _report_repair_or_upgrade(ok, repair_existing=repair_existing, binary=binary,
                                     before=before, driver_cmd=driver_cmd)


def _cua_install_home() -> "Path":
    """Package home shared by the upstream POSIX and Windows installers."""
    return Path(os.environ.get("CUA_DRIVER_RS_HOME") or str(Path.home() / ".cua-driver"))


def _cua_install_lock_dir() -> "Path":
    """Path of the upstream installer's concurrent-install lock dir."""
    return _cua_install_home() / "packages" / ".install.lock.d"


def _cua_windows_install_lock_file() -> "Path":
    """Path of install.ps1's FileShare::None lock file."""
    return _cua_install_home() / "install.lock"


def _clear_stale_windows_cua_install_lock() -> None:
    """Delete install.ps1's lock file only when no process still holds it.
    install.ps1 locks with ``FileShare::None``; mirror it with a zero-share ``CreateFileW`` probe
    and ``FILE_FLAG_DELETE_ON_CLOSE`` so an unlocked leftover is removed atomically, with no window
    in which a new installer could acquire the file between probe and delete."""
    lock_file = _cua_windows_install_lock_file()
    try:
        if not lock_file.is_file():
            return
        import ctypes as _ctypes
        from ctypes import wintypes as _wintypes
        # Win32 constants used by install.ps1's FileShare::None equivalent.
        delete_access, generic_read, generic_write = 0x00010000, 0x80000000, 0x40000000
        open_existing, file_attribute_normal, file_flag_delete_on_close = 3, 0x00000080, 0x04000000
        kernel32 = _ctypes.WinDLL("kernel32", use_last_error=True)
        create_file, close_handle = kernel32.CreateFileW, kernel32.CloseHandle
        dword = _wintypes.DWORD
        create_file.argtypes = [_wintypes.LPCWSTR, dword, dword, _wintypes.LPVOID, dword, dword,
                                _wintypes.HANDLE]
        create_file.restype = _wintypes.HANDLE
        close_handle.argtypes, close_handle.restype = [_wintypes.HANDLE], _wintypes.BOOL
        handle = create_file(
            str(lock_file), generic_read | generic_write | delete_access, 0,  # 0 = FileShare::None
            None, open_existing, file_attribute_normal | file_flag_delete_on_close, None)
        if handle == _wintypes.HANDLE(-1).value:
            logger.debug("Windows cua install lock at %s is still held or cannot be removed "
                         "(winerror %s)", lock_file, _ctypes.get_last_error())
            return
        if not close_handle(handle):
            logger.debug("could not close Windows cua install lock probe at %s (winerror %s)",
                         lock_file, _ctypes.get_last_error())
            return
        if lock_file.exists():
            logger.debug("Windows cua install lock probe succeeded but %s remains", lock_file)
            return
        logger.info("Cleared stale Windows cua-driver install lock at %s", lock_file)
        _print_info(f"    Cleared stale cua-driver install lock ({lock_file}).")
    except Exception as e:
        logger.debug("stale Windows cua install lock check failed: %s", e)


def _clear_stale_cua_install_lock() -> None:
    """Best-effort: remove a stale installer lock left by a dead holder.
    POSIX stamps the holder pid into ``~/.cua-driver/packages/.install.lock.d/info``; Windows holds
    ``~/.cua-driver/install.lock`` open with ``FileShare::None``. Clear either artifact up front
    only when its platform-specific liveness check proves that no install still holds it."""
    if sys.platform == "win32":
        _clear_stale_windows_cua_install_lock()
        return
    lock_dir = _cua_install_lock_dir()
    try:
        if not lock_dir.is_dir():
            return
        try:
            lines = (lock_dir / "info").read_text(encoding="utf-8", errors="replace").splitlines()
            holder_pid = next((int(line.split("=", 1)[1].strip()) for line in lines
                               if line.startswith("pid=")), None)
        except (OSError, ValueError):
            holder_pid = None
        if holder_pid is not None:
            try:
                os.kill(holder_pid, 0)  # windows-footgun: ok — function early-returns on win32
                return  # holder alive → concurrent install running
            except ProcessLookupError:
                pass  # dead holder → stale, clear below
            except PermissionError:
                return  # alive but owned by someone else — treat as live
        else:
            # No readable pid: only clear if old enough that upstream itself would reclaim it.
            try:
                if time.time() - lock_dir.stat().st_mtime < _CUA_LOCK_STALE_AFTER:
                    return
            except OSError:
                return
        shutil.rmtree(lock_dir, ignore_errors=True)
        logger.info("Cleared stale cua-driver install lock at %s", lock_dir)
        _print_info(f"    Cleared stale cua-driver install lock ({lock_dir}).")
    except Exception as e:
        logger.debug("stale cua install lock check failed: %s", e)


def _cua_install_lock_held() -> bool:
    """True when the upstream installer's lock is held by a LIVE process.
    Called after ``_clear_stale_cua_install_lock()``: anything provably stale is already gone, so a
    surviving lock artifact means a concurrent (or orphaned-but-alive) install owns it.

    Upstream waits up to ``LOCK_STALE_AFTER_SECONDS=600`` on a held lock before probing — unattended
    refreshes must not eat that wait (the 11-minute hang class, 87703): they skip instead. Best-effort:
    unreadable state reports not-held so a probe failure can never block an install. See #87703.
    """
    try:
        if sys.platform != "win32":
            return _cua_install_lock_dir().is_dir()
        lock_file = _cua_windows_install_lock_file()
        if not lock_file.is_file():
            return False
        # install.ps1 holds the file with FileShare::None — any open fails with a sharing
        # violation while held. Surviving the stale-clear = held; confirm with an open probe.
        try:
            with open(lock_file, "r+b"):
                return False  # opened fine → not held (racy leftover)
        except OSError:  # sharing violation surfaces as PermissionError
            return True
    except Exception as e:
        logger.debug("cua install lock probe failed: %s", e)
        return False


def _cua_release_endpoint_reachable(timeout: float = 5.0) -> bool:
    """Fast probe: can we reach GitHub's release download host at all?
    When github.com is down the installer dies slowly inside its own retries and eats the whole
    unattended ceiling; a 5s HEAD decides in seconds. Only a connection-level failure counts as
    unreachable — any HTTP response (even 4xx/5xx) proves the path works."""
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request("https://github.com/trycua/cua/releases", method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True  # server answered → reachable
    except Exception as e:
        logger.debug("cua release endpoint probe failed: %s", e)
        return False


def _ps_single_quote(value: str) -> str:
    """Return a PowerShell single-quoted string literal."""
    return "'" + value.replace("'", "''") + "'"


def _cua_driver_autostart_registered_windows() -> bool:
    """Return whether the Windows cua-driver scheduled task is registered."""
    if sys.platform != "win32":
        return False
    try:
        return subprocess.run(["schtasks.exe", "/Query", "/TN", "cua-driver-serve"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=10).returncode == 0
    except Exception:
        return False


def _repair_cua_driver_autostart_windows(driver_cmd: str, *, verbose: bool) -> bool:
    """Best-effort repair for Windows installer autostart quoting failures.
    Older install.ps1 builds interpolated the binary path into a PowerShell command string, which
    split at the first space. If the scheduled task is missing, retry via Start-Process's
    structured ``-FilePath`` / ``-ArgumentList`` parameters instead."""
    if sys.platform != "win32" or _cua_driver_autostart_registered_windows():
        return True
    binary = shutil.which(driver_cmd)
    if not binary:
        return False
    ps = shutil.which("powershell") or shutil.which("powershell.exe") or "powershell"
    ps_cmd = (f"$exe = {_ps_single_quote(binary)}; "
              "$proc = Start-Process -FilePath $exe -ArgumentList @('autostart','enable') "
              "-Verb RunAs -Wait -PassThru -ErrorAction Stop; exit $proc.ExitCode")
    _print_info("    Registering cua-driver auto-start..." if verbose
                else "    Repairing cua-driver auto-start registration...")
    try:
        result = _run_text([ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                           timeout=300, env=_cua_driver_env())
    except subprocess.TimeoutExpired:
        return _fail("    cua-driver autostart registration timed out.")
    except Exception as exc:
        return _fail(f"    cua-driver autostart registration failed: {exc}")
    if result.returncode == 0:
        return True
    _print_warning("    cua-driver autostart registration failed.")
    _print_output_tail(result)
    _print_info("    From an elevated shell, run: cua-driver autostart enable")
    return False


def _remove_quietly(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


def _print_cua_platform_notes(is_windows: bool, is_linux: bool, *, fresh_install: bool) -> None:
    """Host-specific follow-up notes after an install or a compatible-install check."""
    if is_windows:
        _print_info("    cua-driver may spawn a UIAccess worker (cua-driver-uia.exe);")
        _print_info("    Windows/SmartScreen may prompt the first time it runs.")
    elif is_linux:
        _print_warning("    Linux support is alpha.")
    else:
        _print_info("    IMPORTANT — grant macOS permissions now:" if fresh_install
                    else "    Grant macOS permissions if not done yet:")
        _print_info("      System Settings > Privacy & Security > Accessibility")
        _print_info("      System Settings > Privacy & Security > Screen Recording")
        if fresh_install:
            _print_info("    Both must allow the terminal / Hermes process.")


def _kill_installer_tree(proc, *, is_windows: bool) -> None:
    """Kill the installer and its descendants (best-effort)."""
    import signal as _signal
    try:
        if not is_windows:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)  # windows-footgun: ok — POSIX only
            return
        # PowerShell may leave download/install helpers alive after its direct process is killed;
        # they inherit stdout and can keep communicate() and install.lock wedged → kill leaf-up.
        import psutil as _psutil
        try:
            parent = _psutil.Process(proc.pid)
            descendants = parent.children(recursive=True)
        except _psutil.NoSuchProcess:
            return
        except _psutil.Error as e:
            logger.debug("could not enumerate cua-driver installer tree for pid %s: %s",
                         proc.pid, e)
            proc.kill()
            return
        for target, pid in [(c, c.pid) for c in reversed(descendants)] + [(parent, proc.pid)]:
            try:
                target.kill()
            except _psutil.NoSuchProcess:
                pass
            except _psutil.Error as e:
                what = "parent" if target is parent else "child"
                logger.debug("could not kill cua-driver installer %s pid %s: %s", what, pid, e)
                if target is parent:
                    proc.kill()
    except (OSError, ProcessLookupError):
        proc.kill()


def _reap_after_timeout(proc, *, is_windows: bool) -> None:
    """Kill the installer tree, then drain its pipes under a deadline.
    An unbounded drain blocks on an EOF that only arrives when someone kills a surviving descendant
    by hand, so ``_CUA_INSTALLER_TIMEOUT`` would stop bounding anything.

    Bound the drain instead: a kill that landed closes the pipe at once, and one that did not costs
    ``_CUA_INSTALLER_DRAIN_GRACE`` rather than forever. The caller re-raises the original ``TimeoutExpired``
    either way, so the manual re-run hint still prints and the update unwinds. Losing the tail of a
    timed-out installer's log is the cheaper half of that trade. See #87703.
    """
    _kill_installer_tree(proc, is_windows=is_windows)
    try:
        drained_out, _ = proc.communicate(timeout=_CUA_INSTALLER_DRAIN_GRACE)
        # Partial output names WHERE the installer was stuck (lock wait, consent prompt, download).
        # Diagnosability (#87703 post-mortem): the partial output names WHERE the installer was stuck (lock
        # wait, consent prompt, download) — before this, the answer died with the process and the timeout
        # line was unactionable.
        if drained_out:
            logger.warning("cua-driver installer timed out; last output before kill:\n%s",
                           drained_out[-2000:])
    except subprocess.TimeoutExpired:
        # Deliberately not closing proc.stdout: communicate()'s reader threads are still blocked on
        # that handle and closing it underneath them races; they are daemon threads.
        logger.debug("cua-driver installer pipes still open %ss after the kill — "
                     "abandoning the drain, a surviving descendant holds the inherited handle",
                     _CUA_INSTALLER_DRAIN_GRACE)
    except (OSError, ValueError) as e:
        logger.debug("cua-driver installer drain failed: %s", e)


def _cua_installer_command(is_windows: bool):
    """Return ``(install_cmd, manual_hint, script_path)``; all None if the POSIX download fails."""
    if is_windows:
        ps_oneliner = f"irm {_CUA_INSTALL_PS1_URL} | iex"  # mirrors cua_driver_install_hint()
        return (["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_oneliner],
                f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{ps_oneliner}"', None)

    # Download-then-exec instead of `bash -c "$(curl …)"`: no shell=True, no command substitution,
    # and the script lands in a mkstemp file (unpredictable name, 0600) rather than a fixed /tmp
    # path — avoiding both shell injection and a symlink/TOCTOU race. The manual hint stays the
    # upstream one-liner.
    import tempfile as _tempfile
    manual_hint = f'/bin/bash -c "$(curl -fsSL {_CUA_INSTALL_SH_URL})"'
    fd, script_path = _tempfile.mkstemp(prefix="cua-driver-install-", suffix=".sh")
    os.close(fd)
    try:
        dl = _run_text(["curl", "-fsSL", "-o", script_path, _CUA_INSTALL_SH_URL], timeout=120)
        failure = None if dl.returncode == 0 else (dl.stderr or "").strip()[:200]
    except (subprocess.TimeoutExpired, OSError) as e:
        failure = str(e)
    if failure is not None:
        _print_warning(f"    cua-driver installer download failed: {failure}")
        _remove_quietly(script_path)
        return None, None, None
    return ["/bin/bash", script_path], manual_hint, script_path


def _unattended_installer_preflight(install_cmd: list, is_windows: bool):
    """Fail FAST on the two conditions that otherwise consume the whole unattended ceiling:
    (1) install lock held by a live process — upstream would poll it for up to
    LOCK_STALE_AFTER_SECONDS=600 before probing the holder (the 11-minute silent hang class);
    (2) release host unreachable — the installer dies slowly inside its own retries; a 5s HEAD
    answers. Returns the (possibly rewritten) install command, or None to skip this refresh.
    Explicit `install --upgrade` runs never come here and keep upstream's full lock-recovery."""
    if _cua_install_lock_held():
        _fail("    Another cua-driver install is in progress (upstream install lock is held) — "
              "skipping this refresh.",
              f"    If no install is really running, retry with: {_UPGRADE_CMD}", warn=False)
        return None
    if not _cua_release_endpoint_reachable():
        _print_info("    github.com is unreachable — skipping cua-driver refresh "
                    "(will retry on the next update).")
        return None
    if is_windows:
        # -NoAutoStart skips Register-CuaDriverAutostart — the ONLY branch of install.ps1 that
        # self-elevates (UAC). Cost: an existing cua-driver-serve task keeps pointing at the
        # previous binary until the next interactive upgrade. Scriptblock invocation (not `| iex`)
        # is what lets us pass the parameter.
        install_cmd = [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
            f"$sc = irm {_CUA_INSTALL_PS1_URL}; & ([scriptblock]::Create($sc)) -NoAutoStart"]
    return install_cmd


def _installer_popen_kwargs(is_windows: bool, verbose: bool, env: dict) -> dict:
    """Popen kwargs for the upstream installer.
    POSIX: own process group so a timeout kill takes out the whole `curl | bash` pipeline (and the
    exec'd _install-rust.sh), not just the outer shell — surviving grandchildren would keep
    holding the install lock and wedge every later run. Non-verbose (`hermes update` refresh):
    capture the chatty "Next steps" wall and log it so a failure stays debuggable; verbose
    interactive installs stream live."""
    kwargs: dict = {"shell": False, "env": env}
    if not is_windows:
        kwargs["start_new_session"] = True
    if verbose:
        kwargs["creationflags"] = _post_setup_no_window_flags(streams_to_console=True)
    else:
        kwargs.update(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, encoding="utf-8", errors="replace",
                      creationflags=_post_setup_no_window_flags())
    return kwargs


def _record_installer_output(out: str, returncode: int) -> None:
    """Keep a captured (non-verbose) installer transcript without echoing it to the terminal.
    During `hermes update`, sys.stdout is the mirroring _UpdateOutputStream whose `_log` handle is
    ~/.hermes/logs/update.log — write straight to it so the full output is kept (success AND
    failure)."""
    _update_log = getattr(sys.stdout, "_log", None)
    if _update_log is not None:
        try:
            _update_log.write("\n--- cua-driver installer output ---\n" + out + "\n")
            _update_log.flush()
        except Exception:
            pass
    if returncode != 0:
        logger.debug("cua-driver installer output:\n%s", out)


def _run_cua_driver_installer(label: str = "Installing", verbose: bool = True,
                              pin_version: Optional[str] = None, show_progress: bool = True,
                              installer_timeout: Optional[float] = None) -> bool:
    """Run the upstream cua-driver installer (idempotent: always the latest release, so re-running
    upgrades). ``installer_timeout`` lets quiet callers use a shorter ceiling without weakening the
    explicit install path's stale-lock recovery window."""
    system = platform.system()
    is_windows, is_linux = system == "Windows", system == "Linux"
    install_cmd, manual_hint, script_path = _cua_installer_command(is_windows)
    if install_cmd is None:
        return False
    if show_progress:
        _print_info(f"    {label} cua-driver (background computer-use)..." if verbose
                    else f"→ {label} cua-driver (Computer Use)...")
    driver_cmd = _cua_driver_cmd()
    timeout = _CUA_INSTALLER_TIMEOUT if installer_timeout is None else installer_timeout
    installer_env = _cua_driver_env()
    if pin_version:  # both upstream installers honour CUA_DRIVER_RS_VERSION over the baked default
        installer_env["CUA_DRIVER_RS_VERSION"] = pin_version
    # A previous timed-out install can leave upstream's concurrent-install lock behind; clear it
    # when provably stale so the refresh doesn't wedge waiting on a dead holder.
    # See #58762.
    _clear_stale_cua_install_lock()

    # Unattended refreshes (installer_timeout set by `hermes update`) preflight and may skip.
    # Unattended refreshes (installer_timeout set by `hermes update`) fail FAST on the two conditions that
    # otherwise consume the whole ceiling: 1. Install lock held by a live process — upstream would poll it
    # for up to LOCK_STALE_AFTER_SECONDS=600 before probing the holder. That is the 11-minute silent hang
    # class (#87703; observed live 2026-08-25: "cua-driver refreshing timed out after 660s"). 2. Release
    # host unreachable (outage/DNS/firewall) — the installer would die slowly inside its own retries. A 5s
    # HEAD answers now. Explicit `computer-use install --upgrade` runs keep upstream's full lock-recovery
    # semantics — a human is watching and can wait or Ctrl-C.
    if installer_timeout is not None:
        install_cmd = _unattended_installer_preflight(install_cmd, is_windows)
        if install_cmd is None:
            return False
    popen_kwargs = _installer_popen_kwargs(is_windows, verbose, installer_env)
    try:
        proc = subprocess.Popen(install_cmd, **popen_kwargs)
        try:
            communicated = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _reap_after_timeout(proc, is_windows=is_windows)
            raise
        out = None if verbose else communicated[0]
        if out:
            _record_installer_output(out, proc.returncode)
        installed_binary = _resolved_cua_driver_cmd()
        if proc.returncode == 0 and installed_binary:
            if is_windows and not _repair_cua_driver_autostart_windows(installed_binary,
                                                                        verbose=verbose):
                _print_warning("    cua-driver installed, but auto-start was not registered.")
            if verbose:
                _print_success(f"    {driver_cmd} installed.")
                _print_cua_platform_notes(is_windows, is_linux, fresh_install=True)
            return True
        return _fail(f"    cua-driver {label.lower()} did not complete. Re-run manually:",
                     f"      {manual_hint}")
    except subprocess.TimeoutExpired:
        _print_warning(f"    cua-driver {label.lower()} timed out after {timeout}s.")
        if not is_windows:
            _print_info("    If this repeats, a stale installer lock may be present — check "
                        f"{_cua_install_lock_dir()}")
        _print_info(f"    Re-run manually:  {manual_hint}")
        return False
    except Exception as e:
        return _fail(f"    cua-driver {label.lower()} failed: {e}")
    finally:
        if script_path:
            _remove_quietly(script_path)
