"""PM-backed CUA setup and the host integration not supplied by its binary archive."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Optional

from hermes_cli.cli_output import (
    print_info as _print_info, print_success as _print_success, print_warning as _print_warning)


def _run_text(cmd: list, *, timeout, capture_output: bool = True,
              **kwargs) -> subprocess.CompletedProcess:
    """Run a text subprocess with consistent decoding."""
    return subprocess.run(cmd, capture_output=capture_output, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, **kwargs)


def _fail(message: str, *hints: str) -> bool:
    _print_warning(message)
    for hint in hints:
        _print_info(hint)
    return False


def _print_output_tail(result: subprocess.CompletedProcess, printer=None) -> None:
    for line in (result.stderr or result.stdout or "").strip().splitlines()[-3:]:
        (printer or _print_info)(f"      {line[:200]}")


def _post_setup_no_window_flags(*, streams_to_console: bool = False) -> int:
    """Hide Windows children unless their output is going to a real console."""
    from hermes_cli._subprocess_compat import windows_hide_flags
    flags = windows_hide_flags()
    try:
        if flags and streams_to_console and sys.stdout is not None and sys.stdout.isatty():
            return 0
    except Exception:
        pass
    return flags or 0


def _cua_driver_cmd() -> str:
    return os.environ.get("HERMES_CUA_DRIVER_CMD", "").strip() or "cua-driver"


def _cua_version_summary(raw: str, *, limit: int = 120) -> str:
    """Bound an external binary's potentially multiline version banner."""
    return next((line.strip()[:limit] for line in (raw or "").splitlines() if line.strip()), "")


def _resolved_cua_driver_cmd() -> Optional[str]:
    from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd
    return resolve_cua_driver_cmd()


def _cua_driver_env() -> dict:
    from tools.computer_use.cua_backend import sanitized_cua_driver_env
    return sanitized_cua_driver_env()


_CUA_DRIVER_CONTRACT_CACHE: dict = {}


def _cua_driver_contract_status(binary: Optional[str] = None) -> dict:
    """Cache the runtime manifest check by binary identity for UI polling."""
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
    state = _cua_driver_contract_status()
    if not state.get("ready"):
        return False
    if sys.platform == "darwin":
        from tools.computer_use.cua_backend_daemon import _resolve_cua_driver_app_path
        return bool(_resolve_cua_driver_app_path(state["binary"]))
    return sys.platform != "win32" or _cua_driver_autostart_registered_windows()


def install_cua_driver(upgrade: bool = False, show_installer_progress: bool = True) -> bool:
    """Prepare the PM pin and host setup for an explicit install/upgrade command.

    Both CLI modes reconcile the same pin; neither discovers a vendor release.
    A configured override is validated, never replaced or acquired by PM.
    Unattended callers should use PM ensure directly, without interactive host setup.
    """
    from pm import ensure

    override = os.environ.get("HERMES_CUA_DRIVER_CMD", "").strip()
    binary = _resolved_cua_driver_cmd()
    fresh_install = binary is None
    if override:
        if not binary:
            return _fail(f"    HERMES_CUA_DRIVER_CMD does not resolve to an executable: {override}",
                         "    Fix or unset the override before running computer-use install.")
    else:
        if show_installer_progress:
            _print_info("    Preparing the pinned cua-driver with Hermes PM...")
        try:
            ensure("cua-driver", explicit=True)
        except Exception as exc:
            return _fail(f"    cua-driver preparation failed: {exc}")
        binary = _resolved_cua_driver_cmd()

    if not binary:
        return _fail("    PM did not select a usable cua-driver executable.")
    _CUA_DRIVER_CONTRACT_CACHE.clear()
    contract = _cua_driver_contract_status(binary)
    if not contract.get("ready"):
        hint = ("    Update the binary selected by HERMES_CUA_DRIVER_CMD, or unset the override."
                if override else "    Run: hermes computer-use doctor")
        return _fail("    cua-driver runtime contract is unusable: "
                     f"{contract.get('reason') or 'unknown error'}.", hint)
    if sys.platform == "win32" and not _repair_cua_driver_autostart_windows(
            binary, verbose=show_installer_progress):
        return _fail("    cua-driver is compatible, but Windows autostart setup failed.")
    if sys.platform == "darwin":
        from tools.computer_use.cua_backend_daemon import (
            _resolve_cua_driver_app_path, _validate_cua_driver_app_signature)

        app = _resolve_cua_driver_app_path(binary)
        if not app:
            return _fail("    macOS computer use requires the signed CuaDriver.app, not a bare binary.",
                         "    The PM cua-driver package must include its signed macOS app bundle.")
        try:
            _validate_cua_driver_app_signature(app)
            result = _run_text([
                "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
                "LaunchServices.framework/Support/lsregister", "-f", app], timeout=15)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            return _fail(f"    cua-driver macOS app registration failed: {exc}")
        if result.returncode:
            _print_output_tail(result)
            return _fail("    cua-driver macOS app registration failed.")
    if show_installer_progress:
        _print_success(f"    cua-driver ready: {contract.get('version') or 'unknown version'}.")
        _print_cua_platform_notes(sys.platform == "win32", sys.platform == "linux",
                                  fresh_install=fresh_install)
    return True


def _ps_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _cua_driver_autostart_registered_windows(binary: Optional[str] = None) -> bool:
    """A task targeting a previous PM version is not a ready registration."""
    if sys.platform != "win32":
        return False
    from xml.etree import ElementTree

    binary = binary or _resolved_cua_driver_cmd()
    if not binary:
        return False
    try:
        result = subprocess.run(
            ["schtasks.exe", "/Query", "/TN", "cua-driver-serve", "/XML"],
            capture_output=True, timeout=10, creationflags=_post_setup_no_window_flags())
        if result.returncode:
            return False
        # Parse bytes: schtasks' XML declaration carries the output encoding.
        task = ElementTree.fromstring(result.stdout)
        commands = task.findall(".//{*}Exec/{*}Command")
        return any(os.path.normcase((node.text or "").strip().strip('"')) == os.path.normcase(binary)
                   for node in commands)
    except (OSError, subprocess.SubprocessError, ElementTree.ParseError):
        return False


def _repair_cua_driver_autostart_windows(driver_cmd: str, *, verbose: bool) -> bool:
    """Register autostart using structured arguments, including paths with spaces."""
    if sys.platform != "win32":
        return True
    binary = shutil.which(driver_cmd)
    if not binary:
        return False
    if _cua_driver_autostart_registered_windows(binary):
        return True
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
        return _cua_driver_autostart_registered_windows(binary)
    _print_warning("    cua-driver autostart registration failed.")
    _print_output_tail(result)
    _print_info(f"    From an elevated shell, run: & {_ps_single_quote(binary)} autostart enable")
    return False


def _print_cua_platform_notes(is_windows: bool, is_linux: bool, *, fresh_install: bool) -> None:
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
        _print_info("    Allow CuaDriver.app; run `hermes computer-use permissions grant` for guidance.")