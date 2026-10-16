"""Frozen main.py imports for updater code still running after a checkout swap.

These are not the live helpers in main_desktop, main_install_repair or
main_web_build. Returning from an old installer can trigger a retry or report
success, so those call paths stop for relaunch instead.
"""

from pathlib import Path
from typing import NoReturn

from hermes_cli._old_updater import stop_for_relaunch


# Shim to stop the old updater doing work until relaunch. Retain the filename
# as data only, without recording a fingerprint or replacing the live recorder.
_BYTECODE_FINGERPRINT_FILE = ".bytecode-fingerprint"


class ShimQuarantineError(RuntimeError):
    # Shim to stop the old updater doing work until relaunch. Preserve the
    # historical exception payload without performing or claiming quarantine.
    def __init__(self, failed_shims: list[str]):
        self.failed_shims = list(failed_shims)
        super().__init__("could not quarantine live shim(s): " + ", ".join(self.failed_shims))


def _desktop_stamp_path() -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not select a build stamp.
    stop_for_relaunch()


def _detect_broken_lazy_refresh_imports(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None,
) -> None:
    # Shim to stop the old updater doing work until relaunch: unknown, not healthy [].
    return None


def _expected_windows_pe_machines() -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not approve a binary.
    stop_for_relaunch()


def _hermes_exe_shims(scripts_dir: Path) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not select files to rename.
    stop_for_relaunch()


def _insert_python_pin(args: list[str]) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not prepare an install.
    stop_for_relaunch()


def _interpreter_scripts_dir() -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not target live launchers.
    stop_for_relaunch()


def _load_installable_optional_extras(group: str = "all") -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not select dependency retries.
    stop_for_relaunch()


def _parse_pe_machine(path: Path) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not trigger a binary fallback.
    stop_for_relaunch()


def _quarantine_running_hermes_exe(
    scripts_dir: Path, *, max_attempts: int = 4, failed_out: list[str] | None = None,
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not rename live executables.
    stop_for_relaunch()


def _repair_broken_lazy_refresh_imports(
    install_cmd_prefix: list[str], packages: list[str], *, env: dict[str, str] | None = None,
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not repair or claim health.
    stop_for_relaunch()


def _resolve_install_target_python(
    install_cmd_prefix: list[str], env: dict[str, str] | None,
) -> None:
    # Shim to stop the old updater doing work until relaunch: no interpreter to probe.
    return None


def _restore_quarantined_exes(moved: list[tuple[Path, Path]]) -> None:
    # Shim to stop the old updater doing work until relaunch, including finally-block renames.
    return None


def _run_install_with_heartbeat(
    cmd: list[str], *, env: dict[str, str] | None = None, heartbeat_interval_seconds: int = 30,
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not run pip/uv or report success.
    stop_for_relaunch()


def _run_package_only_install(cmd: list[str], *, env: dict[str, str] | None = None) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not run a package-only retry.
    stop_for_relaunch()


def _run_quarantined_install(
    cmd: list[str], *, env: dict[str, str] | None = None, scripts_dir: Path | None = None,
    strict_quarantine: bool = False,
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not install or enable fallback.
    stop_for_relaunch()


def _run_with_idle_timeout(
    cmd: list[str], cwd: Path, *, idle_timeout_seconds: int = 180, indent: str = "    ",
    env: dict[str, str] | None = None,
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not build or synthesize success.
    stop_for_relaunch()


def _self() -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not expose live process scans.
    stop_for_relaunch()


def _verify_console_scripts_installed(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None,
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not reinstall entry points.
    stop_for_relaunch()


def _verify_core_dependencies_installed(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None, group: str = "all",
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not reinstall missing packages.
    stop_for_relaunch()


def _web_ui_build_needed(web_dir: Path) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not claim an unbuilt UI is current.
    stop_for_relaunch()


def _windows_native_machine() -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not approve a fallback architecture.
    stop_for_relaunch()


def _windows_shim_in_process_chain() -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not spawn an install handoff.
    stop_for_relaunch()


def _write_web_ui_build_stamp(project_root: Path, web_dir: Path) -> None:
    # Shim to stop the old updater doing work until relaunch. Never write a success marker.
    return None
