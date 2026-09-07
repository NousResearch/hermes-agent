"""Read-only verification at the Windows Desktop handoff receipt boundary."""
from pathlib import Path

from hermes_cli.main_desktop import (
    _desktop_build_needed,
    _desktop_exe_integrity_error,
    _desktop_packaged_executable,
)


def verify_windows_desktop_update(project_root: Path) -> None:
    """Raise when a zero-exit updater left an unusable or stale packaged app."""
    desktop = project_root / "apps" / "desktop"
    executable = _desktop_packaged_executable(desktop)
    if executable is None:
        raise RuntimeError("The updated Desktop executable is missing")
    error = _desktop_exe_integrity_error(executable)
    if error:
        raise RuntimeError(f"The updated Desktop executable is invalid: {error}")
    resources = executable.parent / "resources"
    for required in (resources / "app.asar", resources / "app.asar.unpacked" / "dist" / "index.html"):
        if not required.is_file():
            raise RuntimeError(f"The updated Desktop bundle is incomplete: {required}")
    if _desktop_build_needed(desktop, project_root, source_mode=False):
        raise RuntimeError("The updated Desktop build is stale, unstamped, or incomplete")
