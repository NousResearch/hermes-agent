"""Permission checks for files stored below ``HERMES_HOME``.

The doctor check is intentionally an audit rather than a write-policy change:
``--fix`` only tightens known paths, and refuses symlinks instead of resolving
them.  That keeps a compromised or surprising state tree from redirecting a
repair outside Hermes' state directory.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.doctor_report import Finding, check_info, check_ok, check_warn, doctor_check


@dataclass(frozen=True)
class _PermissionTarget:
    path: Path
    label: str
    mode: int
    is_directory: bool


def _state_permission_targets(home: Path, display_home: str) -> tuple[_PermissionTarget, ...]:
    """Return the finite set of state paths that may be audited or repaired."""
    return (
        _PermissionTarget(home, display_home, 0o700, True),
        _PermissionTarget(home / "sessions", f"{display_home}/sessions", 0o700, True),
        _PermissionTarget(home / "logs", f"{display_home}/logs", 0o700, True),
        _PermissionTarget(home / ".env", f"{display_home}/.env", 0o600, False),
        _PermissionTarget(home / "config.yaml", f"{display_home}/config.yaml", 0o600, False),
        _PermissionTarget(home / "state.db", f"{display_home}/state.db", 0o600, False),
        _PermissionTarget(home / "state.db-wal", f"{display_home}/state.db-wal", 0o600, False),
        _PermissionTarget(home / "state.db-shm", f"{display_home}/state.db-shm", 0o600, False),
    )


def _tighten_without_following(path: Path, mode: int, is_directory: bool) -> None:
    """Set ``mode`` through an fd opened with ``O_NOFOLLOW``.

    The lstat performed by the caller catches ordinary symlinks for a useful
    doctor finding.  Opening a second time with ``O_NOFOLLOW`` also prevents a
    symlink swap between that audit and the chmod operation.
    """
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError("non-following file access is unavailable on this platform")
    flags = os.O_RDONLY | nofollow
    if is_directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def _permission_issue(target: _PermissionTarget, actual_mode: int) -> str:
    return (f"{target.label} has insecure permissions {actual_mode:04o} (expected {target.mode:04o}) — "
            "run 'hermes doctor --fix' to tighten them")


def _check_target(should_fix: bool, finding: Finding, target: _PermissionTarget) -> None:
    try:
        metadata = target.path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        message = f"Could not inspect permissions for {target.label}: {exc}"
        check_warn(message)
        finding.issues.append(message)
        return

    if stat.S_ISLNK(metadata.st_mode):
        message = f"{target.label} is a symlink; refusing to inspect or modify its target"
        check_warn(message)
        finding.issues.append(message)
        return
    if stat.S_ISDIR(metadata.st_mode) != target.is_directory:
        expected = "directory" if target.is_directory else "regular file"
        message = f"{target.label} is not the expected {expected}; refusing to modify it"
        check_warn(message)
        finding.issues.append(message)
        return

    actual_mode = stat.S_IMODE(metadata.st_mode)
    if actual_mode == target.mode:
        check_ok(f"{target.label} permissions are {target.mode:04o}")
        return

    if not should_fix:
        message = _permission_issue(target, actual_mode)
        check_warn(message)
        finding.issues.append(message)
        return

    try:
        _tighten_without_following(target.path, target.mode, target.is_directory)
    except OSError as exc:
        message = f"{target.label} permissions could not tighten to {target.mode:04o}: {exc}"
        check_warn(message)
        finding.issues.append(message)
        return
    check_ok(f"Tightened {target.label} permissions to {target.mode:04o}")
    finding.fixed += 1


@doctor_check()
def _check_state_permissions(should_fix: bool, finding: Finding) -> None:
    """Report broad state permissions and optionally tighten known local paths."""
    from hermes_cli.doctor import HERMES_HOME, _DHH

    if os.name == "nt":
        check_info("POSIX state-file mode audit is not applicable on Windows")
        return
    if not HERMES_HOME.exists() and not HERMES_HOME.is_symlink():
        check_info(f"{_DHH} does not exist yet; state permission audit skipped")
        return
    targets = _state_permission_targets(HERMES_HOME, _DHH)
    # A symlinked home is itself a finding.  Do not construct child paths from
    # it: their lstat calls would traverse the escaped root before noticing a
    # final-component symlink.
    if HERMES_HOME.is_symlink():
        _check_target(should_fix, finding, targets[0])
        return
    for target in targets:
        _check_target(should_fix, finding, target)
