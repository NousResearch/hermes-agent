"""Small, side-effect-free checks for external secret-source binaries.

The secret-source modules invoke third-party CLIs with live credentials.  Keep
path canonicalisation and the PATH trust policy in one place so a new source
does not accidentally reintroduce relative-path or writable-directory lookup.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Optional


_PROBE_ENV_KEYS = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SystemRoot",
    "WINDIR",
    # Windows native launchers and the official CLIs consult these when
    # locating the system shell, executable suffixes, and user/machine
    # configuration roots.  They are non-secret and safe for a version-only
    # probe; provider credentials are deliberately not included here.
    "SystemDrive",
    "PATHEXT",
    "COMSPEC",
    "ProgramData",
    "APPDATA",
    "LOCALAPPDATA",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
)

# ``.bat`` and ``.cmd`` files are interpreted by ``cmd.exe`` rather than
# being native executable images.  They are not safe for credential-bearing
# subprocesses, even when they are executable according to Windows APIs.
_WINDOWS_EXECUTABLE_SUFFIXES = frozenset({".com", ".exe"})

# Version probes may need to launch a helper from PATH (for example, Linux's
# ``ldd`` is commonly a shell script).  Never hand a probe the caller's PATH:
# it can contain a writable directory even when the already-resolved binary is
# trusted.  These are candidates only; _trusted_probe_path filters them using
# the same root-owned, non-writable directory policy as PATH discovery.
_POSIX_PROBE_PATH = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)


def _get_effective_uid() -> Optional[int]:
    """Return the POSIX effective UID when the platform exposes it."""
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return None
    try:
        return geteuid()
    except AttributeError:
        return None


def resolve_executable(
    path: Path | str,
    *,
    check_parent_dirs: bool = False,
    reject_current_owner: bool = False,
    check_explicit_parent_dirs: bool = False,
) -> Optional[Path]:
    """Return a canonical executable path, or ``None`` when it is unsafe.

    Callers that resolve a user-specified ``binary_path`` set
    ``check_explicit_parent_dirs``.  On POSIX that mode permits a private
    chain owned by root or the current user, while rejecting another
    unprivileged owner and group/world-writable parents.  PATH results set
    ``check_parent_dirs`` and ``reject_current_owner``: on POSIX they require
    a root-owned leaf and root-owned, non-group/world-writable parent chain.
    Profile-local installations such as ``~/.local/bin/op`` therefore remain
    available through an explicit absolute ``binary_path`` without weakening
    the stricter PATH policy.
    """
    if check_parent_dirs and check_explicit_parent_dirs:
        return None
    try:
        candidate = Path(path).expanduser()
    except (TypeError, ValueError, RuntimeError):
        return None
    if not candidate.is_absolute():
        return None

    try:
        resolved = candidate.resolve(strict=True)
        info = resolved.stat()
    except (OSError, RuntimeError):
        return None
    if not stat.S_ISREG(info.st_mode):
        return None

    if os.name == "nt":
        # ``os.access(..., X_OK)`` is effectively an existence check on
        # Windows.  Restrict executable candidates to native executable
        # suffixes so a credential-bearing child cannot be redirected to an
        # arbitrary data file or a cmd.exe-interpreted script.
        if (
            not os.access(resolved, os.X_OK)
            or resolved.suffix.lower() not in _WINDOWS_EXECUTABLE_SUFFIXES
        ):
            return None
    elif not info.st_mode & (
        stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    ):
        return None

    effective_uid: Optional[int] = None
    if os.name != "nt":
        effective_uid = _get_effective_uid()
        # Root-owned system binaries are a valid trust anchor when Hermes
        # itself runs as root.  A non-root process must still reject a PATH
        # binary owned by that process, since it can be planted or replaced
        # by the current user.
        if (
            reject_current_owner
            and effective_uid is not None
            and effective_uid != 0
            and info.st_uid == effective_uid
        ):
            return None

        # A group/world-writable executable can be replaced by another user
        # before the credential-bearing child is started.  This applies to
        # both explicit absolute paths and PATH results; explicit paths are
        # intentionally allowed to remain user-owned otherwise.
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            return None

    if check_parent_dirs and os.name == "nt":
        # Windows does not expose a portable owner/mode API equivalent to the
        # POSIX checks below.  PATH discovery is therefore limited to the
        # machine-managed installation roots; profile-local installs remain
        # supported through an explicit absolute ``binary_path``.
        trusted_roots = []
        for env_name in (
            "ProgramFiles",
            "ProgramFiles(x86)",
            "ProgramW6432",
            "SystemRoot",
        ):
            value = os.environ.get(env_name)
            if value:
                try:
                    trusted_roots.append(Path(value).resolve())
                except (OSError, RuntimeError):
                    pass
        if not any(
            resolved == root or root in resolved.parents
            for root in trusted_roots
        ):
            return None
    elif check_parent_dirs and os.name != "nt":
        # PATH is a shared trust boundary.  A non-root owner can replace a
        # leaf even when its current mode is safe, and a writable directory
        # owned by another user can be made writable/replaced by that owner.
        # Keep PATH candidates to a root-owned chain; users with profile-local
        # installs can opt in with an explicit absolute path above.
        if info.st_uid != 0:
            return None
        # A safe executable can still be replaced through a writable parent
        # directory.  Check every canonical directory up to the filesystem
        # root.
        for parent in (resolved.parent, *resolved.parent.parents):
            try:
                parent_info = parent.stat()
            except OSError:
                return None
            if not stat.S_ISDIR(parent_info.st_mode):
                return None
            if parent_info.st_uid != 0:
                return None
            if parent_info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                return None
    elif check_explicit_parent_dirs and os.name == "nt":
        # Windows has no portable stdlib equivalent of POSIX ownership/mode
        # checks.  Keep explicit paths inside the normal per-user or
        # machine-managed roots; administrators can still use an absolute
        # path in Program Files, while profile-local installs remain valid.
        trusted_roots = []
        try:
            trusted_roots.append(Path.home().resolve())
        except (OSError, RuntimeError):
            pass
        for env_name in (
            "USERPROFILE",
            "LOCALAPPDATA",
            "ProgramFiles",
            "ProgramFiles(x86)",
            "ProgramW6432",
            "SystemRoot",
        ):
            value = os.environ.get(env_name)
            if value:
                try:
                    trusted_roots.append(Path(value).resolve())
                except (OSError, RuntimeError):
                    pass
        if not any(
            resolved == root or root in resolved.parents
            for root in trusted_roots
        ):
            return None
    elif check_explicit_parent_dirs and os.name != "nt":
        # An explicit profile path is allowed to be owned by the invoking
        # user, but every canonical parent must still be private.  Checking
        # the resolved chain prevents a symlinked ancestor from bypassing the
        # policy before the credential-bearing child starts.
        effective_uid = _get_effective_uid()
        if effective_uid is None:
            return None
        trusted_owners = {0, effective_uid}
        if info.st_uid not in trusted_owners:
            return None
        for parent in (resolved.parent, *resolved.parent.parents):
            try:
                parent_info = parent.stat()
            except OSError:
                return None
            if not stat.S_ISDIR(parent_info.st_mode):
                return None
            if parent_info.st_uid not in trusted_owners:
                return None
            if parent_info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                return None

    return resolved


def _trusted_probe_path(path: Path) -> str:
    """Return a credential-free PATH containing only trusted directories."""
    candidates = [path.parent, *(Path(item) for item in _POSIX_PROBE_PATH)]
    trusted = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if os.name != "nt" and not _is_root_owned_private_chain(resolved):
            continue
        value = str(resolved)
        if value not in seen:
            trusted.append(value)
            seen.add(value)
    return os.pathsep.join(trusted)


def _is_root_owned_private_chain(path: Path) -> bool:
    """Whether a canonical directory and every ancestor are trusted on POSIX."""
    for directory in (path, *path.parents):
        try:
            info = directory.stat()
        except OSError:
            return False
        if not stat.S_ISDIR(info.st_mode):
            return False
        if info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            return False
    return True


def probe_environment(path: Path) -> dict[str, str]:
    """Build the minimal environment used by a version-only probe."""
    env = {}
    for key in _PROBE_ENV_KEYS:
        if key == "PATH":
            env[key] = _trusted_probe_path(path)
        elif (value := os.environ.get(key)) is not None:
            env[key] = value
    return env


def probe_version(path: Path, pattern: str, *, timeout: float = 5) -> bool:
    """Run a version-only probe without passing provider credentials."""
    env = probe_environment(path)
    try:
        proc = subprocess.run(
            [str(path), "--version"],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0:
        return False
    output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    return re.search(pattern, output) is not None
