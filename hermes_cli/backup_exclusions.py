"""Per-file exclusion predicates for backups: runtime ``*.lock`` files and non-regular entries."""

import stat
from pathlib import Path

# ``*.lock`` files are live process locks (auth.lock, runtime/active_sessions.lock, gateway.lock,
# scope locks). A Windows ``msvcrt.locking`` owner makes one unreadable, which marked an otherwise
# complete backup incomplete; restored elsewhere it names a PID that does not exist. Dependency
# lockfiles a skill or plugin ships are user data and stay.
_RUNTIME_LOCK_SUFFIX = ".lock"
_KEPT_LOCKFILE_NAMES = frozenset({
    "uv.lock", "poetry.lock", "pdm.lock", "Pipfile.lock", "Cargo.lock", "yarn.lock", "bun.lock",
    "flake.lock", "Gemfile.lock", "composer.lock", "deno.lock", "mix.lock", "pubspec.lock",
    "Podfile.lock", "renv.lock", "conan.lock", "requirements.lock",
})


def is_runtime_lock(name: str) -> bool:
    return name.endswith(_RUNTIME_LOCK_SUFFIX) and name not in _KEPT_LOCKFILE_NAMES


def is_non_regular_path(path: Path) -> bool:
    """True for symlinks, sockets, devices, and other non-regular filesystem entries.

    A failed ``lstat`` is not treated as an exclusion: the archive writer must see the path and
    report the read failure instead of silently claiming a complete backup.
    """
    try:
        return not stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False
