"""Cross-process single-flight lock for source-checkout updates."""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path


class UpdateLockError(RuntimeError):
    """Base error for update-lock acquisition failures."""


class UpdateLockBusyError(UpdateLockError):
    """Another process currently owns the update lock."""


class UpdateLockUnavailableError(UpdateLockError):
    """The checkout could not provide a usable common git directory."""


def _common_git_dir(project_root: Path) -> Path:
    dot_git = project_root / ".git"
    if dot_git.is_dir():
        return dot_git.resolve()
    if not dot_git.is_file():
        raise UpdateLockUnavailableError(f"not a git checkout: {project_root}")

    try:
        gitdir_line = dot_git.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise UpdateLockUnavailableError(f"could not read {dot_git}: {exc}") from exc
    prefix = "gitdir:"
    if not gitdir_line.lower().startswith(prefix):
        raise UpdateLockUnavailableError(f"invalid gitdir file: {dot_git}")

    git_dir = Path(gitdir_line[len(prefix) :].strip())
    if not git_dir.is_absolute():
        git_dir = project_root / git_dir
    git_dir = git_dir.resolve()
    commondir = git_dir / "commondir"
    if not commondir.is_file():
        return git_dir
    try:
        common = Path(commondir.read_text(encoding="utf-8").strip())
    except OSError as exc:
        raise UpdateLockUnavailableError(f"could not read {commondir}: {exc}") from exc
    if not common.is_absolute():
        common = git_dir / common
    return common.resolve()


def get_update_lock_path(project_root: Path) -> Path:
    """Return a stable lock path for this source checkout or package install.

    Git worktrees share their common git directory.  Pip installations have no
    ``.git`` directory, so derive a user-writable, installation-specific path
    from the canonical install root instead of failing closed before the update
    can start.  The path deliberately does not include ``HERMES_HOME``: two
    profiles using one installation must still serialize upgrades.
    """

    project_root = Path(project_root)
    if (project_root / ".git").exists():
        return _common_git_dir(project_root) / "hermes-update.lock"

    install_identity = str(project_root.resolve())
    digest = hashlib.sha256(install_identity.encode("utf-8")).hexdigest()[:24]
    state_home = Path(
        os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))
    )
    return state_home / "hermes" / f"update-{digest}.lock"


class UpdateLock:
    """An advisory, nonblocking lock with deterministic release."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> "UpdateLock":
        if self._fd is not None:
            return self
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            raise UpdateLockUnavailableError(f"could not open {self.path}: {exc}") from exc

        try:
            if os.name == "nt":
                import msvcrt

                if os.path.getsize(self.path) == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in {errno.EACCES, errno.EAGAIN} or isinstance(exc, PermissionError):
                raise UpdateLockBusyError(
                    f"another Hermes update already holds {self.path}"
                ) from exc
            raise UpdateLockUnavailableError(f"could not lock {self.path}: {exc}") from exc

        self._fd = fd
        return self

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "UpdateLock":
        return self.acquire()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()


def acquire_update_lock(project_root: Path) -> UpdateLock:
    """Acquire the checkout-scoped update lock or fail closed."""

    return UpdateLock(get_update_lock_path(project_root)).acquire()
