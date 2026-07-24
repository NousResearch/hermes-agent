"""Cross-process single-flight lock for source-checkout updates."""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from pathlib import Path


class UpdateLockError(RuntimeError):
    """Base error for update-lock acquisition failures."""


class UpdateLockBusyError(UpdateLockError):
    """Another process currently owns the update lock."""


class UpdateLockUnavailableError(UpdateLockError):
    """The installation could not provide a usable lock location."""


_FALLBACK_LOCK_DIRNAME = "hermes-update-locks"


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

    install_root = project_root.resolve()
    install_identity = str(install_root)
    digest = hashlib.sha256(install_identity.encode("utf-8")).hexdigest()[:24]
    filename = f"update-{digest}.lock"

    # A venv/site-packages tree is normally writable by its owner. Keeping the
    # lock beside that installation makes the scope explicit and avoids every
    # user/profile state directory becoming a competing lock namespace.
    try:
        if install_root.is_dir() and os.access(install_root, os.W_OK):
            return install_root / filename
    except OSError:
        pass

    # System-wide pip installs are commonly read-only. The fallback is still
    # deterministic and keyed by the canonical installation root, but lives
    # in a private temp directory rather than HOME/XDG state. The directory is
    # created and ownership-checked by UpdateLock.acquire before opening the
    # lock file.
    return Path(tempfile.gettempdir()) / _FALLBACK_LOCK_DIRNAME / filename


def _prepare_lock_parent(path: Path) -> None:
    """Create a lock parent, hardening the shared fallback directory."""
    is_fallback = path.parent.name == _FALLBACK_LOCK_DIRNAME
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if is_fallback else 0o755)
    if not is_fallback or os.name == "nt":
        return
    try:
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise OSError(f"lock parent is not a directory: {path.parent}")
        os.chmod(path.parent, 0o700)
        stat_result = path.parent.stat()
        if hasattr(os, "geteuid") and stat_result.st_uid != os.geteuid():
            raise OSError(f"lock parent is not owned by the current user: {path.parent}")
    except OSError as exc:
        raise UpdateLockUnavailableError(
            f"could not secure update lock directory {path.parent}: {exc}"
        ) from exc


class UpdateLock:
    """An advisory, nonblocking lock with deterministic release."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> "UpdateLock":
        if self._fd is not None:
            return self
        try:
            _prepare_lock_parent(self.path)
            flags = os.O_RDWR | os.O_CREAT
            if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(self.path, flags, 0o600)
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
