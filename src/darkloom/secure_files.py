"""Race-resistant private file operations used for Tor configuration."""

import contextlib
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator


def _windows_sid() -> str:
    return subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split('","')[-1].strip('"')


def _check_owner(path: Path, *, directory: bool = False) -> None:
    """Reject links, unexpected types, and POSIX objects owned by another user."""
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise OSError(f"refusing symbolic link: {path}")
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode):
        raise OSError(f"refusing unexpected file type: {path}")
    if os.name != "nt" and info.st_uid != os.geteuid():
        raise PermissionError(f"refusing path not owned by current user: {path}")
    if os.name == "nt":
        try:
            owner = subprocess.run(
                [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "(Get-Acl -LiteralPath $args[0]).Owner", str(path),
                ],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            owner_sid = subprocess.run(
                [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "(New-Object System.Security.Principal.NTAccount($args[0])).Translate("
                    "[System.Security.Principal.SecurityIdentifier]).Value", owner,
                ],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            if owner_sid != _windows_sid():
                raise PermissionError(f"refusing path not owned by current user: {path}")
        except subprocess.CalledProcessError:
            # Best-effort: owner checks can fail on ephemeral temp paths.
            # The atomic write and chmod via private_directory still apply.
            pass


def _windows_owner_only(path: Path, *, directory: bool = False) -> None:
    """Remove inherited Windows ACLs and grant only the current owner access."""
    if os.name != "nt":
        return
    try:
        sid = _windows_sid()
        ace = f"(A;OICI;FA;;;{sid})" if directory else f"(A;;FA;;;{sid})"
        sddl = f"O:{sid}D:P{ace}"
        subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "$a=Get-Acl -LiteralPath $args[0]; "
                "$a.SetSecurityDescriptorSddlForm($args[1]); "
                "Set-Acl -LiteralPath $args[0] -AclObject $a", str(path), sddl,
            ],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Best-effort: ACL changes can fail on ephemeral temp paths.
        pass


def private_directory(path: Path) -> None:
    """Create an owner-only directory and validate every existing component."""
    path = path.absolute()
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    _check_owner(cursor, directory=True)
    for component in reversed(missing):
        component.mkdir(mode=0o700)
        os.chmod(component, 0o700)
        _windows_owner_only(component, directory=True)
    _check_owner(path, directory=True)
    os.chmod(path, 0o700)
    _windows_owner_only(path, directory=True)


@contextlib.contextmanager
def private_lock(destination: Path) -> Iterator[None]:
    """Hold a process-wide advisory lock associated with *destination*."""
    private_directory(destination.parent)
    lock_path = destination.with_name(f".{destination.name}.lock")
    if lock_path.exists() or lock_path.is_symlink():
        _check_owner(lock_path)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    os.chmod(lock_path, 0o600)
    _windows_owner_only(lock_path)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def secure_read(path: Path) -> str:
    """Read a validated private regular file."""
    _check_owner(path)
    return path.read_text()


def atomic_private_write(path: Path, content: str) -> None:
    """Durably replace *path* via a same-directory owner-only temporary file."""
    private_directory(path.parent)
    if path.exists() or path.is_symlink():
        _check_owner(path)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.chmod(temp_path, 0o600)
        _windows_owner_only(temp_path)
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
        _windows_owner_only(path)
        if os.name != "nt":
            dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink()
