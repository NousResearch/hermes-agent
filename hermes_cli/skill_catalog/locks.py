"""Writer-owned, deadline-bounded OS leases; lock files are never unlinked."""
import ctypes
import errno
import hashlib
import os
import stat
import time
from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from functools import cache
from pathlib import Path

from hermes_cli.skill_catalog.validation import CatalogError, _path

if os.name == "nt":
    import msvcrt
    from ctypes import wintypes
else:
    import fcntl


_CONTENTION = frozenset((errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK, errno.EDEADLK))


def _unavailable():
    return CatalogError("ownership_state_unavailable", "Catalog coordination is unavailable.")


@cache
def _windows_api():
    class Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                 wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
    kernel.LockFileEx.restype = wintypes.BOOL
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel, Overlapped


def _try_lock(fd, shared):
    if os.name != "nt":
        fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        return
    kernel, overlapped = _windows_api()
    flags = 1 | (0 if shared else 2)  # FAIL_IMMEDIATELY, EXCLUSIVE_LOCK
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
    if not kernel.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(overlapped())):
        number = ctypes.get_last_error()
        if number in (33, 36):  # LOCK_VIOLATION / SHARING_BUFFER_EXCEEDED
            raise OSError(errno.EAGAIN, "Catalog lease is held.")
        raise ctypes.WinError(number)


def _redirect(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _windows_open(path):
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        current.mkdir(exist_ok=True)
        if _redirect(current):
            raise OSError(errno.ELOOP, "Redirected coordination directory")
    kernel, _ = _windows_api()
    # OPEN_ALWAYS, OPEN_REPARSE_POINT: inspect the entry, never follow its final redirect.
    handle = kernel.CreateFileW(str(path), 0xC0000000, 7, None, 4, 0x00200000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY | os.O_NOINHERIT)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    return fd


def _posix_open(path):
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(path.name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
                       | os.O_NONBLOCK, 0o600, dir_fd=directory)
    finally:
        os.close(directory)


def _open_file(path):
    path = path.absolute()
    fd = _windows_open(path) if os.name == "nt" else _posix_open(path)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise OSError(errno.EINVAL, "Invalid coordination file")
        os.set_inheritable(fd, False)
        return fd
    except BaseException:
        os.close(fd)
        raise


class FileLease:
    """Keep the writer's own descriptor locked until close, including on parent death."""

    def __init__(self, path: Path, *, shared: bool = False, deadline: float | None = None):
        self.path = Path(path)
        self.shared = shared
        self.deadline = time.monotonic() + 3 if deadline is None else deadline
        self._fd = None

    def acquire(self):
        if self._fd is not None:
            return self
        try:
            self._fd = _open_file(self.path)
            self._poll()
        except OSError:
            self.close()
            raise _unavailable() from None
        except BaseException:
            self.close()
            raise
        return self

    def _poll(self):
        while True:
            try:
                _try_lock(self._fd, self.shared)
                return
            except OSError as exc:
                if exc.errno not in _CONTENTION:
                    raise
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise CatalogError("busy", "Catalog coordination is busy.")
            time.sleep(min(0.01, remaining))

    def close(self):
        """Closing releases the kernel lock without replacing its inode."""
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                os.close(fd)
            except OSError:
                raise _unavailable() from None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.close()


def _target_modes(home, targets):
    modes = {}
    try:
        root = home / "skills"
        if root.exists() and _redirect(root) or root.is_symlink():
            raise ValueError("Redirected skills root")
        for target in targets:
            _path(target)
            current = root
            parts = target.split("/")
            for depth, part in enumerate(parts, 1):
                if part.startswith("."):
                    raise ValueError("Reserved target")
                current /= part
                if current.is_symlink() or current.exists() and _redirect(current):
                    raise ValueError("Redirected target")
                key = "/".join(parts[:depth])
                modes[key] = modes.get(key, False) or depth == len(parts)
    except (ValueError, OSError, UnicodeError):
        raise CatalogError("ownership_conflict", "Catalog target is not a local skill path.") from None
    return sorted(modes.items(), key=lambda item: (item[0].count("/"), item[0].encode("utf-8")))


@contextmanager
def writer_hold(home: Path, targets: Iterable[str] = (), *, deadline=None, root_exclusive=False):
    """Acquire root, then coalesced ancestors/endpoints in depth/exact-UTF8 order."""
    home = Path(home)
    modes = _target_modes(home, targets)
    deadline = time.monotonic() + 3 if deadline is None else deadline
    directory = home / "cache/skill-catalog/live-v1"
    with ExitStack() as stack:
        stack.enter_context(FileLease(directory / "writer-barrier.lock", shared=not root_exclusive,
                                      deadline=deadline))
        for target, exclusive in modes:
            name = hashlib.sha256(target.encode("utf-8")).hexdigest()
            stack.enter_context(FileLease(directory / "writers" / name, shared=not exclusive,
                                          deadline=deadline))
        yield
