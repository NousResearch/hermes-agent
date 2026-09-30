"""No-follow filesystem operations and cross-process locks for marketplace state.

Ported from PR #107111 (Carl Taylor); kept local to avoid broad shared-utils changes.
"""
import errno
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Union
from hermes_constants import get_hermes_home

def sync_directory(path: Union[str, Path]) -> None:
    """Flush directory-entry changes on POSIX and Windows."""
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.FlushFileBuffers.argtypes = (wintypes.HANDLE,)
    kernel32.FlushFileBuffers.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateFileW(
        str(path),
        0xC0000000,
        0x00000001 | 0x00000002,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel32.FlushFileBuffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


@contextmanager
def secure_parent_directory(
    path: Union[str, Path], root: Union[str, Path], *, create: bool = False
):
    """Hold a no-follow parent descriptor for a path beneath a trusted root."""
    path = Path(path).absolute()
    root = Path(root).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise OSError(errno.EPERM, f"Path escapes trusted root: {path}") from exc
    if not relative.parts:
        raise OSError(errno.EINVAL, "A child path is required")

    root_real = root.resolve(strict=True)
    if os.name != "posix":
        import ctypes
        from ctypes import wintypes

        class FileAttributeTagInfo(ctypes.Structure):
            _fields_ = [
                ("file_attributes", wintypes.DWORD),
                ("reparse_tag", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.GetFileInformationByHandleEx.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        invalid = wintypes.HANDLE(-1).value
        handles = []
        parent = root_real
        paths = [root_real]
        for part in relative.parts[:-1]:
            parent /= part
            paths.append(parent)
        try:
            for current in paths:
                if not current.exists():
                    if create:
                        current.mkdir()
                    else:
                        raise FileNotFoundError(current)
                handle = kernel32.CreateFileW(
                    str(current),
                    0x00000080,
                    0x00000001 | 0x00000002,
                    None,
                    3,
                    0x02000000 | 0x00200000,
                    None,
                )
                if handle == invalid:
                    raise ctypes.WinError(ctypes.get_last_error())
                info = FileAttributeTagInfo()
                if not kernel32.GetFileInformationByHandleEx(
                    handle, 9, ctypes.byref(info), ctypes.sizeof(info)
                ):
                    kernel32.CloseHandle(handle)
                    raise ctypes.WinError(ctypes.get_last_error())
                if info.file_attributes & 0x400:
                    kernel32.CloseHandle(handle)
                    raise OSError(errno.ELOOP, f"Unsafe reparse-point path: {current}")
                handles.append(handle)
            yield None, parent, relative.name
        finally:
            for handle in reversed(handles):
                kernel32.CloseHandle(handle)
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(root_real, flags)
    try:
        for part in relative.parts[:-1]:
            try:
                child = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, root_real.joinpath(*relative.parts[:-1]), relative.name
    finally:
        os.close(fd)


def secure_open_file(
    path: Union[str, Path],
    root: Union[str, Path],
    flags: int,
    mode: int = 0o600,
    *,
    create_parent: bool = False,
) -> int:
    """Open a regular child without releasing parent containment first."""
    with secure_parent_directory(path, root, create=create_parent) as (
        parent_fd,
        parent,
        name,
    ):
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        candidate = parent / name
        if parent_fd is None and os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateFileW.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            kernel32.CreateFileW.restype = wintypes.HANDLE
            kernel32.GetFileInformationByHandleEx.argtypes = (
                wintypes.HANDLE,
                ctypes.c_int,
                wintypes.LPVOID,
                wintypes.DWORD,
            )
            kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL

            access = 0x80000000
            if flags & (os.O_WRONLY | os.O_RDWR):
                access |= 0x40000000
            if flags & os.O_CREAT:
                creation = 1 if flags & os.O_EXCL else 2 if flags & os.O_TRUNC else 4
            else:
                creation = 5 if flags & os.O_TRUNC else 3
            handle = kernel32.CreateFileW(
                str(candidate),
                access,
                0x00000001 | 0x00000002,
                None,
                creation,
                0x00000080 | 0x00200000,
                None,
            )
            invalid = wintypes.HANDLE(-1).value
            if handle == invalid:
                raise ctypes.WinError(ctypes.get_last_error())

            class FileAttributeTagInfo(ctypes.Structure):
                _fields_ = [
                    ("file_attributes", wintypes.DWORD),
                    ("reparse_tag", wintypes.DWORD),
                ]

            info = FileAttributeTagInfo()
            if not kernel32.GetFileInformationByHandleEx(
                handle, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):
                kernel32.CloseHandle(handle)
                raise ctypes.WinError(ctypes.get_last_error())
            if info.file_attributes & (0x10 | 0x400):
                kernel32.CloseHandle(handle)
                raise OSError(errno.ELOOP, f"Unsafe control path: {candidate}")
            try:
                fd_flags = flags & (
                    os.O_APPEND
                    | os.O_WRONLY
                    | os.O_RDWR
                    | getattr(os, "O_TEXT", 0)
                    | getattr(os, "O_BINARY", 0)
                )
                return msvcrt.open_osfhandle(handle, fd_flags)
            except BaseException:
                kernel32.CloseHandle(handle)
                raise

        fd = (
            os.open(name, flags | nofollow, mode, dir_fd=parent_fd)
            if parent_fd is not None
            else os.open(candidate, flags | nofollow, mode)
        )
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise OSError(errno.EPERM, f"Control path is not a regular file: {path}")
        return fd


def secure_atomic_write_text(
    path: Union[str, Path],
    content: str,
    root: Union[str, Path],
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically write beneath a held no-follow parent directory."""
    with secure_parent_directory(path, root, create=True) as (parent_fd, parent, name):
        if parent_fd is None:
            if (parent / name).is_symlink():
                raise OSError(errno.ELOOP, f"Unsafe symlinked control file: {parent / name}")
            fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=parent)
            try:
                with os.fdopen(fd, "w", encoding=encoding) as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, parent / name)
                sync_directory(parent)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return
        tmp_name = f".{name}.tmp-{os.getpid()}-{time.time_ns()}"
        fd = os.open(
            tmp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            with os.fdopen(fd, "w", encoding=encoding) as handle:
                fd = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(tmp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def secure_unlink(
    path: Union[str, Path], root: Union[str, Path], *, missing_ok: bool = False
) -> None:
    """Unlink a child through a held no-follow parent directory."""
    with secure_parent_directory(path, root) as (parent_fd, parent, name):
        try:
            if parent_fd is not None:
                os.unlink(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            else:
                os.unlink(parent / name)
                sync_directory(parent)
        except FileNotFoundError:
            if not missing_ok:
                raise


class PluginOperationError(Exception):
    pass

def _lock_file(handle) -> None:
    """Acquire a blocking one-byte cross-process lock."""
    if os.name == "nt":
        import errno
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                time.sleep(0.1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _open_lock_path(path: Path):
    """Open a regular lock file beneath a held no-follow parent."""
    if path.is_symlink():
        raise PluginOperationError(f"Lock path must not be a symlink: {path}")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = secure_open_file(path, get_hermes_home(), flags, create_parent=True)
    except OSError as exc:
        raise PluginOperationError(
            f"Could not safely open lock file {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise PluginOperationError(f"Lock path must be a regular file: {path}")
    return os.fdopen(fd, "a+b")
