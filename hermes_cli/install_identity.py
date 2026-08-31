"""Stable opaque identity shared by every profile in one Hermes install."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Optional
import uuid

from hermes_constants import get_default_hermes_root

_INSTALL_ID_FILENAME = "install_id"
_INSTALL_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_INSTALL_ID_CACHE: dict[str, Optional[str]] = {"root": None, "value": None}
_INSTALL_ID_LOCK = threading.Lock()
_INSTALL_ID_PUBLICATION_LOCK = threading.Lock()


@contextlib.contextmanager
def _install_id_file_lock(root: Path):
    """Serialize identity publication across processes on POSIX and Windows."""
    lock_path = root / ".install_id.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    windows = os.name == "nt"
    try:
        if windows:
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
                os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if windows:
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _fsync_directory(path: Path) -> None:
    """Best-effort durability for the directory entry after replace."""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def read_or_create_install_id(root: Path | None = None) -> Optional[str]:
    """Read or atomically mint the opaque id for the physical install.

    ``None`` means the id could neither be read nor persisted. Returning an
    ephemeral id would violate the authority and connection-registry contract.
    """
    root = get_default_hermes_root() if root is None else root
    path = root / _INSTALL_ID_FILENAME
    try:
        existing = path.read_text(encoding="utf-8").strip().lower()
        if _INSTALL_ID_RE.fullmatch(existing):
            return existing
    except FileNotFoundError:
        pass
    except (OSError, UnicodeDecodeError):
        return None

    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    try:
        # Windows byte-range locks can report a same-process lock conflict
        # instead of waiting for another thread. Serialize threads here, then
        # retain the file lock as the cross-process publication fence.
        with _INSTALL_ID_PUBLICATION_LOCK, _install_id_file_lock(root):
            try:
                existing = path.read_text(encoding="utf-8").strip().lower()
                if _INSTALL_ID_RE.fullmatch(existing):
                    return existing
            except FileNotFoundError:
                pass
            except (OSError, UnicodeDecodeError):
                return None

            minted = uuid.uuid4().hex
            fd, tmp_name = tempfile.mkstemp(dir=str(root), prefix=".install_id-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(minted + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, path)
                _fsync_directory(root)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
                raise

            committed = path.read_text(encoding="utf-8").strip().lower()
            return committed if _INSTALL_ID_RE.fullmatch(committed) else None
    except OSError:
        return None


def get_install_id(
    *,
    cache: dict[str, Optional[str]] | None = None,
) -> Optional[str]:
    """Return the process-cached stable id for the active Hermes root."""
    root = get_default_hermes_root()
    root_key = str(root)
    target_cache = _INSTALL_ID_CACHE if cache is None else cache
    cached = target_cache.get("value")
    if cached and target_cache.get("root") in (None, root_key):
        return cached

    with _INSTALL_ID_LOCK:
        cached = target_cache.get("value")
        if cached and target_cache.get("root") in (None, root_key):
            return cached
        value = read_or_create_install_id(root)
        if value:
            target_cache["root"] = root_key
            target_cache["value"] = value
        return value


__all__ = ["get_install_id", "read_or_create_install_id"]
