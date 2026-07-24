#!/usr/bin/env python3
"""Break-glass helper: remove only a malformed Anthropic shared scope marker.

Stdlib only — no Hermes imports. Use when `hermes auth` cannot start.

  python scripts/break_glass_anthropic_scope.py \\
      --root /absolute/hermes/root \\
      --backup-dir /absolute/owner-only/dir \\
      --yes
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore
try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None  # type: ignore


def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _check_abs(path: Path, name: str) -> None:
    if not path.is_absolute():
        _die(f"{name} must be absolute", 2)


def _safe_regular(path: Path, *, must_exist: bool) -> None:
    if path.is_symlink():
        _die(f"refusing symlink: {path}", 1)
    if path.exists():
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode):
            _die(f"not a regular file: {path}", 1)
        if st.st_nlink != 1:
            _die(f"refusing hard-linked path: {path}", 1)
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            _die(f"not owned by current user: {path}", 1)
        mode = stat.S_IMODE(st.st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            _die(f"group/world permissions on {path}", 1)
    elif must_exist:
        _die(f"missing: {path}", 1)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    if path.is_symlink():
        _die(f"refusing symlinked dir: {path}", 1)
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode):
        _die(f"not a directory: {path}", 1)
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        _die(f"dir not owned by current user: {path}", 1)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _acquire_lock(lock_path: Path, timeout: float = 30.0):
    if fcntl is None and msvcrt is None:
        _die("no kernel cross-process lock primitive available", 1)
    _ensure_dir(lock_path.parent)
    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")
        try:
            lock_path.chmod(0o600)
        except OSError:
            pass
    fh = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
    deadline = time.monotonic() + timeout
    while True:
        try:
            if fcntl:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return fh
        except (BlockingIOError, OSError, PermissionError):
            if time.monotonic() >= deadline:
                fh.close()
                _die("lock timeout", 1)
            time.sleep(0.05)


def _release_lock(fh) -> None:
    try:
        if fcntl:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        elif msvcrt:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    if not args.yes:
        _die("--yes is required", 2)

    root = Path(args.root)
    backup_dir = Path(args.backup_dir)
    _check_abs(root, "--root")
    _check_abs(backup_dir, "--backup-dir")
    _ensure_dir(root)
    _ensure_dir(backup_dir)

    marker = root / "shared" / "anthropic_pool_scope.json"
    auth = root / "auth.json"
    lock_path = auth.with_suffix(".lock")

    # Snapshot auth bytes before any mutation for identity proof.
    auth_before = None
    if auth.exists() or auth.is_symlink():
        _safe_regular(auth, must_exist=True)
        auth_before = auth.read_bytes()

    if not marker.exists() and not marker.is_symlink():
        print("no marker present; nothing to do")
        return 0

    _safe_regular(marker, must_exist=True)
    fh = _acquire_lock(lock_path)
    try:
        # Re-check under lock
        if not marker.exists() and not marker.is_symlink():
            print("marker already removed")
            return 0
        _safe_regular(marker, must_exist=True)
        st = marker.lstat()
        raw = marker.read_bytes()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = backup_dir / f"{ts}-{os.getpid()}-{uuid.uuid4().hex[:8]}-anthropic-scope.json"
        if backup.exists():
            _die(f"backup already exists: {backup}", 1)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(backup), flags, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        # Remove marker by verified inode
        if marker.lstat().st_ino != st.st_ino:
            _die("marker inode changed under lock", 1)
        marker.unlink()
        _fsync_dir(marker.parent)
    finally:
        _release_lock(fh)

    if auth_before is not None:
        after = auth.read_bytes()
        if after != auth_before:
            _die("root auth.json changed unexpectedly", 1)
    print(f"marker removed; forensic backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
