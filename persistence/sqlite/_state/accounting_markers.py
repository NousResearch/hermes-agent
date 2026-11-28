"""Physical-store witnesses for queued accounting and interrupted transfers."""

import hashlib
import os
import stat
import uuid

def _accounting_marker_identity(db):
    """Resolve a store's marker identity without creating a marker directory."""
    from pathlib import Path
    from persistence.sqlite._state.lineage_fence import _managed_root_and_current, SessionLineageInventoryError
    try:
        st = os.stat(db.db_path, follow_symlinks=False)
        if (not stat.S_ISREG(st.st_mode) or db._db_file_identity != (st.st_dev, st.st_ino)
                or db._db_file_was_replaced()):
            raise SessionLineageInventoryError("Accounting store identity is unavailable or changed")
        if st.st_nlink != 1:
            raise SessionLineageInventoryError("Hardlinked accounting stores cannot be inventoried safely")
        managed = _managed_root_and_current(db.db_path)
        root = managed[0] if managed else Path(db.db_path).resolve().parent
        physical = hashlib.sha256(f"{st.st_dev}:{st.st_ino}".encode()).hexdigest()
        location = hashlib.sha256(str(Path(db.db_path).resolve()).encode()).hexdigest()
        return root, bool(managed), f"usage-{location}-{physical}-"
    except OSError as exc:
        raise SessionLineageInventoryError("Cannot inventory accounting markers") from exc


def _legacy_accounting_marker_present(db, prefix):
    """Read legacy managed-home markers without creating, unlocking or deleting them."""
    from persistence.sqlite._state.lineage_fence import _managed_root_and_current, _validate_lineage_parent, SessionLineageInventoryError
    managed = _managed_root_and_current(db.db_path)
    if managed is None:
        return False
    legacy_root = managed[0] / ".session-lineage-locks"
    directory = legacy_root / "accounting"
    try:
        for path in (legacy_root, directory):
            try:
                st = path.lstat()
            except FileNotFoundError:
                return False
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise SessionLineageInventoryError("Unsafe legacy accounting marker directory")
            _validate_lineage_parent(path, st)
        _, location, physical, _ = prefix.split("-", 3)
        with os.scandir(directory) as scan:
            for count, entry in enumerate(scan, 1):
                if count > 8192:
                    raise SessionLineageInventoryError("Legacy accounting marker inventory exceeds bound")
                parts = entry.name.split("-", 3)
                if len(parts) == 4 and parts[0] == "usage" and (parts[1] == location or parts[2] == physical):
                    return True
        return False
    except OSError as exc:
        raise SessionLineageInventoryError("Cannot inspect legacy accounting marker inventory") from exc


def require_legacy_accounting_clear(db):
    from persistence.sqlite._state.lineage_fence import _managed_root_and_current, SessionLineageInventoryError
    if _managed_root_and_current(db.db_path) is None:
        return
    _, _, prefix = _accounting_marker_identity(db)
    if _legacy_accounting_marker_present(db, prefix):
        raise SessionLineageInventoryError("Unresolved legacy accounting marker; inspect before accepting usage")


def _token_marker_prefix(self):
    """Locate this physical store's cross-process pending-accounting markers."""
    from persistence.sqlite._state.lineage_fence import _lineage_lock_directory, SessionLineageInventoryError
    try:
        root, managed, prefix = _accounting_marker_identity(self)
        if managed:
            lock_root = _lineage_lock_directory(root, create=True)
        else:
            lock_root = root / ".session-lineage-locks"
            lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lst = lock_root.lstat()
        if not stat.S_ISDIR(lst.st_mode) or stat.S_ISLNK(lst.st_mode):
            raise SessionLineageInventoryError("Unsafe lineage lock directory")
        directory = lock_root / "accounting"
        directory.mkdir(mode=0o700, exist_ok=True)
        dst = directory.lstat()
        if not stat.S_ISDIR(dst.st_mode) or stat.S_ISLNK(dst.st_mode):
            raise SessionLineageInventoryError("Unsafe accounting marker directory")
        return directory, prefix
    except SessionLineageInventoryError:
        raise
    except OSError as exc:
        raise SessionLineageInventoryError("Cannot inventory accounting markers") from exc

def hold_pending_usage(self):
    """Hold a raw advisory marker before accepting any volatile delta."""
    from persistence.sqlite._state.lineage_fence import _advisory_lock_file, SessionLineageInventoryError
    from persistence.sqlite._state.lineage_fence import _managed_root_and_current

    if _managed_root_and_current(self.db_path) is None:
        return
    directory, prefix = _token_marker_prefix(self)
    if _legacy_accounting_marker_present(self, prefix):
        raise SessionLineageInventoryError("Unresolved legacy accounting marker; inspect before accepting usage")
    if self._token_pending_marker is not None:
        if self._token_pending_marker_path is None:
            raise SessionLineageInventoryError("Accounting marker release is unresolved")
        if (self._token_pending_marker_path.parent != directory
                or not self._token_pending_marker_path.name.startswith(prefix)):
            raise SessionLineageInventoryError("Accounting marker identity changed")
        try:
            held = self._token_pending_marker_path.lstat()
        except OSError as exc:
            raise SessionLineageInventoryError("Held accounting marker is unavailable") from exc
        if (not stat.S_ISREG(held.st_mode) or (held.st_dev, held.st_ino) != self._token_pending_marker_identity):
            raise SessionLineageInventoryError("Held accounting marker was replaced")
        return
    if self._token_pending_marker_path is None:
        self._token_pending_marker_path = directory / f"{prefix}{os.getpid()}-{uuid.uuid4().hex}.lock"
    elif (self._token_pending_marker_path.parent != directory
          or not self._token_pending_marker_path.name.startswith(prefix)):
        raise SessionLineageInventoryError("Accounting marker identity changed")
    directory_stat = directory.lstat()
    marker = _advisory_lock_file(self._token_pending_marker_path, 0)
    marker_fd = marker.__enter__()
    try:
        if (_token_marker_prefix(self) != (directory, prefix)
                or not os.path.samestat(directory_stat, directory.lstat())):
            raise SessionLineageInventoryError("Accounting store identity changed while accepting")
    except Exception:
        marker.__exit__(None, None, None)
        raise
    self._token_pending_marker = marker
    held = os.fstat(marker_fd)
    self._token_pending_marker_identity = (held.st_dev, held.st_ino)

def release_idle_usage(self):
    """Called under the queue condition only after a known-idle transition."""
    if (self._token_pending_marker is not None and not self._token_queue
            and not self._token_writer_busy and not self._token_writer_uncertain):
        if self._token_pending_marker_path is None:
            raise RuntimeError("Accounting marker release is unresolved")
        # Only the owning handle removes its own known-idle marker while still locked.
        try:
            held = self._token_pending_marker_path.lstat()
        except FileNotFoundError:
            held = None  # no pending work remains; releasing the unlinked lock is safe
        if held is not None:
            if (not stat.S_ISREG(held.st_mode)
                    or (held.st_dev, held.st_ino) != self._token_pending_marker_identity):
                raise RuntimeError("Idle accounting marker was replaced")
            self._token_pending_marker_path.unlink()
        self._token_pending_marker_path = None
        self._token_pending_marker.__exit__(None, None, None)
        self._token_pending_marker = None
        self._token_pending_marker_identity = None

def pending_usage(db):
    """A transfer may proceed only when local and cross-process accounting is empty."""
    from persistence.sqlite._state.lineage_fence import _managed_root_and_current

    with db._token_queue_cond:
        if db._token_queue or db._token_writer_busy or db._token_writer_uncertain or db._token_pending_marker:
            return True
    if _managed_root_and_current(db.db_path) is None:
        return False
    require_legacy_accounting_clear(db)
    directory, prefix = _token_marker_prefix(db)
    _, location, physical, _ = prefix.split("-", 3)
    with os.scandir(directory) as entries:
        for count, entry in enumerate(entries, 1):
            if count > 8192:
                return True
            parts = entry.name.split("-", 3)
            if len(parts) == 4 and parts[0] == "usage" and (parts[1] == location or parts[2] == physical):
                # A marker left by a crashed writer is still unresolved work.
                return True
    return False
