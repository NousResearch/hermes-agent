"""Read-only compatibility with pending accounting markers from the old store."""

import hashlib
import os
import stat

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
