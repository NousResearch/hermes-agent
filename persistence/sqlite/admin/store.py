"""``StoreAdmin`` on one SQLite ``state.db``: doctor probes, journal mode, schema and FTS repair, VACUUM, backup and
restore, and the path-targeted verify, recovery and ledger probes."""

from __future__ import annotations

from ._backup import BackupAdmin
from ._doctor import DoctorAdmin
from ._maintenance import MaintenanceAdmin

__all__ = ["SQLiteStoreAdmin"]


class SQLiteStoreAdmin(DoctorAdmin, MaintenanceAdmin, BackupAdmin):
    """Every ``StoreAdmin`` operation on the one file the handle was opened for."""
