"""Physical file administration of SQLite stores: ``SQLiteStoreAdmin`` and ``SQLiteProfileRepairAdmin``.

Each handle is bound to one ``state.db`` file. The path-targeted operations (``verify_file``, ``recover_transcripts``,
``has_runtime_ledger_rows``) take the file they address and need no live store.
"""

from __future__ import annotations

from .profile_repair import SQLiteProfileRepairAdmin
from .store import SQLiteStoreAdmin

__all__ = ["SQLiteProfileRepairAdmin", "SQLiteStoreAdmin"]
