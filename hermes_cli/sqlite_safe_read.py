"""Compatibility-only surface for retired SQLite safe-read ownership.

Current Hermes code imports :mod:`storage.sqlite_safe_read` directly. This
module remains for the frozen updater ABI and the scheduled plugin constant.
"""

from storage.sqlite_safe_read import (
    LiveConnectionError,
    SQLITE_HEADER_MAGIC,
    has_live_connection,
    offline_file_access,
    read_header_bytes_preopen,
)

# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# SQLITE_HEADER_MAGIC is re-exported from the canonical storage owner above.
# ---- END PLUGIN-COMPAT ----

__all__ = [
    "LiveConnectionError",
    "SQLITE_HEADER_MAGIC",
    "has_live_connection",
    "offline_file_access",
    "read_header_bytes_preopen",
]
