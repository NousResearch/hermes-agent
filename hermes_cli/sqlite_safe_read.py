"""Compatibility-only surface for the retired SQLite safe-read owner.

First-party code must import :mod:`storage.sqlite_safe_read` directly.
"""

# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"
# ---- END PLUGIN-COMPAT ----
