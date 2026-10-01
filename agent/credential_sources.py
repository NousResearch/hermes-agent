"""External plugin registration compatibility; removal belongs to auth."""
# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# External plugins historically import register here. No internal consumers.
from auth.source_removal import register  # noqa: F401
# ---- END PLUGIN-COMPAT ----
