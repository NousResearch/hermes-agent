"""Leaf constants shared by ``gateway/run.py`` and its ``run_*`` mixin modules.

Kept import-cycle free (imports nothing from ``gateway.run``) because these values
are used as default-argument sentinels, which must resolve at ``def`` time.
"""

from typing import Any

# Sentinel for "caller did not pass metadata" vs "caller passed None".
_UNSET = object()


def _csv_or_list_to_set(raw: Any) -> set[str]:
    """Normalize a config list or comma-separated scalar into a string set."""
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}
