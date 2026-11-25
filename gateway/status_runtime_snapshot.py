"""Read live runtime state before its asynchronous disk writer has finished."""

from typing import Optional


def primary_fatal_platforms_snapshot() -> Optional[set[str]]:
    """Primary platforms currently marked fatal, including writes still queued to disk."""
    from gateway import status

    with status._runtime_status_state_lock:
        state = status._runtime_status_state
        if state is None:
            return None
        platforms = state.get("platforms", {})
        if not isinstance(platforms, dict):
            return None
        return {name for name, entry in platforms.items()
                if isinstance(name, str) and ":" not in name
                and isinstance(entry, dict) and entry.get("state") == "fatal"}
