"""Process-consumed plugin activation choices, not a plugin health report.

The first discovery/mount allow/deny lists survive config saves and new RPC clients.
A force-discovery is not a backend restart: already-mounted API routes may remain.
"""
from threading import Lock

from hermes_constants import get_process_hermes_home, hermes_home_key

_snapshots: dict[tuple[str, str], tuple[frozenset[str], frozenset[str]]] = {}
_lock = Lock()


def record_plugin_config(consumer: str, enabled, disabled) -> None:
    """Capture the actual gate inputs once per process consumer and owning home."""
    home = hermes_home_key()
    with _lock:
        _snapshots.setdefault((home, consumer), (frozenset(enabled or ()), frozenset(disabled)))


def annotate_restart_state(rows: list[dict]) -> bool | None:
    """Add row metadata and return the aggregate; another profile's runtime is unknown.

    ``status`` remains the saved choice. Restart metadata only compares activation
    configuration, not whether plugin imports, dependencies or hooks succeeded.
    """
    home = hermes_home_key()
    with _lock:
        snapshots = [state for (owner, _), state in _snapshots.items() if owner == home]
    known = home == hermes_home_key(get_process_hermes_home()) and bool(snapshots)
    for row in rows:
        # Portable skills and MCP configs also live in the process-cached manager;
        # a new session does not guarantee their saved activation is consumed.
        names = {row['name'], row['key']}
        row['applies_on'] = 'backend_restart'
        row['restart_required'] = (
            any(((bool(names & enabled) or row['default_enabled']) and not bool(names & disabled))
                != (row['status'] == 'enabled') for enabled, disabled in snapshots)
            if known else None
        )
    states = [row['restart_required'] for row in rows]
    return True if True in states else None if None in states else False
