"""Composite worker-fingerprint comparison for the Kanban dispatcher.

Split out of ``hermes_cli.kanban_db_dispatch`` (``_pid_recycled``), which reaches it late-bound.
"""

from __future__ import annotations


def composite_fingerprint_recycled(pid: int, recorded: str) -> bool:
    """True when ``pid`` no longer matches a composite ``<epoch>|<start>`` spawn fingerprint.

    The epoch must match exactly; the start time tolerates the platform's 2-second clock drift
    (``gateway.status.start_time_fingerprints_match``), so a macOS worker is not declared recycled
    because its start time was re-read a second apart. An unreadable or malformed fingerprint is
    treated as foreign: signalling it could hit a stranger.
    """
    from hermes_cli.kanban_db_dispatch import _process_fingerprint

    current = _process_fingerprint(pid)
    if current is None or "|" not in current:
        return True
    recorded_epoch, recorded_start = recorded.split("|", 1)
    current_epoch, current_start = current.split("|", 1)
    if recorded_epoch != current_epoch:
        return True
    from gateway.status import start_time_fingerprints_match

    try:
        return not start_time_fingerprints_match(recorded_start, current_start)
    except (TypeError, ValueError):
        return True
