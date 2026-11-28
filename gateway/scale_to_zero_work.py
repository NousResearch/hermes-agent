"""Read existing durable work leases before suspending a shared machine.

The dashboard can execute turns in a different process from the gateway. Client
heartbeats and the gateway's in-memory registries cannot account for detached work.
These reads never create a store or reclaim another process's work.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path


def _delegation_owner_may_be_live(pid, started) -> bool:
    from gateway.status import (
        _pid_exists,
        get_process_start_time,
        start_time_fingerprints_match,
    )

    # Legacy records may lack an identity. Uncertainty must not authorize suspend.
    if not pid:
        return True
    if not _pid_exists(int(pid)):
        return False
    current = get_process_start_time(int(pid))
    return (
        started is None
        or current is None
        or start_time_fingerprints_match(started, current)
    )


def profile_has_durable_work(home: Path, *, now: float) -> bool:
    from hermes_state import _compression_lock_holder_process_is_dead
    from hermes_state_holders import read_only_db_uri

    path = home / "state.db"
    try:
        path.stat()
    except FileNotFoundError:
        return False
    with closing(
        sqlite3.connect(read_only_db_uri(path), uri=True, timeout=0.05)
    ) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('session_turn_leases', 'async_delegations')"
            )
        }
        if "session_turn_leases" in tables:
            holders = conn.execute(
                "SELECT holder FROM session_turn_leases WHERE expires_at > ?", (now,)
            )
            if any(
                not _compression_lock_holder_process_is_dead(holder)
                for (holder,) in holders
            ):
                return True
        if "async_delegations" in tables:
            owners = conn.execute(
                "SELECT owner_pid, owner_started_at FROM async_delegations "
                "WHERE state IN ('running', 'stalling', 'finalizing')"
            )
            if any(
                _delegation_owner_may_be_live(pid, started) for pid, started in owners
            ):
                return True
    return False


def has_durable_work() -> bool:
    from hermes_cli.profiles import profiles_to_serve
    from hermes_constants import get_process_hermes_home

    # Suspension freezes every process, including work in a standalone or parked
    # profile's dashboard. Enumerate this install, ignoring any request's scope.
    homes = {
        home
        for _name, home in profiles_to_serve(
            True, include_standalone=True, include_parked=True
        )
    }
    homes.add(get_process_hermes_home())
    now = time.time()
    return any(profile_has_durable_work(home, now=now) for home in homes)
