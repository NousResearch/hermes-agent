"""Which profile homes THIS process ticks, and who owns the cron tick for a given home.

One gateway process per host multiplexes every profile, so "does a gateway own cron for profile
X" stopped being answerable from the launch home alone: ``gateway.status.owns_gateway_runtime_lock``
is a process-global boolean set once at boot, and every path resolver behind
``is_gateway_runtime_lock_active`` goes through ``get_process_hermes_home()``, which ignores the
per-tick ``_profile_cron_scope`` override. Both therefore answer about the LAUNCH home while the
ticker is scoped to some other profile.

The two predicates here keep those questions apart:

* :func:`owns_cron_tick_for` — this process is the host gateway AND ticks that home.
* :func:`live_gateway_ticking` — a DIFFERENT, live host gateway holds the runtime lock and its
  published served set covers that home.

Every probe failure answers "no": ownership claims are certainties, never guesses.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

from hermes_constants import get_hermes_home, hermes_home_key

# Home key -> home path for every profile this process ticks, republished each ticker cycle so a
# profile created or tombstoned mid-run is reflected without a restart.
_ticked_homes: dict[str, Path] = {}
# Home key -> resolved cron store, taken while the home exists: a departed profile's home may be
# gone by the time its store state is dropped, and a symlinked cron/ resolves elsewhere.
_ticked_stores: dict[str, str] = {}
_ticked_lock = threading.Lock()


def register_ticked_homes(homes) -> None:
    """Publish the set of profile homes this process's cron ticker owns this cycle.

    A home that leaves the set also loses its cron worker pool: pools are per home and otherwise
    live until process exit, so serving (or churning) many profiles leaks one ThreadPoolExecutor
    and its worker threads per home ever ticked.
    """
    from cron.store_health import forget_stores, store_key

    resolved, stores = {}, {}
    for home in homes:
        path = Path(home)
        key = hermes_home_key(path)
        resolved[key] = path
        stores[key] = store_key(path)
    with _ticked_lock:
        departed = set(_ticked_homes) - set(resolved)
        # A store another still-ticked profile reaches (cron/ symlinked to one place) keeps its state.
        departed_stores = {_ticked_stores[k] for k in departed if k in _ticked_stores} - set(stores.values())
        _ticked_homes.clear()
        _ticked_homes.update(resolved)
        _ticked_stores.clear()
        _ticked_stores.update(stores)
    if departed:
        # Late import: cron.scheduler imports this module.
        from cron.scheduler import discard_parallel_pools
        discard_parallel_pools(departed)
        forget_stores(departed_stores)


def ticked_homes() -> dict:
    """Snapshot of ``home key -> home`` for the profiles this process ticks."""
    with _ticked_lock:
        return dict(_ticked_homes)


def serves_profile(home: Optional[Union[Path, str]] = None) -> bool:
    """True when THIS process's cron ticker owns ``home`` (default: the active cron scope)."""
    key = hermes_home_key(home if home is not None else get_hermes_home())
    with _ticked_lock:
        return key in _ticked_homes


def owns_cron_tick_for(home: Optional[Union[Path, str]] = None) -> bool:
    """True when this process is the host gateway multiplexer AND ticks ``home``.

    The runtime lock alone is not enough: it proves only that this process is the host gateway,
    not that its ticker visits the profile whose store is being ticked right now.
    """
    try:
        from gateway import status as gateway_status

        if not gateway_status.owns_gateway_runtime_lock():
            return False
    except Exception:
        return False
    return serves_profile(home)


def record_serves_profile(record: Any, home: Optional[Union[Path, str]] = None) -> bool:
    """True when a gateway runtime-status ``record``'s served set covers ``home``.

    The record is read from the holder's own home, so it always proves that home — that is the
    pre-multiplex meaning, and a single-profile gateway publishes an EMPTY ``served_profiles``.
    Other profiles need an explicit entry in the published served set.
    """
    if not isinstance(record, dict):
        return False
    from gateway.status import (
        _get_process_hermes_home, _profile_label_for_home, _same_hermes_home)

    try:
        target = Path(home) if home is not None else get_hermes_home()
        record_home = record.get("hermes_home")
        if not isinstance(record_home, str) or not record_home.strip():
            record_home = _get_process_hermes_home()
        if _same_hermes_home(record_home, target):
            return True
    except Exception:
        return False
    served = record.get("served_profiles")
    if not isinstance(served, list):
        return False
    label = _profile_label_for_home(target)
    return label is not None and label in served


def live_gateway_ticking(home: Optional[Union[Path, str]] = None) -> Optional[dict]:
    """Runtime-status record of ANOTHER live host gateway that ticks ``home``, else None.

    ``None`` covers every "we cannot prove it" case: this process holds the lock itself, no lock
    is held, the record does not belong to the lock holder, its heartbeat is stale, or its served
    set does not name ``home``.
    """
    try:
        from gateway import status as gateway_status

        if gateway_status.owns_gateway_runtime_lock():
            return None
        if not gateway_status.is_gateway_runtime_lock_active():
            return None
        holder_pid = gateway_status.get_running_pid(cleanup_stale=False)
        record = gateway_status.read_runtime_status()
        if (
            holder_pid is None
            or not isinstance(record, dict)
            or record.get("pid") != holder_pid
            or gateway_status.runtime_status_is_stale(record)
        ):
            return None
    except Exception:
        return None
    return record if record_serves_profile(record, home) else None


def _claim_owner_is_dead(claim: dict[str, Any]) -> bool:
    """True when the claim's ``by`` names a process on THIS host that provably no longer exists.
    ``_machine_id()`` stamps ``host:pid[:token]``; a foreign host, an explicit HERMES_MACHINE_ID,
    or any liveness-probe failure returns False (fail safe: only a proven death shortens the TTL)."""
    parts = str(claim.get("by") or "").split(":")
    if len(parts) < 2 or not parts[1].isdigit():
        return False
    try:
        import socket
        if parts[0] != socket.gethostname():
            return False
        from gateway.status import _pid_exists
        return not _pid_exists(int(parts[1]))
    except Exception:
        return False
