"""Per-profile state.db housekeeping for the gateway: auto-archive, auto-prune and space reclaim.

The gateway never runs a full VACUUM. VACUUM holds the write lock for the whole rewrite (minutes
on a multi-GB store, far longer over NFS): run in the constructor it held the boot before any
adapter connected, so operators turned ``vacuum_after_prune`` off and could then only shrink the
file with a stop. Run on the housekeeping thread it would stall every writer for the same time.
Instead ``vacuum_after_prune`` gives freed pages back in short ``incremental_vacuum`` batches on
the housekeeping tick, interleaved with live writers (``reclaim_free_pages_incrementally``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def housekeeping_state_db_maintenance(
    launch: Optional[tuple[Path, Path]] = None, *, startup: bool = False,
) -> None:
    """One profile's pass; see ``gateway.run._housekeeping_state_db_maintenance``.

    *startup* (the constructor's pass) prunes but never reclaims: the reclaim rides the
    housekeeping thread, once the gateway is serving."""
    from gateway.run import _profile_sessions_dir
    from hermes_cli.config import load_config as _load_full_config
    from hermes_state_registry import acquire, release_or_close
    sess_cfg = (_load_full_config().get("sessions") or {})
    if not (sess_cfg.get("auto_archive", False) or sess_cfg.get("auto_prune", False)):
        return
    vacuum = bool(sess_cfg.get("vacuum_after_prune", True))
    adb = acquire()
    try:
        if sess_cfg.get("auto_archive", False):
            adb.maybe_auto_archive(
                idle_days=float(sess_cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(sess_cfg.get("min_interval_hours", 24)))
        if sess_cfg.get("auto_prune", False):
            adb.maybe_auto_prune_and_vacuum(
                retention_days=int(sess_cfg.get("retention_days", 90)),
                min_interval_hours=int(sess_cfg.get("min_interval_hours", 24)),
                min_vacuum_interval_days=int(sess_cfg.get("min_vacuum_interval_days", 30)),
                vacuum=vacuum, full_vacuum=False,
                sessions_dir=_profile_sessions_dir(launch))
            if vacuum and not startup:
                adb.reclaim_free_pages_incrementally()
    finally:
        release_or_close(adb)
