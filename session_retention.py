"""Post-commit work of the session delete, prune and archive verbs, and the automatic archive and prune passes, over the persistence contracts.

A delete or prune verb returns the ids it removed; its caller runs ``after_sessions_removed`` right after the verb commits: the ``on_session_delete`` plugin hook, then retired-media collection, then transcript-file removal. An archive that hid rows runs ``emit_session_deleted`` with ``reason="archive"``. The plugin dispatch, the media release and the spillover reference scan are application services, imported where they run.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

from hermes_startup_watchdog import report_startup_progress
from persistence.core.admin import AdminUnsupported, StoreAdmin
from persistence.housekeeping.maintenance import AUTO_PRUNE_STALE_OPEN_SOURCES, PruneCriteria

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

#: The automatic VACUUM runs only when more than this fraction of the store's pages is free.
AUTO_VACUUM_MIN_FREELIST_RATIO = 0.25


def emit_session_deleted(store_home, deleted_ids, *, reason, primary=None, sessions_dir=None):
    """Fire ``on_session_delete`` after a committed delete or deliberate archive.

    Call right after the write commits (before follow-up cleanup that can raise), with the ids it removed or, for ``reason="archive"``, hid. *primary* is the session the caller asked for; without it the first id stands in (bulk sweeps). ``deleted_ids`` rides along when more than one row went. Dispatch runs under the store's own profile home *store_home*, so a process touching another profile's store reaches THAT profile's plugins. Late-bound and fail-open: never breaks the delete.
    """
    ids = sorted({sid for sid in (deleted_ids or ()) if isinstance(sid, str) and sid})
    if not ids:
        return
    try:
        from hermes_cli.lifecycle import invoke_hook
        from hermes_constants import (
            get_hermes_home, profile_name_for_home, reset_hermes_home_override, set_hermes_home_override,
        )

        home = os.path.abspath(os.fspath(store_home))
        profile = profile_name_for_home(home)
        payload: dict = {"session_id": primary if primary in ids else ids[0], "reason": reason, "profile": profile}
        if len(ids) > 1:
            payload["deleted_ids"] = ids
        if sessions_dir is not None:
            payload["sessions_dir"] = str(sessions_dir)
        rebind = profile is not None and os.path.realpath(home) != os.path.realpath(get_hermes_home())
        token = set_hermes_home_override(home) if rebind else None
        try:
            invoke_hook("on_session_delete", **payload)
        finally:
            if token is not None:
                reset_hermes_home_override(token)
    except Exception:
        logger.warning("on_session_delete dispatch failed", exc_info=True)


def remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
    """Remove ``<id>.json``/``.jsonl``, the legacy ``session_<id>.json`` snapshot, and gateway ``request_dump_<id>_*.json``; OSError is swallowed so a filesystem hiccup never blocks a store operation. Every historical writer name is swept because a deleted session's snapshot can carry plaintext secrets."""
    if sessions_dir is None:
        return
    targets = [sessions_dir / f"{session_id}{suffix}" for suffix in (".json", ".jsonl")]
    targets.append(sessions_dir / f"session_{session_id}.json")
    try:
        # glob.escape: a session id carrying ``[`` / ``?`` / ``*`` is a PATTERN otherwise, so the
        # dump sweep either matches nothing or matches another session's files.
        targets.extend(sessions_dir.glob(f"request_dump_{glob.escape(session_id)}_*.json"))
    except OSError:
        pass
    for p in targets:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def collect_retired_spillover(store, store_home) -> None:
    """Unlink retired spillover archives no surviving message references (a branch or compression sibling sharing the pointer keeps it until it dies too), then drop the collected candidates. Unknown reference set: keep files and candidates."""
    from tools.tool_result_storage import referenced_spillover_names
    rows = store.admissions.retired_candidates("spillover")
    spill_dir = Path(store_home) / "cache" / "spillover"
    if not rows or (referenced := referenced_spillover_names(spill_dir)) is None:
        return
    done = []
    for key, name in rows:
        try:
            if name not in referenced:
                (spill_dir / name).unlink(missing_ok=True)
        except OSError:
            continue  # stays a candidate for the next delete/prune pass
        done.append(key)
    store.admissions.drop_retired_candidates(done)


def collect_retired_media(store, store_home) -> None:
    """Release retired media candidates no admission or retained transcript row owns, and drop the candidates whose file is gone; spillover candidates first. Synchronous store reads and writes: async callers run it off the event loop."""
    from gateway.session_ingress_media import release_unheld_media
    collect_retired_spillover(store, store_home)
    rows = store.admissions.retired_candidates("media")
    if rows:
        release_unheld_media(store, [reference for _key, reference in rows])
        gone = [key for key, reference in rows if not Path(reference["path"]).exists()]
        if gone:
            store.admissions.drop_retired_candidates(gone)


def after_sessions_removed(store, store_home, removed_ids: Sequence[str], *, reason: str, primary: Optional[str] = None,
                           sessions_dir: Optional[Path] = None) -> None:
    """The work a delete or prune runs after its write commits, in order: the plugin hook over *removed_ids*, retired-media collection (also when nothing was removed), then the transcript files of every removed id under *sessions_dir*."""
    emit_session_deleted(store_home, removed_ids, reason=reason, primary=primary, sessions_dir=sessions_dir)
    collect_retired_media(store, store_home)
    for session_id in removed_ids:
        remove_session_files(sessions_dir, session_id)


def archive_sessions(store, store_home, older_than_days: Optional[float] = None, source: Optional[str] = None, *,
                     criteria: PruneCriteria = PruneCriteria()) -> int:
    """Bulk soft-hide with the prune filter surface, archiving each match's lineage as a unit; idempotent. Returns the matches. Unarchived rows only unless ``criteria.archived`` says otherwise. A lineage is matched through its TIP only: an old compression ancestor never qualifies on its own age, or the fan-out would hide an open, recently active continuation."""
    criteria = replace(criteria, archived=False if criteria.archived is None else criteria.archived, lineage_tips_only=True)
    rows = store.maintenance.list_prune_candidates(older_than_days, source, criteria=criteria)
    for row in rows:
        change = store.sessions.set_archived(row.session_id, True)
        emit_session_deleted(store_home, change.archived_ids, reason="archive", primary=row.session_id)
    return len(rows)


def run_auto_archive(store, idle_days: float = 3, min_interval_hours: int = 24, exclude_pinned: bool = True) -> dict[str, Any]:
    """Idempotent, non-destructive auto-archive of sessions idle for ``idle_days``; the ``last_auto_archive`` state key gates runs within ``min_interval_hours``. Never raises."""
    result: dict[str, Any] = {"skipped": False, "archived": 0}
    try:
        now = time.time()
        try:
            last = float(store.state_meta.get("last_auto_archive") or 0.0)
        except (TypeError, ValueError):
            last = 0.0  # corrupt meta; treat as no prior run
        if last and now - last < min_interval_hours * 3600:
            result["skipped"] = True
            return result
        # Startup-watchdog lease: the archive sweep is I/O-bound (near-zero CPU),
        # which the watchdog's CPU fallback misreads as a parked deadlock.
        # No-op when the watchdog is not armed; never raises.
        report_startup_progress(900.0, phase="state_db_auto_archive")
        ids = () if idle_days is None else store.maintenance.list_idle_archive_candidates(idle_days, exclude_pinned=exclude_pinned)
        for session_id in ids:
            # Sweep provenance: a later compression/resume of this lineage un-hides it.
            store.maintenance.auto_archive_lineage(session_id)
        archived = result["archived"] = len(ids)
        # Record even a zero-archive run so we don't re-sweep every call.
        store.state_meta.set("last_auto_archive", str(now))
        if archived > 0:
            logger.info(
                "state.db auto-archive: archived %d session(s) idle >= %s days", archived, idle_days,
            )
    except Exception as exc:  # health: allow BLE001 -- the automatic archive never raises; the error rides in the result
        logger.warning("state.db auto-archive failed: %s", exc)
        result["error"] = str(exc)
    return result


def _seconds_since(now: float, raw) -> Optional[float]:
    """Age of a state key timestamp; None when unset or corrupt (= no prior run)."""
    try:
        return now - float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def run_auto_prune_and_vacuum(
    store, admin: StoreAdmin | AdminUnsupported | None, *, store_home, media_root: str, retention_days: int = 90, min_interval_hours: int = 24, vacuum: bool = True,
    sessions_dir: Optional[Path] = None, min_vacuum_interval_days: int = 30,
    min_vacuum_freelist_ratio: float = AUTO_VACUUM_MIN_FREELIST_RATIO,
) -> dict[str, Any]:
    """Idempotent startup auto-maintenance (never raises): prune inactive sessions, reap stale open state-owned rows, optional VACUUM.

    Runs at most once per ``min_interval_hours`` (the ``last_auto_prune`` state key). Stale-open reconciliation: cron, kanban, subagent and one-shot CLI rows never set ``ended_at`` when their process dies and prune only deletes ended rows, so after pruning, open rows from ``AUTO_PRUNE_STALE_OPEN_SOURCES`` older than ``retention_days`` are closed (``startup_orphan_reap``); they stay resumable and age from their close. Transcript files of pruned sessions under *sessions_dir* are removed in the same sweep. Messaging and UI sources are never touched here.

    *admin* is the store's physical admin handle, or ``AdminUnsupported`` (or None) when the backend has none. With it, the pass holds the auto-maintenance lock (held elsewhere: skipped) and may VACUUM: only when rows were freed, ``min_vacuum_interval_days`` passed since ``last_vacuum``, no other holder blocks it and the freelist ratio is above ``min_vacuum_freelist_ratio``. Without it, the lock and VACUUM steps are skipped. Returns ``{"skipped", "pruned", "closed", "vacuumed"}`` plus ``"freelist_ratio"`` when a VACUUM was considered, ``"vacuum_skipped_holders"`` when holders blocked it and ``"error"`` on failure.
    """
    result: dict[str, Any] = {"skipped": False, "pruned": 0, "closed": 0, "vacuumed": False}
    if retention_days is None or retention_days < 0:
        # A negative retention would build a future cutoff and match every ended
        # session; auto_prune=false is the disable switch, not a negative bound.
        logger.warning(
            "state.db auto-maintenance skipped: sessions.retention_days=%r is outside the allowed "
            "range (a whole number of days >= 0); set sessions.auto_prune: false to disable pruning",
            retention_days)
        result["skipped"] = True
        return result
    if isinstance(admin, AdminUnsupported):
        admin = None
    maintenance_lock = None
    if admin is not None:
        maintenance_lock = admin.try_acquire_auto_maintenance_lock()
        if maintenance_lock is None:
            result["skipped"] = True
            return result
    try:
        now = time.time()
        since_prune = _seconds_since(now, store.state_meta.get("last_auto_prune"))
        if since_prune is not None and since_prune < min_interval_hours * 3600:
            result["skipped"] = True
            return result
        # Prune first: orphans closed below get a full retention window.
        # Startup-watchdog leases: each long step is I/O-bound (near-zero CPU), which the
        # watchdog's CPU fallback misreads as a parked deadlock; the lease is renewed per step.
        report_startup_progress(900.0, phase="state_db_auto_prune")
        outcome = store.maintenance.prune_sessions(
            older_than_days=retention_days, exclude_active_write_guards=True, media_root=media_root)
        after_sessions_removed(store, store_home, outcome.deleted_ids, reason="prune", sessions_dir=sessions_dir)
        result["pruned"] = pruned = outcome.count
        report_startup_progress(900.0, phase="state_db_auto_sweep")
        closed = store.maintenance.sweep_orphaned_sessions(
            max_idle_seconds=float(retention_days) * 86400.0,
            sources=AUTO_PRUNE_STALE_OPEN_SOURCES, exclude_pinned=True,
            respect_gateway_heartbeats=False,  # state-owned lifecycles, not gateway heartbeats
        )
        result["closed"] = len(closed)
        # VACUUM only if rows were freed, the time throttle passed AND the
        # freelist ratio passed: it holds an exclusive lock for a full rewrite.
        since_vacuum = _seconds_since(now, store.state_meta.get("last_vacuum"))
        vacuum_due = since_vacuum is None or since_vacuum >= min_vacuum_interval_days * 86400
        if admin is not None and vacuum and pruned > 0 and vacuum_due:
            result["freelist_ratio"] = ratio = admin.freelist_ratio()
            # VACUUM retires the store generation another writer still holds, which every agent
            # then refuses turns in: automatic maintenance only ever SKIPS while a holder exists.
            holders = admin.vacuum_blockers()
            if holders:
                result["vacuum_skipped_holders"] = len(holders)
                logger.debug(
                    "state.db auto-maintenance: skipping VACUUM, %d other process(es) hold the "
                    "store or a WAL sidecar (%s)",
                    len(holders), ", ".join(f"{pid}:{target}" for pid, target in holders[:3]))
            elif ratio is None or ratio > min_vacuum_freelist_ratio:
                try:
                    # VACUUM rewrites every page with ~zero CPU: renew the lease here so
                    # a multi-minute rewrite on a large store never outlives the clamp.
                    report_startup_progress(900.0, phase="state_db_auto_vacuum")
                    admin.vacuum()
                    result["vacuumed"] = True
                    store.state_meta.set("last_vacuum", str(now))
                except Exception as exc:  # health: allow BLE001 -- a failed VACUUM leaves the prune result standing
                    logger.warning("state.db VACUUM failed: %s", exc)
            else:
                logger.debug("state.db auto-maintenance: skipping VACUUM, only "
                             "%.1f%% of pages reclaimable (threshold %.0f%%)",
                             ratio * 100.0, min_vacuum_freelist_ratio * 100.0)
        # Record even when pruned == 0 so the throttle holds.
        store.state_meta.set("last_auto_prune", str(now))
        if closed or pruned > 0:
            logger.info("state.db auto-maintenance: closed %d stale open session(s), "
                        "pruned %d session(s) inactive for %d days%s",
                        len(closed), pruned, retention_days, " + VACUUM" if result["vacuumed"] else "")
    except Exception as exc:  # health: allow BLE001 -- maintenance never blocks startup; the error rides in the result
        # Maintenance must never block startup.
        logger.warning("state.db auto-maintenance failed: %s", exc)
        result["error"] = str(exc)
    finally:
        if maintenance_lock is not None:
            admin.release_auto_maintenance_lock(maintenance_lock)
    return result
