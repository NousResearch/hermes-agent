"""Attach originating-session progress checks while tracked work is active.

The heartbeat poller calls :func:`reconcile_active_work_watches` before it
restores or fires watches. Settled work is cleared first, so a restart cannot
deliver a stale check. User-authored heartbeats are left in place.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("gateway.run")

_SKIP_PLATFORMS = frozenset({
    "local", "webhook", "msgraph_webhook", "api_server", "a2a", "cron",
})


def _continuation_id(store: Any, session_id: str, entries: list) -> str:
    """Live chat id for work still recorded on a compressed parent.

    Prefer the store's compression tip. A republished routing entry keeps
    ``prev_session_id`` when the tip lookup is not on the test double.
    """
    resolver = getattr(store, "_compression_tip_for_session_id", None)
    if callable(resolver):
        try:
            tip = resolver(session_id)
        except Exception:
            logger.debug(
                "active-work progress: compression tip failed for %s",
                session_id,
                exc_info=True,
            )
            tip = None
        if tip:
            return str(tip)
    for entry in entries:
        prev = getattr(entry, "prev_session_id", None)
        current = getattr(entry, "session_id", None)
        if prev == session_id and current:
            return current
    return session_id


def _deliverable(entry: Any) -> bool:
    origin = getattr(entry, "origin", None)
    if origin is None or getattr(entry, "suspended", False):
        return False
    platform = getattr(origin, "platform", None)
    name = getattr(platform, "value", None) or str(platform or "")
    if name in _SKIP_PLATFORMS or not getattr(origin, "chat_id", None):
        return False
    return bool(getattr(entry, "session_id", None) and getattr(entry, "session_key", None))


def _reconcile_sync(runner: Any, home: Any) -> tuple[list[tuple], list[str]]:
    from gateway.run import _profile_runtime_scope
    from hermes_cli.active_work_progress import (
        collect_active_session_ids,
        reconcile_active_work_result,
    )

    watches: list[tuple] = []
    drop_keys: list[str] = []
    with _profile_runtime_scope(home):
        active, scans_complete = collect_active_session_ids()
        by_id: dict[str, Any] = {}
        store = getattr(runner, "session_store", None)
        entries: list = []
        store_ok = store is not None
        if store is not None:
            try:
                entries = list(store.list_sessions())
            except Exception:
                logger.debug("active-work progress: session list failed", exc_info=True)
                store_ok = False
        for entry in entries:
            sid = getattr(entry, "session_id", None)
            if sid and sid not in by_id:
                by_id[sid] = entry
        if store_ok:
            live: set[str] = set()
            for sid in active:
                tip = _continuation_id(store, sid, entries)
                entry = by_id.get(tip)
                # No routing row yet: keep the check on the tip, but do not
                # invent a watch. A known non-chat origin is not persisted, so
                # the following restore cannot revive it.
                if entry is None or _deliverable(entry):
                    live.add(tip)
            result = reconcile_active_work_result(live, scans_complete=scans_complete)
        else:
            result = reconcile_active_work_result(set(), scans_complete=False)
        for sid, action in result.actions.items():
            if action == "preserved-user":
                continue
            entry = by_id.get(sid)
            if entry is not None and _deliverable(entry):
                watches.append((entry.session_key, entry.origin, entry.session_id))
        for sid in result.cleared:
            entry = by_id.get(sid)
            if entry is not None and getattr(entry, "session_key", None):
                drop_keys.append(entry.session_key)
    return watches, drop_keys


async def reconcile_active_work_watches(runner: Any) -> None:
    """Reconcile managed progress checks, then let the caller restore and poll."""
    from gateway.run_heartbeat_restore import _watched_homes
    from hermes_constants import get_hermes_home

    store = getattr(runner, "session_store", None)
    home = getattr(store, "_routing_home", None) or get_hermes_home()
    try:
        homes = _watched_homes(runner, home)
    except Exception:
        logger.debug("active-work progress: home scan failed", exc_info=True)
        return
    offload = getattr(runner, "_run_in_executor_with_context", None)
    for watched_home in homes:
        try:
            if offload is None:
                watches, drop_keys = _reconcile_sync(runner, watched_home)
            else:
                watches, drop_keys = await offload(_reconcile_sync, runner, watched_home)
        except Exception:
            logger.debug("active-work progress: reconcile failed for %s", watched_home, exc_info=True)
            continue
        unregister = getattr(runner, "_unregister_heartbeat_watch", None)
        register = getattr(runner, "_register_heartbeat_watch", None)
        if unregister is not None:
            for key in drop_keys:
                unregister(key)
        if register is not None:
            for key, source, session_id in watches:
                register(key, source, session_id)
