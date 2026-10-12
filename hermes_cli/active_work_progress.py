"""Active-work progress checks on the originating session.

Uses the existing session heartbeat. No new config key: the cadence is the
approved 180 seconds. A user-authored heartbeat is never replaced. The check
is removed when the tracked work settles. Terminal Kanban notifications stay
on their own event path and are not delayed by this cadence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from hermes_cli.heartbeat import (
    ACTIVE_WORK_HEARTBEAT_ORIGIN,
    HeartbeatManager,
    HeartbeatState,
    load_heartbeat,
)

logger = logging.getLogger(__name__)

# Approved fleet cadence. Not a config key: gateway_notify_interval is the
# generic "still working" notice and must not be reused for this check.
ACTIVE_WORK_INTERVAL_SECONDS = 180

ACTIVE_WORK_PROMPT = (
    "Check active tracked work that originated in this chat: Kanban tasks, "
    "delegated agents, and background jobs. If something meaningful changed "
    "since the last update in this same chat, send one concise update: what "
    "changed, what is next, and any blocker. If nothing meaningful changed, "
    "reply exactly [SILENT]. Do not invent progress. Do not broadcast to other "
    "chats. Do not include internal child-process or build logs. Do not repeat "
    "a completion or blocker the system already delivered."
)

# Same live set as tools/async_delegation.py _LIVE_STATES. stalling is still
# in flight; omitting it clears the originating check while that agent runs.
_LIVE_DELEGATION_STATES = ("running", "stalling", "finalizing")
_ACTIVE_KANBAN_STATUS = "running"
# Child-process completion chatter stays off. This module never enables
# delegation.surface_child_process_notifications.


def reconcile_active_work(active_session_ids: Iterable[str]) -> dict[str, str]:
    """Install, keep, or clear managed checks. Cleared ids are ``__cleared__``."""
    result = reconcile_active_work_result(active_session_ids)
    actions = dict(result.actions)
    if result.cleared:
        actions["__cleared__"] = ",".join(result.cleared)
    return actions


class ScanResult:
    """One work-source scan. ``ok`` false means the source could not be read."""

    def __init__(self, ids: set[str], ok: bool) -> None:
        self.ids = ids
        self.ok = ok


class ReconcileResult:
    def __init__(self, actions: dict[str, str], cleared: list[str]) -> None:
        self.actions = actions
        self.cleared = cleared


def reconcile_active_work_result(
    active_session_ids: Iterable[str], *, scans_complete: bool = True,
) -> ReconcileResult:
    """Ensure a managed 180s check on each active session; clear the rest.

    An incomplete scan must not look like settled work. Existing managed
    checks stay until every source has been read successfully.
    """
    active = {sid for sid in active_session_ids if sid}
    actions: dict[str, str] = {}
    for sid in sorted(active):
        actions[sid] = HeartbeatManager(sid).ensure_managed(
            ACTIVE_WORK_PROMPT, ACTIVE_WORK_INTERVAL_SECONDS, origin=ACTIVE_WORK_HEARTBEAT_ORIGIN,
        )
    if not scans_complete:
        return ReconcileResult(actions, [])
    cleared: list[str] = []
    for sid in _managed_session_ids():
        if sid in active:
            continue
        if HeartbeatManager(sid).clear_if_origin(ACTIVE_WORK_HEARTBEAT_ORIGIN):
            cleared.append(sid)
    return ReconcileResult(actions, cleared)


def _managed_session_ids() -> list[str]:
    from hermes_cli.heartbeat import _META_PREFIX, _get_session_db

    db = _get_session_db()
    if db is None or not hasattr(db, "list_meta_prefix"):
        return []
    found = []
    try:
        rows = db.list_meta_prefix(_META_PREFIX)
    except Exception:
        logger.debug("active-work progress: heartbeat list failed", exc_info=True)
        return []
    for key, raw in rows:
        try:
            state = HeartbeatState.from_json(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.debug("active-work progress: skipped unreadable heartbeat %s", key, exc_info=True)
            continue
        if state.origin == ACTIVE_WORK_HEARTBEAT_ORIGIN and state.status == "active":
            found.append(key[len(_META_PREFIX):])
    return found


def collect_active_session_ids(
    *,
    kanban_ids: Optional[Iterable[str]] = None,
    delegation_ids: Optional[Iterable[str]] = None,
    background_ids: Optional[Iterable[str]] = None,
) -> tuple[set[str], bool]:
    """Union of originating sessions with in-flight tracked work.

    Pass an iterable to override a scanner in tests. ``None`` runs the live
    scanner. A failed live scanner is not an empty result: the returned flag
    is false so reconciliation keeps existing checks.
    """
    scans = [
        ScanResult(set(kanban_ids), True) if kanban_ids is not None else _scan_kanban_sessions(),
        ScanResult(set(delegation_ids), True) if delegation_ids is not None else _scan_delegation_sessions(),
        ScanResult(set(background_ids), True) if background_ids is not None else _scan_background_sessions(),
    ]
    return {sid for scan in scans for sid in scan.ids if sid}, all(scan.ok for scan in scans)


def _scan_kanban_sessions() -> ScanResult:
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli.kanban_db_connect import connect as kanban_connect
    except Exception:
        logger.debug("active-work progress: kanban import failed", exc_info=True)
        return ScanResult(set(), False)
    found: set[str] = set()
    ok = True
    try:
        boards = kb.list_boards(include_archived=False)
    except Exception:
        logger.debug("active-work progress: kanban board list failed", exc_info=True)
        return ScanResult(set(), False)
    for board in boards:
        slug = (board or {}).get("slug") or getattr(kb, "DEFAULT_BOARD", None)
        try:
            conn = kanban_connect(board=slug)
        except Exception:
            logger.debug("active-work progress: kanban connect %s failed", slug, exc_info=True)
            ok = False
            continue
        try:
            for task in kb.list_tasks(conn, status=_ACTIVE_KANBAN_STATUS):
                if task.session_id:
                    found.add(task.session_id)
        except Exception:
            logger.debug("active-work progress: kanban list %s failed", slug, exc_info=True)
            ok = False
        finally:
            try:
                conn.close()
            except Exception:
                logger.debug("active-work progress: kanban close %s failed", slug, exc_info=True)
    return ScanResult(found, ok)


def _scan_delegation_sessions() -> ScanResult:
    path = _state_db_path()
    if path is None:
        return ScanResult(set(), False)
    if not path.exists():
        return ScanResult(set(), True)
    found: set[str] = set()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception:
        logger.debug("active-work progress: state.db open failed", exc_info=True)
        return ScanResult(set(), False)
    try:
        rows = conn.execute(
            "SELECT origin_session_id, parent_session_id FROM async_delegations "
            f"WHERE state IN ({', '.join('?' * len(_LIVE_DELEGATION_STATES))}) "
            "AND COALESCE(stop_state, '') = ''",
            _LIVE_DELEGATION_STATES,
        ).fetchall()
    except sqlite3.Error:
        logger.debug("active-work progress: delegation scan failed", exc_info=True)
        return ScanResult(set(), False)
    finally:
        conn.close()
    for origin, parent in rows:
        sid = (origin or "").strip() or (parent or "").strip()
        if sid:
            found.add(sid)
    return ScanResult(found, True)


def _scan_background_sessions(registry: object | None = None) -> ScanResult:
    """Parent sessions of running background processes.

    Processes handed off by a child are not a separate progress subject: the
    parent's delegation row already covers that work, and child build output
    stays suppressed.
    """
    if registry is None:
        try:
            from tools.process_registry import process_registry as registry
        except ImportError:
            logger.debug("active-work progress: process registry unavailable", exc_info=True)
            return ScanResult(set(), False)
    found: set[str] = set()
    running = getattr(registry, "_running", {}) or {}
    for session in list(running.values()):
        if getattr(session, "handoff_note", ""):
            continue
        parent = getattr(session, "parent_session_id", "") or ""
        if parent:
            found.add(parent)
    return ScanResult(found, True)


def _state_db_path() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "state.db"
    except ImportError:
        logger.debug("active-work progress: hermes home unavailable", exc_info=True)
        return None


def managed_prompt_is_current(session_id: str) -> bool:
    state = load_heartbeat(session_id)
    return bool(
        state and state.origin == ACTIVE_WORK_HEARTBEAT_ORIGIN
        and state.prompt == ACTIVE_WORK_PROMPT
        and state.interval_seconds == ACTIVE_WORK_INTERVAL_SECONDS
        and state.status == "active"
    )
