"""Fleet activity: everything working right now that the per-chat "live" dot in the
dashboard cannot see.

The WebUI's chat sidebar only reflects the PTY-backed chat session it is attached to
(``ChatSidebar``'s sidecar "live" badge) — Kanban workers (any board, any profile) and
gateway messaging sessions (Telegram/WhatsApp/Discord/...) mid-turn never show up
anywhere in the dashboard. This router is the single aggregation point for both, reusing
existing storage (kanban's per-board SQLite DBs, the gateway routing index already
persisted in each profile's ``state.db``) — no new service, no new persistence.

Extracted as its own router (not folded into ``status.py``) per ``web/AGENTS.md``: "a new
surface is a new web_routers/<surface>.py, not a growing web_server.py".
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from hermes_cli.web_deps import late

router = APIRouter()
_log = logging.getLogger("hermes_cli.web_server")

# Same env var / default as aos/pacing_governor.py's DEFAULT_STATE_PATH and the kanban
# dispatcher's vendored mirror (hermes_cli/kanban_db_dispatch_pacing.py) -- one on-disk
# contract, three independent readers, never a shared writer lock (write_state's
# write-then-rename already makes a torn read impossible, not just unlikely).
_PACE_STATE_PATH = Path(os.environ.get(
    "HERMES_PACING_STATE_PATH", "~/.qwickapps/state/provider_pace.json")).expanduser()

# Late-bound so a test's monkeypatch on the owning module wins at call time (web/AGENTS.md).
_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")
_session_db_path_for_profile = late("_session_db_path_for_profile", "hermes_cli.web_server_sessions")
_collect_profile_gateway_topology_cached = late(
    "_collect_profile_gateway_topology_cached", "hermes_cli.web_server_gateway")


def _kanban_running_tasks() -> List[Dict[str, Any]]:
    """Running tasks across EVERY kanban board (``default`` + ``kanban/boards/*``).

    Boards live outside any single profile's home by design (``kanban_db.kanban_home()``
    is shared across profiles so the dispatcher/worker handoff never forks per profile) —
    this is naturally fleet-wide with no profile scoping needed. Best-effort per board, the
    same shape as ``kanban_db_dispatch.count_running_tasks_other_boards``: one unreadable
    board must not blank the whole panel.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect as _kanban_connect

    try:
        boards = kb.list_boards(include_archived=False)
    except Exception:
        _log.debug("fleet activity: list_boards failed", exc_info=True)
        return []

    now = time.time()
    tasks: List[Dict[str, Any]] = []
    for meta in boards:
        slug = meta.get("slug") or kb.DEFAULT_BOARD
        try:
            path = kb.kanban_db_path(board=slug).expanduser()
            if not path.exists():
                continue
            conn = _kanban_connect(board=slug)
        except Exception:
            _log.debug("fleet activity: skipping unreadable board %r", slug, exc_info=True)
            continue
        try:
            for t in kb.list_tasks(conn, status="running"):
                started = t.started_at
                heartbeat = t.last_heartbeat_at
                tasks.append({
                    "board": slug,
                    "board_name": meta.get("name") or slug,
                    "task_id": t.id,
                    "title": t.title,
                    "profile": t.assignee,
                    "started_at": started,
                    "elapsed_seconds": max(0, int(now - started)) if started else None,
                    "last_heartbeat_at": heartbeat,
                    "heartbeat_age_seconds": max(0, int(now - heartbeat)) if heartbeat else None,
                })
        except Exception:
            _log.debug("fleet activity: list_tasks failed for board %r", slug, exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    tasks.sort(key=lambda row: row["started_at"] or 0, reverse=True)
    return tasks


def _gateway_sessions_for_profile(profile: Optional[str], *, label: str) -> List[Dict[str, Any]]:
    """Gateway routing-index entries with a turn in flight for one profile's ``state.db``.

    ``active_turn_started_at`` (``gateway/session.py::SessionEntry``) is the durable marker
    of an executing turn — CAS-cleared on normal unwind — so a non-null value here is exactly
    the gateway-side equivalent of the dashboard's per-chat "live" dot. Best-effort: a missing
    or unreadable store (profile never had gateway traffic) yields no rows, never an error.
    """
    try:
        db_path = _session_db_path_for_profile(profile)
        if not Path(db_path).exists():
            return []
        db = _open_session_db_for_profile(profile, read_only=True)
    except Exception:
        _log.debug("fleet activity: cannot open session db for profile %r", profile, exc_info=True)
        return []

    try:
        raw = db.load_gateway_routing_entries()
    except Exception:
        _log.debug("fleet activity: load_gateway_routing_entries failed for %r", profile, exc_info=True)
        return []
    finally:
        with contextlib.suppress(Exception):
            db.close()

    now = time.time()
    out: List[Dict[str, Any]] = []
    for session_key, entry_json in raw.items():
        try:
            data = json.loads(entry_json or "{}")
        except Exception:
            continue
        started_raw = data.get("active_turn_started_at")
        if not started_raw:
            continue  # no turn in flight on this lane right now
        try:
            started_epoch = datetime.fromisoformat(started_raw).timestamp()
        except Exception:
            continue
        out.append({
            "profile": label,
            "session_key": session_key,
            "platform": data.get("platform"),
            "display_name": data.get("display_name"),
            "chat_type": data.get("chat_type"),
            "started_at": started_epoch,
            "elapsed_seconds": max(0, int(now - started_epoch)),
        })
    return out


def _all_gateway_sessions() -> List[Dict[str, Any]]:
    """Active-turn gateway sessions across every profile this install serves.

    Mirrors ``status.py``'s own-profile-first-then-others shape (``_merge_profile_gateway_platforms``):
    the calling process's own ``state.db`` is read via the ``profile=None`` fast path (matches
    ``/api/status``'s zero-arg call), other served profiles (multiplex) are added by name.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name
        own_name = get_active_profile_name()
    except Exception:
        own_name = "default"

    targets: List[tuple[Optional[str], str]] = [(None, own_name)]
    try:
        topology = _collect_profile_gateway_topology_cached()
        for name in topology.get("profiles") or []:
            if name and name != own_name:
                targets.append((name, name))
    except Exception:
        _log.debug("fleet activity: profile topology unavailable", exc_info=True)

    sessions: List[Dict[str, Any]] = []
    for profile_arg, label in targets:
        sessions.extend(_gateway_sessions_for_profile(profile_arg, label=label))
    sessions.sort(key=lambda row: row["started_at"], reverse=True)
    return sessions


def _provider_pace() -> Optional[Dict[str, Any]]:
    """Pace-vs-actual per provider, straight off the on-disk state file the pacing
    governor's poll tick writes (``aos/pacing_governor.py::write_state``, same file the
    kanban dispatcher's spawn-routing already reads via
    ``hermes_cli/kanban_db_dispatch_pacing.py``). This router never imports ``aos`` (a
    separate repo/venv, not on hermes-agent's runtime ``sys.path`` today) and never
    recomputes pace math -- it only reads the doc the governor already evaluated.

    Fail-open like every other reader of this file: a missing/unreadable/corrupt state
    file (governor never ran, or ran on a host without this file) returns ``None`` so the
    caller omits the key entirely -- it must never turn into a 500 for the rest of the
    fleet activity panel, which has nothing to do with the pacing governor's health.
    """
    try:
        doc = json.loads(_PACE_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    providers = doc.get("providers")
    if not isinstance(providers, dict):
        return None
    return {
        "generated_at": doc.get("generated_at"),
        "chain": doc.get("chain") or [],
        "reserved_lane_pct": doc.get("reserved_lane_pct"),
        "providers": providers,
    }


@router.get("/api/fleet/activity")
async def get_fleet_activity():
    """Everything working right now, fleet-wide: running Kanban tasks across every board
    (card, profile, elapsed, last heartbeat) plus gateway sessions mid-turn across every
    served profile (platform, chat, elapsed) — the panel `@raaj` uses to see all agents at
    a glance, regardless of surface (WebUI chat, gateway messaging, Kanban worker).

    Also surfaces ``provider_pace`` (pace-vs-actual per provider from the pacing governor's
    state file, t_1eb32e10 item 5 / t_9fa39b57) when that state file exists and is
    readable; ``null`` when it does not (fresh install, governor never polled, or a host
    without the governor at all) -- never an error for the rest of the panel."""
    kanban_tasks = await run_in_threadpool(_kanban_running_tasks)
    gateway_sessions = await run_in_threadpool(_all_gateway_sessions)
    provider_pace = await run_in_threadpool(_provider_pace)
    return {
        "kanban_tasks": kanban_tasks,
        "gateway_sessions": gateway_sessions,
        "count": len(kanban_tasks) + len(gateway_sessions),
        "provider_pace": provider_pace,
    }
