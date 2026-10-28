"""Skills that open kanban cards will force-load, read for the curator's inactivity protection.

A card's ``skills`` column is passed to its worker as ``--skills``; a missing name makes the worker
exit with ``Unknown skill(s)`` before doing any work, so the card burns ``kanban.failure_limit`` and
auto-blocks. The curator therefore treats every skill named by a card that can still be dispatched
as in use, the same way it treats cron-referenced skills.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Set

logger = logging.getLogger(__name__)

# Cards in these states will never spawn a worker again, so their skills need no protection.
_FINISHED_STATUSES = ("done", "archived")


def _canonical_skill_name(raw: object) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    try:
        from agent.skill_utils import normalize_skill_lookup_name
        value = normalize_skill_lookup_name(value) or value
    except Exception:
        logger.debug("kanban skill refs: could not normalize %r", raw, exc_info=True)
    return value.strip().lstrip("/")


def _board_skill_names(db_path: Path, assignee: Optional[str]) -> Set[str]:
    if not db_path.exists():
        return set()
    query = (
        "SELECT skills FROM tasks WHERE skills IS NOT NULL AND skills NOT IN ('', '[]') "
        f"AND status NOT IN ({', '.join('?' for _ in _FINISHED_STATUSES)})"
    )
    params: list = list(_FINISHED_STATUSES)
    if assignee is not None:
        query += " AND LOWER(assignee) = LOWER(?)"
        params.append(assignee)
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
    try:
        rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError:
        logger.debug("kanban skill refs: unreadable board %s", db_path, exc_info=True)
        return set()
    finally:
        conn.close()
    names: Set[str] = set()
    for (raw,) in rows:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, list):
            names.update(n for s in parsed if (n := _canonical_skill_name(s)))
    return names


def _board_db_paths() -> Iterable[Path]:
    from hermes_cli.kanban_db import list_boards
    return [Path(meta["db_path"]) for meta in list_boards(include_archived=False) if meta.get("db_path")]


def referenced_skill_names(assignee: Optional[str]) -> Set[str]:
    """Skill names force-loaded by every not-yet-finished card on any live board assigned to
    *assignee* (``None`` = any assignee). Best-effort: an unreadable board contributes nothing,
    never an exception, so a broken board cannot stop the curator."""
    names: Set[str] = set()
    try:
        paths = list(_board_db_paths())
    except Exception:
        logger.debug("kanban skill refs: could not list boards", exc_info=True)
        return names
    for path in paths:
        try:
            names |= _board_skill_names(path, assignee)
        except Exception:
            logger.debug("kanban skill refs: failed reading %s", path, exc_info=True)
    return names
