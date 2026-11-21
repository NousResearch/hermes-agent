"""Lane-model overrides: board-level, time-boxed model routing for spawns.

A row in ``lane_model_overrides`` re-routes every dispatcher spawn in its lane
(one assignee, or ``''`` for the whole board) whose card has no per-card
``model_override`` of its own. Rows carry a mandatory ``expires_at`` on the
dispatcher clock, so a capacity workaround ("route everything to provider X
for two hours") cannot silently become a standing default the way editing a
profile's ``config.yaml`` does.

Precedence at spawn: per-card override > assignee lane > board-wide lane >
profile default. The lane route is applied to the in-memory claimed ``Task``
only — it is never persisted onto the card, or it would outlive its window.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from hermes_cli.kanban_db_connect import write_txn

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


@dataclass
class LaneModelOverride:
    """An active lane route with an expiry on the dispatcher clock."""

    assignee: Optional[str]
    provider: str
    model: str
    reasoning_effort: Optional[str] = None
    reason: Optional[str] = None
    created_by: Optional[str] = None
    created_at: int = 0
    expires_at: int = 0

    @property
    def route(self) -> str:
        return f"{self.provider}/{self.model}"

    def ttl_remaining(self, now: Optional[int] = None) -> int:
        """Seconds left before the dispatcher stops applying this row."""
        current = int(time.time()) if now is None else int(now)
        return max(0, int(self.expires_at) - current)


def _row(row) -> LaneModelOverride:
    return LaneModelOverride(
        assignee=row["assignee"] or None,
        provider=row["provider"],
        model=row["model"],
        reasoning_effort=row["reasoning_effort"],
        reason=row["reason"],
        created_by=row["created_by"],
        created_at=int(row["created_at"]),
        expires_at=int(row["expires_at"]),
    )


def _lane_key(assignee: Optional[str]) -> str:
    return (assignee or "").strip()


def set_lane_model_override(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    expires_at: int,
    reasoning_effort: Optional[str] = None,
    reason: Optional[str] = None,
    assignee: Optional[str] = None,
    created_by: Optional[str] = None,
    now: Optional[int] = None,
) -> LaneModelOverride:
    """Install (or replace) the override for ``assignee`` (None = board-wide).

    Re-setting the same lane is an upsert, so extending a window never stacks
    duplicate rows.
    """
    provider = (provider or "").strip()
    model = (model or "").strip()
    if not provider or not model:
        raise ValueError("a lane-model override needs both a provider and a model")
    created = int(time.time()) if now is None else int(now)
    if int(expires_at) <= created:
        raise ValueError("a lane-model override must expire in the future")
    key = _lane_key(assignee)
    with write_txn(conn):
        conn.execute(
            "INSERT INTO lane_model_overrides "
            "(assignee, provider, model, reasoning_effort, reason, created_by, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(assignee) DO UPDATE SET "
            "  provider=excluded.provider, model=excluded.model, "
            "  reasoning_effort=excluded.reasoning_effort, reason=excluded.reason, "
            "  created_by=excluded.created_by, created_at=excluded.created_at, "
            "  expires_at=excluded.expires_at",
            (key, provider, model, reasoning_effort, reason, created_by, created, int(expires_at)),
        )
    return LaneModelOverride(
        assignee=key or None, provider=provider, model=model,
        reasoning_effort=reasoning_effort, reason=reason, created_by=created_by,
        created_at=created, expires_at=int(expires_at),
    )


def get_lane_model_override(
    conn: sqlite3.Connection,
    *,
    assignee: Optional[str] = None,
    now: Optional[int] = None,
) -> Optional[LaneModelOverride]:
    """Active override for ``assignee``: its own lane beats the board-wide one.

    Expiry is exclusive — a row is live strictly before ``expires_at`` — so
    the TTL boundary belongs to the profile default. Filtering here (not only
    in the expiry sweep) means a lapsed row is invisible even before the
    dispatcher deletes it.
    """
    current = int(time.time()) if now is None else int(now)
    row = conn.execute(
        "SELECT * FROM lane_model_overrides "
        "WHERE assignee IN (?, '') AND expires_at > ? "
        # '' sorts before any real name, so DESC puts the assignee row first.
        "ORDER BY assignee DESC LIMIT 1",
        (_lane_key(assignee), current),
    ).fetchone()
    return _row(row) if row else None


def list_lane_model_overrides(
    conn: sqlite3.Connection,
    *,
    now: Optional[int] = None,
) -> list[LaneModelOverride]:
    """Active overrides, board-wide lane first."""
    current = int(time.time()) if now is None else int(now)
    rows = conn.execute(
        "SELECT * FROM lane_model_overrides WHERE expires_at > ? ORDER BY assignee ASC",
        (current,),
    ).fetchall()
    return [_row(r) for r in rows]


def clear_lane_model_override(
    conn: sqlite3.Connection,
    *,
    assignee: Optional[str] = None,
) -> Optional[LaneModelOverride]:
    """Drop the override for ``assignee``; return what was removed.

    Read and delete under ONE writer lock: a row read before the lock could
    be replaced by a concurrent ``set`` whose new window this call would then
    delete while reporting the old one.
    """
    key = _lane_key(assignee)
    with write_txn(conn):
        row = conn.execute(
            "SELECT * FROM lane_model_overrides WHERE assignee = ?", (key,),
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM lane_model_overrides WHERE assignee = ?", (key,))
    return _row(row)


def clear_all_lane_model_overrides(conn: sqlite3.Connection) -> list[LaneModelOverride]:
    """Drop EVERY override in one transaction; return exactly what was deleted."""
    with write_txn(conn):
        rows = [_row(r) for r in conn.execute(
            "SELECT * FROM lane_model_overrides ORDER BY assignee ASC"
        ).fetchall()]
        if rows:
            conn.execute("DELETE FROM lane_model_overrides")
    return rows


def expire_lane_model_overrides(
    conn: sqlite3.Connection,
    *,
    now: Optional[int] = None,
) -> list[LaneModelOverride]:
    """Delete every elapsed override and return them.

    The dispatcher calls this once per tick. Because the rows are DELETED, each
    expiry is reported exactly once. The report is read under the same writer
    lock as the DELETE, so a window renewed in between is neither deleted nor
    announced as expired.
    """
    current = int(time.time()) if now is None else int(now)
    # Cheap unlocked probe so an idle tick never takes the writer lock.
    if conn.execute(
        "SELECT 1 FROM lane_model_overrides WHERE expires_at <= ? LIMIT 1", (current,),
    ).fetchone() is None:
        return []
    with write_txn(conn):
        rows = conn.execute(
            "SELECT * FROM lane_model_overrides WHERE expires_at <= ? ORDER BY assignee ASC",
            (current,),
        ).fetchall()
        if rows:
            conn.execute("DELETE FROM lane_model_overrides WHERE expires_at <= ?", (current,))
    return [_row(r) for r in rows]


def lane_successor_label(override: Optional[LaneModelOverride]) -> str:
    """Name what routes a lane once a row is gone.

    ``override`` is the still-active lookup for that lane: None means the
    profile default; a surviving row (the board-wide lane under a retired
    assignee lane, or a renewed window) is named with its route, so a receipt
    never claims "profile default" while a lane is still routing the cards.
    """
    if override is None:
        return "profile default"
    scope = f"lane {override.assignee}" if override.assignee else "board-wide lane"
    return f"{scope} {override.route}"


def apply_lane_model_override(
    task: "Task",
    override: Optional[LaneModelOverride],
    *,
    now: Optional[int] = None,
) -> Optional[str]:
    """Route the in-memory ``task`` through ``override`` unless the card is pinned.

    Returns the ``source`` label for the spawn's route line, or ``None`` when
    the lane did not apply. A per-card override always wins: it is the
    narrower, explicitly chosen instruction, and silently overwriting it from
    a board-level row would make ``set-model`` unreliable. A card-level
    reasoning effort likewise beats the lane's.
    """
    if override is None or task.model_override:
        return None
    task.model_override = override.model
    task.provider_override = override.provider
    if task.reasoning_effort is None:
        task.reasoning_effort = override.reasoning_effort
    return f"lane-override({override.ttl_remaining(now)}s remaining)"
