"""The dispatcher's routing phase — who runs this card, or is it parked?

Split out of ``hermes_cli.kanban_db_dispatch`` so that file (already over the
size cap) does not grow. The dispatcher calls :func:`route_lanes` before it
spawns anything.

The phase answers one question per card in the ready/review lanes: can the
board run it? A card with no assignee cannot be spawned, so it is parked —
moved to the ``parked`` column where it is visible and countable — unless
``kanban.default_assignee`` names someone who can.
"""

from __future__ import annotations

import sqlite3
from typing import Callable, Optional


def lane_rows(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    """Unclaimed rows of one lane in dispatch order."""
    return conn.execute(
        "SELECT id, assignee FROM tasks "
        f"WHERE status = '{status}' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()


def any_spawnable_review(
    conn: sqlite3.Connection,
    review_rows: list[sqlite3.Row],
    *,
    profile_exists: Optional[Callable[[str], bool]] = None,
    check_respawn_guard: Optional[Callable[..., Optional[str]]] = None,
    per_profile_cap: Optional[int] = None,
    per_profile_running: Optional[dict[str, int]] = None,
) -> bool:
    """Mirror review dispatch gates before reserving ready-lane capacity.

    Unavailable profile metadata retains the historic fail-open behavior. A
    review row that the dispatcher would refuse this tick — its assignee
    already at the per-profile cap, or respawn-guarded — cannot consume the
    reservation, so it must not withhold capacity from an otherwise ready
    task (one such row would pin ``ready_budget`` to 0).
    """
    if not review_rows:
        return False
    running = per_profile_running or {}
    for row in review_rows:
        assignee = row["assignee"]
        if not assignee:
            continue
        if profile_exists is not None and not profile_exists(assignee):
            continue
        if per_profile_cap is not None and running.get(assignee, 0) >= per_profile_cap:
            continue
        if check_respawn_guard is None:
            return True
        if check_respawn_guard(conn, row["id"], lane="review") is None:
            return True
    return False


def route_lanes(
    conn: sqlite3.Connection,
    *,
    default_assignee: Optional[str],
    dry_run: bool,
    apply_assignee: Callable[[str], bool],
    lane_statuses: tuple[str, ...] = ("ready", "review"),
) -> tuple[dict[str, list[sqlite3.Row]], list[str], list[str]]:
    """Route every card in the named lanes.

    Returns ``(lanes, routed_ids, parked_ids)`` where ``lanes`` maps a lane
    status to its rows AFTER routing.

    Two things happen here, both INDEPENDENT of the spawn budget:

    * a parked card gets the fallback assignee and returns to its lane, so
      ``kanban.default_assignee`` keeps working for cards parked before it
      was set;
    * an unassigned card in a lane is moved to ``parked``, because a card
      nobody can run must not sit in ``ready`` where the board counts it as
      queued work it cannot deliver.

    The lanes are re-read afterwards so a card routed in this phase can be
    spawned in the SAME tick.
    """
    from hermes_cli import kanban_db_parked as _parked

    lanes: dict[str, list[sqlite3.Row]] = {}
    routed: list[str] = []
    parked: list[str] = []
    if default_assignee:
        routed.extend(_parked.route_parked_cards(
            conn, default_assignee, dry_run=dry_run, apply_assignee=apply_assignee,
        ))
    for status in lane_statuses:
        rows = lane_rows(conn, status)
        lane_routed, lane_parked = _parked.route_lane_cards(
            conn, rows, source_status=status, default_assignee=default_assignee,
            dry_run=dry_run, apply_assignee=apply_assignee,
        )
        routed.extend(lane_routed)
        parked.extend(lane_parked)
        lanes[status] = lane_rows(conn, status)
    return lanes, routed, parked
