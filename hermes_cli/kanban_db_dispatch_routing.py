"""Dispatch-time routing guard: the last net between a ready card and a worker.

Why a second check exists
-------------------------
:mod:`hermes_cli.kanban_routing_guard` guards every write seam that *knows* it is
writing an assignee (CLI ``assign``/``create``, the dashboard, the decomposer).
A card can still reach ``ready`` with an owning assignee through a seam nobody
has taught yet, through raw SQL, or because it was written before the guard
shipped. On 2026-10-05 exactly that happened: a verification fixture landed on
the live board naming ``profiles/wordsmith/SOUL.md`` while assigned to
``wordsmith``, the dispatcher spawned a wordsmith worker, and it had to be
reclaimed before it wrote. Creation-time guards cannot close that class - only
the dispatcher, which sees every row that is about to become a process, can.

So this module re-runs the *same* guard over every row the dispatcher is about
to spawn, with the same module (:func:`guard_assignee`, path-based detection,
config-driven lanes). It is a net, not a second policy: it can only ever move an
assignee off a lane that owns a surface the card edits, and it never invents an
assignee for a card that has none.

Outcomes
--------
* Guard does not fire -> the row spawns exactly as before (zero writes).
* Guard rewrites   -> the new assignee is persisted on the row (one write txn
  that also appends the ``assigned`` event and the guard's audit trail) and the
  claim/spawn proceeds with that lane.
* No legal lane    -> the spawn is skipped for this tick, the row keeps its
  assignee, and the audit trail is written once. Later ticks re-evaluate the
  same row cheaply, so fixing the config (or reassigning the card) heals it on
  the next tick without any manual board surgery.

Interaction with the per-tick auto-assign-to-default path
--------------------------------------------------------
``kanban.default_assignee`` is applied by
:func:`hermes_cli.kanban_db_dispatch._apply_default_assignee` when a ready row
reaches the dispatcher with no assignee at all (``#27145``). That write happens
in the lane loop, *before* this net sees the row, and it is deliberately
unguarded: the default lane is the operator's routing intent, not a routing
decision the guard should second-guess. The net runs immediately after, inside
:func:`hermes_cli.kanban_db_dispatch._dispatch_lane_task`, so a default assignee
that owns a surface the card edits is corrected (or the spawn skipped) in the
same tick - before the claim, and therefore before any worker exists. The board
ends the tick with the guard's lane, not the raw default: the ``assigned``
events read ``source=kanban.default_assignee`` followed by
``source=routing_guard``.

The guard is also the reason ``_resolve_default_assignee`` may refuse to write a
default at all when the profile does not exist; the two checks are independent
and both fail toward "do not spawn the wrong lane".

Cost per tick
-------------
Detection is two regex passes per ready/review row (title + body) plus one
profile lookup per candidate lane, resolved ONCE per tick by
:func:`spawnable_guard_lanes` - not per row. Rows the guard does not touch cost
no writes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from hermes_cli import kanban_routing_guard as rg

__all__ = [
    "DispatchRoutingOutcome",
    "GuardLaneSet",
    "apply_dispatch_routing_guard",
    "decide_dispatch_routing",
    "spawnable_guard_lanes",
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardLaneSet:
    """The guard's candidate lanes for one dispatcher tick.

    ``names`` restricts candidate lanes to the profiles THIS home can spawn
    (``None`` = eligibility unknown, fall back to the guard's config-only
    semantics). ``default_assignee`` is the tick's ``kanban.default_assignee``
    as an explicit value - ``""`` disables that lane - so the guard never
    re-reads config behind the dispatcher's back.
    """

    default_assignee: str = ""
    names: Optional[list[str]] = None


@dataclass(frozen=True)
class DispatchRoutingOutcome:
    """What the net decided for one ready/review row.

    Callers must spawn for ``assignee`` when ``skipped`` is False, and must not
    spawn at all when it is True. ``decision`` is the guard's
    :class:`hermes_cli.kanban_routing_guard.GuardDecision` (carrying the matched
    paths and the reason for logs/telemetry); its ``rewritten`` flag says
    whether the net changed anything this tick.
    """

    assignee: Optional[str]
    decision: Optional[rg.GuardDecision] = None
    skipped: bool = False
    reason: str = ""
    audit_written: bool = False


def spawnable_guard_lanes(
    profile_exists: Optional[Callable[[str], bool]],
    *,
    fallback_assignee: Optional[str] = None,
    default_assignee: str = "",
) -> Optional[GuardLaneSet]:
    """Resolve this tick's guard lanes, filtered to profiles the home can spawn.

    ``profile_exists`` is the dispatcher's own predicate (already narrowed by
    ``kanban.dispatch_profiles``), so "legal lane" means exactly "a lane this
    home is allowed to place a worker on". ``None`` means the profiles module is
    unimportable and the guard must fall back to its config-only semantics -
    the same fail-open the rest of dispatch applies.

    Both candidate lanes are checked once here; the ready and review loops share
    the result so they can never disagree about which lanes are legal.
    """
    if profile_exists is None:
        return None
    fallback = fallback_assignee if fallback_assignee is not None else rg.configured_fallback_assignee()
    names: list[str] = []
    for candidate in (fallback, default_assignee):
        lane = (candidate or "").strip()
        if not lane or lane in names:
            continue
        try:
            if profile_exists(lane):
                names.append(lane)
        except Exception:  # pragma: no cover - predicate is stdlib + fs lookups
            _log.debug("kanban dispatch: routing-guard lane probe failed for %r", lane, exc_info=True)
    return GuardLaneSet(default_assignee=(default_assignee or "").strip(), names=names)


def decide_dispatch_routing(
    assignee: Optional[str],
    title: Optional[str],
    body: Optional[str],
    *,
    lanes: Optional[GuardLaneSet] = None,
) -> rg.GuardDecision:
    """Read-only guard decision for one row (no DB access, no writes).

    Only the title and body are inspected. Comments are deliberately excluded:
    the creation-time seams the guard already covers read the same two fields,
    and re-reading every comment of every ready row on every tick would put a
    per-row query in the dispatcher's hot path for text that cannot change
    where a card is routed.
    """
    if lanes is None:
        return rg.guard_assignee(assignee, title or "", body or "")
    return rg.guard_assignee(
        assignee,
        title or "",
        body or "",
        default_assignee=lanes.default_assignee,
        eligible_names=lanes.names,
    )


def _already_audited(conn, task_id: str, payload: dict) -> bool:
    """True when the card's latest event is already this exact guard audit.

    The unassigned case re-evaluates the same row every tick; without this the
    board would collect a comment + event per tick forever (the same reason
    ``skipped_nonspawnable`` writes its diagnostic once). Any newer event -
    reassignment, comment, status change - re-arms the audit, so a genuinely new
    skip is always recorded.
    """
    row = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or row["kind"] != rg.ROUTING_GUARD_EVENT_KIND:
        return False
    try:
        last = json.loads(row["payload"]) if row["payload"] else None
    except (TypeError, ValueError):
        return False
    return last == payload


def _record_skip_audit(conn, task_id: str, decision: rg.GuardDecision) -> bool:
    """Write the "no legal lane" audit once. Returns True when it was written."""
    if _already_audited(conn, task_id, decision.audit_payload()):
        return False
    rg.record_guard_decision(conn, task_id, decision, author="dispatcher")
    return True


def _apply_rewrite(conn, task_id: str, original: str, decision: rg.GuardDecision) -> bool:
    """Persist the rewrite + its audit trail in ONE transaction.

    The UPDATE is a compare-and-swap on the assignee we just read: if another
    writer moved or claimed the row between the lane snapshot and this moment,
    nothing is written and the caller skips the spawn for this tick rather than
    spawning a route it no longer owns. Mirrors
    ``_apply_default_assignee``'s "only while still mine" discipline.

    ``consecutive_failures`` is reset exactly like :func:`kanban_db.assign_task`
    does on a handoff: the streak is per task/profile, and the old profile's
    failures must not follow the card to its new lane.
    """
    from hermes_cli import kanban_db as kb

    with kb.write_txn(conn):
        updated = conn.execute(
            "UPDATE tasks SET assignee = ?, consecutive_failures = 0, "
            "last_failure_error = NULL "
            "WHERE id = ? AND assignee = ? AND claim_lock IS NULL",
            (decision.assignee, task_id, original),
        )
        if updated.rowcount != 1:
            return False
        # ``from`` lets the respawn guard read this as a real handoff.
        kb._append_event(
            conn, task_id, "assigned",
            {"assignee": decision.assignee, "from": original, "source": "routing_guard"},
        )
        rg.record_guard_decision(conn, task_id, decision, author="dispatcher")
    return True


def apply_dispatch_routing_guard(
    conn,
    task_id: str,
    assignee: Optional[str],
    title: Optional[str],
    body: Optional[str],
    *,
    lanes: Optional[GuardLaneSet] = None,
    dry_run: bool = False,
) -> DispatchRoutingOutcome:
    """Run the routing net for one ready/review row the dispatcher is about to spawn.

    ``assignee`` must be the assignee this tick will actually spawn for (the
    auto-assigned default when ``kanban.default_assignee`` just filled an
    unassigned row) - it is both the guard input and the CAS key. ``dry_run``
    reports the decision without writing anything, so ``hermes kanban dispatch
    --dry-run`` names the lane the real tick would use.

    Never raises for a guard failure: a broken guard must degrade to "spawn as
    routed" (the historical behaviour) rather than stop the board.
    """
    try:
        decision = decide_dispatch_routing(assignee, title, body, lanes=lanes)
    except Exception:
        _log.warning(
            "kanban dispatch: routing guard failed on %s; spawning as routed", task_id, exc_info=True,
        )
        return DispatchRoutingOutcome(assignee=assignee)

    if not decision.rewritten:
        return DispatchRoutingOutcome(assignee=assignee, decision=decision)

    if decision.assignee is None:
        # No legal lane: never spawn the owning profile. The row keeps its
        # assignee so the board still shows the operator's intent; this tick
        # (and every tick until the config or the card changes) skips it.
        _log.warning(
            "kanban dispatch: skipping %s - %s", task_id, decision.reason,
        )
        if dry_run:
            return DispatchRoutingOutcome(
                assignee=None, decision=decision, skipped=True, reason=decision.reason,
            )
        audit_written = False
        try:
            audit_written = _record_skip_audit(conn, task_id, decision)
        except Exception:
            _log.warning(
                "kanban dispatch: could not record routing-guard audit for %s", task_id, exc_info=True,
            )
        return DispatchRoutingOutcome(
            assignee=None, decision=decision, skipped=True,
            reason=decision.reason, audit_written=audit_written,
        )

    if dry_run:
        return DispatchRoutingOutcome(assignee=decision.assignee, decision=decision)

    try:
        applied = _apply_rewrite(conn, task_id, str(assignee), decision)
    except Exception:
        _log.warning(
            "kanban dispatch: could not persist routing-guard rewrite for %s; "
            "skipping this tick", task_id, exc_info=True,
        )
        applied = False
    if not applied:
        reason = "card changed (assignee or claim) while the routing guard was rerouting it"
        return DispatchRoutingOutcome(
            assignee=None, decision=decision, skipped=True, reason=reason,
        )
    _log.info(
        "kanban dispatch: routing guard rerouted %s: %s -> %s (%s)",
        task_id, decision.original_assignee, decision.assignee, decision.lane_source,
    )
    return DispatchRoutingOutcome(assignee=decision.assignee, decision=decision, audit_written=True)
