"""Board-liveness invariant (ADR 0020 §2.3/§3.4/§3.5/§4.1 — build card t_68d5b5a4).

The incident this closes is the one nobody saw: a board with **open goals, zero running, zero
ready, nothing promotable** sitting in silence because every existing watcher is per-card.
The kernel emits a board-level event the moment that predicate holds, and tells the two
failure shapes apart so the alert stays trustworthy (ADR §3.4):

``DEAD``
    Open ``todo``/``ready`` work exists but nothing runs and nothing can be promoted
    (``ready`` whose assignee is not a live profile, ``todo`` gated on a parent chain that is
    itself stuck). This is "the chain broke and the pipeline is dead" — alert now, then once
    per UTC day while it stays dead.

``QUIET_BY_DESIGN``
    Every open goal legitimately waits on a person or a timer: ``blocked`` (operator decision
    owns the clock), ``triage`` (a specifier owns it), ``scheduled`` (``SCHEDULED_UNTIL`` owns
    it), or a ``todo`` gated *only* on such cards. The board *should* be silent, so this fires
    at most once per open-goal set (fingerprint-keyed), never daily.

``scheduled``-only boards are ``HEALTHY`` — the fleet wake driver owns them (ADR §3.5).

Board-level events need a row to hang on (subscriptions and cursors are keyed by ``task_id``),
so a per-board sentinel task ``board:<slug>:liveness`` is created idempotently
(:func:`ensure_sentinel_task`, created lazily on first emit). It is deliberately:

* ``status='scheduled'`` — invisible to the estate ``stall-sentinel.py`` (its parked set is
  ``blocked/triage/todo`` and its STRANDED arms cover ``ready``/``todo``/``running`` only), and
  inert for ``fleet-wake-driver.py``, which skips marker-less scheduled rows by design. The
  ADR suggested ``todo``; that shape makes the sentinel itself a permanent STRANDED finding on
  every board, so this is a deliberate, documented deviation.
* ``assignee IS NULL`` — never spawnable (``_has_spawnable`` requires an assignee), so it can
  never satisfy ``promotable_any`` or be dispatched;
* excluded from every count here by the ``board:%:liveness`` filter, and from ``list_tasks``
  so dashboards/CLI never show it.

Because it is a non-done task, the notifier's existing ``auto_subscribe_stream`` backfill gives
it a subscription on the operator stream automatically — which is what makes FIX D work with no
new delivery path (ADR §2.4).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

STATE_DEAD = "DEAD"
STATE_QUIET_BY_DESIGN = "QUIET_BY_DESIGN"
STATE_HEALTHY = "HEALTHY"

#: ``task_events.kind`` per state (ADR §4.1).
EVENT_KINDS = {
    STATE_DEAD: "board_dead_quiet",
    STATE_QUIET_BY_DESIGN: "board_quiet_by_design",
}

#: Open (not done/archived) statuses counted as "the board still has goals".
OPEN_STATUSES: tuple[str, ...] = ("todo", "ready", "running", "blocked", "triage", "scheduled")

#: Statuses whose silence is legitimate: a person or a timer owns the clock (ADR §3.4).
PARKED_BY_DESIGN_STATUSES: tuple[str, ...] = ("blocked", "triage", "scheduled")

#: Block kinds that are an explicit request for a human decision.
OWNER_DECISION_KINDS: tuple[str, ...] = ("needs_input", "capability")

#: SQL predicate excluding the per-board sentinels from every task-counting query here.
SENTINEL_EXCLUSION = "id NOT LIKE 'board:%:liveness'"

DEFAULT_INTERVAL_SECONDS = 300

_SENTINEL_TITLE = "[system] board liveness sentinel — never dispatch, never edit"
_SENTINEL_BODY = (
    "Kernel-managed row (hermes_cli/kanban_board_liveness.py, ADR 0020 §4.1). Board-level "
    "liveness events (`board_dead_quiet` / `board_quiet_by_design`) anchor here so the normal "
    "subscription cursor delivers them. It is scheduled and unassigned on purpose: the "
    "dispatcher cannot spawn it, the wake driver ignores it, and the stall sentinel does not "
    "scan it. Delete it and the next connect() re-creates it."
)


def sentinel_id(board: str) -> str:
    """``board:<slug>:liveness`` — the anchor task id for board-level events."""
    return f"board:{board}:liveness"


def is_sentinel_id(task_id: str) -> bool:
    return bool(re.match(r"^board:.+:liveness$", task_id or ""))


def ensure_sentinel_task(conn: sqlite3.Connection, board: str) -> Optional[str]:
    """Create the board's liveness sentinel once; returns its id (None on bad input).

    ``INSERT OR IGNORE`` makes it safe to call from any lane and any number of times (ADR §6.3),
    including concurrently from both build lanes. Never touches an existing row: an operator who
    re-points or archives the sentinel is left alone.
    """
    board = (board or "").strip()
    if not board:
        return None
    task_id = sentinel_id(board)
    from hermes_cli import kanban_db as _kb

    with _kb.write_txn(conn):
        conn.execute(
            "INSERT OR IGNORE INTO tasks "
            "(id, title, body, assignee, status, priority, created_by, created_at, "
            " workspace_kind, tenant) "
            "VALUES (?, ?, ?, NULL, 'scheduled', 0, 'system', ?, 'scratch', NULL)",
            (task_id, _SENTINEL_TITLE, _SENTINEL_BODY, int(time.time())),
        )
    return task_id


@dataclass
class LivenessResult:
    """Verdict of one board-liveness check (ADR §2.3)."""

    state: str
    board: Optional[str] = None
    counts: dict = field(default_factory=dict)
    reason: str = ""
    #: Task ids that made the verdict actionable (DEAD) or owned it (QUIET) — payload detail.
    suspects: list[str] = field(default_factory=list)
    #: Every open (non-sentinel) task id at check time — the QUIET fingerprint's input.
    open_ids: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        return self.state in EVENT_KINDS

    def payload(self, *, dedup_key: str, now: int) -> dict:
        payload = {
            "state": self.state,
            "board": self.board,
            "reason": self.reason,
            "dedup_key": dedup_key,
            "checked_at": now,
            **self.counts,
        }
        if self.suspects:
            payload["suspects"] = list(self.suspects[:20])
        return payload


def _default_assignee_live() -> Optional[Callable[[str], bool]]:
    """The dispatcher's own "is this assignee a real profile" oracle, if importable.

    Reusing ``kanban_db_dispatch`` keeps "promotable" meaning exactly what the dispatcher would
    act on (ADR §2.3 — "nothing new to invent"). ``None`` = introspection unavailable, in which
    case callers fall back to "assignee present", mirroring ``_has_spawnable``.
    """
    try:
        from hermes_cli import kanban_db_dispatch as _kbd

        return _kbd._profile_exists_fn()
    # With no dispatcher introspection available the caller degrades to "assignee present",
    # mirroring _has_spawnable rather than losing the beat, so this lookup is fail-open.
    # health: allow BLE001 -- fail-open oracle; "unknown" must mean "assume spawnable"
    except Exception:  # pragma: no cover - import failure means "assume spawnable"
        return None


def _open_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    placeholders = ",".join("?" * len(OPEN_STATUSES))
    return conn.execute(
        f"SELECT id, status, assignee, block_kind, claim_lock, created_at FROM tasks "
        f"WHERE status IN ({placeholders}) AND {SENTINEL_EXCLUSION} ORDER BY created_at ASC",
        OPEN_STATUSES,
    ).fetchall()


def _parents(conn: sqlite3.Connection, task_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT t.id, t.status FROM task_links l JOIN tasks t ON t.id = l.parent_id "
        "WHERE l.child_id = ?",
        (task_id,),
    ).fetchall()


def _parents_satisfied(rows: list[sqlite3.Row]) -> bool:
    return all(r["status"] in ("done", "archived") for r in rows)


def _gated_only_on_owner_wait(rows: list[sqlite3.Row]) -> bool:
    """True when every unfinished parent is itself waiting on a person/timer (ADR §3.4).

    ``todo`` behind such a parent is quiet-by-design, not dead: the clock belongs to the
    operator or the wake driver, exactly like the block itself.
    """
    unfinished = [r for r in rows if r["status"] not in ("done", "archived")]
    if not unfinished:
        return False
    return all(
        r["status"] in ("triage", "scheduled")
        or (r["status"] == "blocked")
        for r in unfinished
    )


def check_board_liveness(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
    now: Optional[int] = None,
    assignee_live: Optional[Callable[[str], bool]] = None,
) -> LivenessResult:
    """Evaluate the dead-quiet invariant for one board DB (ADR §2.3).

    Pure read; no writes, no gateway. ``assignee_live`` is injectable so tests (and any caller
    that wants a stricter/looser "is this profile real" rule) can pin the promotable predicate.
    """
    now = int(time.time()) if now is None else int(now)
    rows = _open_rows(conn)
    counts = {
        "open_goals": len(rows), "running": 0, "ready": 0, "todo": 0, "blocked": 0,
        "triage": 0, "scheduled": 0, "promotable": 0,
    }
    by_status: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        by_status.setdefault(r["status"], []).append(r)

    live = _default_assignee_live() if assignee_live is None else assignee_live
    open_ids = [str(r["id"]) for r in rows]
    base = {"board": board, "counts": counts, "open_ids": open_ids}

    def _spawnable(row: sqlite3.Row) -> bool:
        assignee = row["assignee"]
        if not assignee or row["claim_lock"]:
            return False
        return True if live is None else bool(live(assignee))

    if not rows:
        return LivenessResult(STATE_HEALTHY, **base, reason="no open goals")
    if counts["running"]:
        return LivenessResult(
            STATE_HEALTHY, **base, reason=f"{counts['running']} running",
        )

    promotable: list[sqlite3.Row] = []
    for row in by_status.get("ready", []):
        if _spawnable(row):
            promotable.append(row)
    todo_stuck: list[sqlite3.Row] = []
    for row in by_status.get("todo", []):
        parents = _parents(conn, str(row["id"]))
        if _parents_satisfied(parents):
            if _spawnable(row):
                promotable.append(row)
            elif _gated_only_on_owner_wait(parents):
                continue  # quiet by design: the unfinished parent waits on a person
            else:
                todo_stuck.append(row)  # parents satisfied but no live assignee ⇒ dead
        elif not _gated_only_on_owner_wait(parents):
            todo_stuck.append(row)  # gated on a chain that is itself stuck ⇒ dead (chain break)

    counts["promotable"] = len(promotable)
    if promotable:
        return LivenessResult(
            STATE_HEALTHY, **base, reason=f"{len(promotable)} promotable",
        )

    dead_suspects = [str(r["id"]) for r in (by_status.get("ready", []) + todo_stuck)]
    if dead_suspects:
        ready_dead = [str(r["id"]) for r in by_status.get("ready", [])]
        why = []
        if ready_dead:
            why.append(f"{len(ready_dead)} ready card(s) no live profile can claim")
        if todo_stuck:
            why.append(f"{len(todo_stuck)} todo card(s) gated on an unfinished chain")
        return LivenessResult(
            STATE_DEAD, **base, suspects=dead_suspects,
            reason="open goals exist, nothing running or promotable — " + "; ".join(why),
        )

    if counts["blocked"] or counts["triage"]:
        return LivenessResult(
            STATE_QUIET_BY_DESIGN, **base,
            suspects=[str(r["id"]) for r in (by_status.get("blocked", []) + by_status.get("triage", []))],
            reason="all open cards wait on an operator decision or triage",
        )
    # Only scheduled work (or todos gated purely on scheduled/decision cards): the wake
    # driver owns the clock — never alert (ADR §3.5).
    return LivenessResult(
        STATE_HEALTHY, **base,
        reason="quiet but timer/schedule-owned (scheduled work outstanding)"
        if counts["scheduled"] else "no open goals",
    )


def _dead_dedup_key(board: str, now: int) -> str:
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    return f"board-dead:{board}:{day}"


def _quiet_dedup_key(board: str, result: LivenessResult) -> str:
    """Fingerprint of the *open-goal set* — the alert re-arms only when it changes (ADR §4.1)."""
    digest = hashlib.sha1(
        ",".join(sorted(result.open_ids)).encode("utf-8", "replace")
    ).hexdigest()[:12]
    return f"board-quiet:{board}:{digest}"


def _already_emitted(conn: sqlite3.Connection, task_id: str, dedup_key: str) -> bool:
    """True when an event with this dedup key already exists on the sentinel (any kind).

    Keyed on the payload, not on the kind: the DEAD/QUIET keys are disjoint by construction, so
    an existing row with the same key is by definition the same episode.
    """
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind IN (?, ?) ORDER BY id DESC",
        (task_id, EVENT_KINDS[STATE_DEAD], EVENT_KINDS[STATE_QUIET_BY_DESIGN]),
    ).fetchall()
    for r in rows:
        raw = r["payload"]
        if not raw:
            continue
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("dedup_key") == dedup_key:
            return True
    return False


def _sentinel_has_sub(conn: sqlite3.Connection, task_id: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM kanban_notify_subs WHERE task_id = ? LIMIT 1", (task_id,)
    ).fetchone())


def _inherit_stream_destination(conn: sqlite3.Connection, task_id: str) -> bool:
    """Give the sentinel the subscription the board's cards already have (FIX C routing).

    The board alert must reach "the butler/orchestrator route", which in this kernel is whoever
    already receives the board's card notifications — normally one stream row created by the
    notifier's ``auto_subscribe_stream`` backfill. That backfill only runs inside the gateway
    notifier tick, so a freshly created sentinel (upgraded board, or a board whose first alert
    lands in the same tick it is created) would otherwise have no cursor at all and its FIRST
    dead-quiet event would be silently skipped: this copies the destination here, BEFORE the
    event is written, so the new row starts caught up below the event id and delivers it.

    ``kinds`` is deliberately inherited as unfiltered: the sentinel only ever carries liveness
    kinds, and inheriting a source row's narrower allowlist could drop the alert outright.

    Returns True when a destination exists (pre-existing or newly inherited).
    """
    from hermes_cli import kanban_db_notify as _kbn

    if _sentinel_has_sub(conn, task_id):
        return True
    row = conn.execute(
        "SELECT platform, chat_id, thread_id, user_id, user_id_alt, chat_type, "
        "       notifier_profile, delivery_mode FROM kanban_notify_subs "
        "WHERE task_id NOT LIKE 'board:%:liveness' AND platform IS NOT NULL "
        "  AND platform != 'api_server' "
        "ORDER BY created_at ASC LIMIT 1",
    ).fetchone()
    if row is None or not row["platform"] or not row["chat_id"]:
        return False
    try:
        _kbn.add_notify_sub(
            conn, task_id=task_id, platform=str(row["platform"]), chat_id=str(row["chat_id"]),
            thread_id=row["thread_id"] or None, user_id=row["user_id"], user_id_alt=row["user_id_alt"],
            chat_type=row["chat_type"], notifier_profile=row["notifier_profile"],
            delivery_mode=row["delivery_mode"], kinds=None,
        )
    except sqlite3.Error:  # pragma: no cover - subscription write failure must not lose the event
        return False
    return True


def _emit_board_liveness(
    conn: sqlite3.Connection,
    *,
    board: str,
    state: str,
    result: Optional[LivenessResult] = None,
    now: Optional[int] = None,
) -> Optional[dict]:
    """Write one board-liveness event on the sentinel, honouring the dedup rules.

    Returns the payload written, or ``None`` when nothing was emitted (healthy, unknown state,
    or already reported for this episode/day). Dedup is stored in the event payload rather than
    a side table so a replay is a plain ``SELECT`` on ``task_events`` (ADR §5.1).
    """
    kind = EVENT_KINDS.get(state)
    if kind is None:
        return None
    now = int(time.time()) if now is None else int(now)
    result = result or LivenessResult(state, board=board)
    if state == STATE_DEAD:
        dedup_key = _dead_dedup_key(board, now)
    else:
        dedup_key = _quiet_dedup_key(board, result)

    from hermes_cli import kanban_db as _kb

    task_id = ensure_sentinel_task(conn, board)
    assert task_id is not None  # board is non-empty by caller contract
    if _already_emitted(conn, task_id, dedup_key):
        return None
    payload = result.payload(dedup_key=dedup_key, now=now)
    # Inherit the board's existing notification destination BEFORE the event row exists, so a
    # brand-new subscription's "start caught up" cursor never skips the alert it was created for.
    _inherit_stream_destination(conn, task_id)
    with _kb.write_txn(conn):
        _kb._append_event(conn, task_id, kind, payload)
    return payload


def _config() -> dict:
    try:
        from hermes_cli.config import load_config

        kanban = (load_config() or {}).get("kanban", {})
    # An unreadable config disables the watcher (absent ⇒ no board alerts) rather than raising
    # inside the dispatcher's tick, so this boundary is deliberately blind.
    # health: allow BLE001 -- fail-open config read; absence of config means "do not alert"
    except Exception:  # pragma: no cover - config unreadable ⇒ watcher disabled by absence
        return {}
    block = kanban.get("board_liveness") if isinstance(kanban, dict) else None
    return block if isinstance(block, dict) else {}


def board_liveness_enabled() -> bool:
    return bool(_config().get("enabled", True))


def board_liveness_interval_seconds() -> int:
    try:
        value = int(_config().get("interval_seconds", DEFAULT_INTERVAL_SECONDS))
    except (TypeError, ValueError):
        value = DEFAULT_INTERVAL_SECONDS
    return value if value > 0 else DEFAULT_INTERVAL_SECONDS


def check_and_emit_board_liveness(
    conn: sqlite3.Connection,
    *,
    board: str,
    now: Optional[int] = None,
    assignee_live: Optional[Callable[[str], bool]] = None,
    emit: bool = True,
) -> LivenessResult:
    """Check one board and (by default) emit its event — the dispatcher's entry point.

    The verdict is always computable (callers, dashboards and replays may ask even with the
    watcher switched off); only the *emission* honours ``kanban.board_liveness.enabled``, so the
    knob means the same thing wherever this entry point is called from.
    """
    result = check_board_liveness(conn, board=board, now=now, assignee_live=assignee_live)
    if emit and result.actionable and board_liveness_enabled():
        _emit_board_liveness(conn, board=board, state=result.state, result=result, now=now)
    return result


__all__: Iterable[str] = (
    "STATE_DEAD",
    "STATE_QUIET_BY_DESIGN",
    "STATE_HEALTHY",
    "EVENT_KINDS",
    "OPEN_STATUSES",
    "PARKED_BY_DESIGN_STATUSES",
    "SENTINEL_EXCLUSION",
    "DEFAULT_INTERVAL_SECONDS",
    "sentinel_id",
    "is_sentinel_id",
    "ensure_sentinel_task",
    "LivenessResult",
    "check_board_liveness",
    "_emit_board_liveness",
    "board_liveness_enabled",
    "board_liveness_interval_seconds",
    "check_and_emit_board_liveness",
)
