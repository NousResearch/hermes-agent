"""Ready-queue admission — the one gate between ``todo`` and ``ready``.

A board whose ``ready`` lane is deeper than the fleet can drain in an hour does
not run more work: it makes the failure ENQUEUE something the account cannot move —
cards are written, ACKed and linked, while the ratio of open to closed cards
grows to 45,000:1 and the lane's own control channel (a card) waits in the same
queue it is meant to be piped from. Admission is the missing refusal: filing
into a full lane is refused, never silently.

Two functions are the whole mechanism, and everything else derives from them:

``decide(conn, ...)``
    The ONE predicate. Read-only: it answers "would this demand be admitted
    right now?", with the depth/budget snapshot that produced the answer.

``admit_to_ready(conn, task_id, ...)``
    The ONE writer of ``tasks.status = 'ready'`` for a *transition*. Every
    path that re-enters the ready population (dependency promotion, operator
    promotion, unblock, review round-trip, reclaim, reopen, transfer) goes
    through it, so ``admit_state``/``ready_since`` can never drift from the
    status they describe. ``create_task`` is the one exception that is not a
    transition: it evaluates the same predicate before its INSERT and writes
    the resulting columns with the row (see ``kanban_db.create_task``).

Budget
    Derived, not invented: the throughput the board has ACTUALLY been closing
    with — DISTINCT cards completed in the trailing ``admission_window_hours``
    — floored at ``admission_budget_floor`` and overridable outright by
    ``admission_budget``. A lane's budget is the SAME derivation over the
    lane's OWN completions (never a share of the board's), overridable per lane
    by ``admission_lane_budgets``; an unassigned card has no lane bound at all.
    Both overrides are explicit, and neither one leaks: the board pin does not
    set a lane's bound, and the lane bound is not a fraction of the board's.
    Lane budget binds BEFORE the global depth check: a full lane refuses even
    when the fleet has room.

Exemptions (exactly three, each reported as its own reason)
    * a ``todo`` card re-entering the lane it already held (re-entry — the
      demand was admitted once; refusing it would strand a running card's
      retry), 
    * P0 / fault work (``admission_p0_priority`` and above),
    * a consent ask (``admit_reason='consent'`` on the create surface).

State, on the task row
    ``admit_state`` is ``admitted`` or ``deferred`` (NULL only for the
    pre-mechanism backlog, which is reported and never mass-admitted), and
    ``ready_since`` is stamped on every ENTRY into the ready population and
    cleared on claim, so "how long has this been waiting" is a column read
    rather than an event replay. Both columns are cleared on archive, so a
    card that leaves the board is not miscounted if its id is ever reused.

Config keys (``kanban:`` section; the keys ship inert, ``admission_enabled_at``
unset means the mechanism is off and every path behaves exactly as before).
These 8 keys are the §5.1 contract, and
``tests/hermes_cli/test_kanban_admission.py`` asserts them against the §5.1
table copied as a literal, so a contract edit is a visible two-place change.

    admission_enabled_at    unset  epoch; unset = OFF
    admission_budget        unset  >0 pins the BOARD budget (budget_source=pin_override)
    admission_window_hours     24  trailing window the budget derives from
    admission_budget_floor      5  floor under the derived budget
    admission_p0_priority      90  priority >= this is a P0 fault (exempt)
    admission_lane_budgets  unset  {lane: int} overrides; unassigned = no lane bound
    ageing_warn_hours          24  routine tier, in hours spent ready
    ageing_escalate_hours      72  escalate tier, in hours spent ready
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# ``tasks.admit_state`` values. NULL is the pre-mechanism backlog.
ADMITTED = "admitted"
DEFERRED = "deferred"

# The reasons a decision can carry; every one is a reportable token, never prose.
REASON_DISABLED = "admission_off"
REASON_RE_ENTRY = "re-entry"
REASON_EXEMPT_P0 = "exempt_p0_fault"
REASON_EXEMPT_CONSENT = "exempt_consent"
REASON_UNDER_BUDGET = "under_budget"
REASON_OVER_BUDGET = "over_budget"

# ``budget_source`` tokens. SOURCE_COHORT must name the derivation actually
# used, so it says "x1": the budget is the cohort unscaled (§5.1).
SOURCE_PIN = "pin_override"
SOURCE_FLOOR = "floor_min"
SOURCE_COHORT = "completed_window_1x"

# ``report['disposition']`` tokens the create surfaces expose (A4 / §3.4).
DISPOSITION_CREATED = "created"
DISPOSITION_PARKED = "parked"
DISPOSITION_DEDUPED = "deduped"
DISPOSITION_COMMENT_ON_ORIGIN = "comment_on_origin"

# Config defaults. Kept here (not only in ``config_defaults``) so the kernel is
# correct on a board whose config.yaml predates the keys. This is the same 8-key
# §5.1 contract as ``DEFAULT_CONFIG["kanban"]``, and the namesake parity test in
# ``tests/hermes_cli/test_kanban_admission.py`` pins both to the §5.1 table.
DEFAULTS: dict[str, Any] = {
    "admission_enabled_at": None,
    "admission_budget": None,
    "admission_window_hours": 24,
    "admission_budget_floor": 5,
    "admission_p0_priority": 90,
    "admission_lane_budgets": None,
    "ageing_warn_hours": 24,
    "ageing_escalate_hours": 72,
}


def kanban_config() -> dict:
    """The ``kanban:`` section of ``config.yaml``, or ``{}`` when unreadable.

    Never raises: a board that cannot read its config must fall back to the
    built-in defaults rather than refuse every filing.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return (cfg.get("kanban", {}) if isinstance(cfg, dict) else {}) or {}
    except Exception:
        return {}


def _num(cfg: dict, key: str, cast, unset=0):
    """One config value, cast.

    ``DEFAULTS[key]`` is what an absent, empty or null key means — that is the
    whole reason the module carries defaults: a board whose config.yaml predates
    the keys must still be governed by the §5.1 values. ``unset`` is what a key
    with no numeric default (``admission_budget``, ``admission_enabled_at``) and
    a malformed value fall back to; for those, unset IS a value.
    """
    default = DEFAULTS[key]
    val = cfg.get(key, default)
    if val is None or val == "":
        val = default
    fallback = cast(default) if default is not None else unset
    if val is None:
        return fallback
    try:
        return cast(val)
    except (TypeError, ValueError):
        return fallback


def _int_cfg(cfg: dict, key: str) -> int:
    return _num(cfg, key, int)


def _float_cfg(cfg: dict, key: str) -> float:
    return _num(cfg, key, float)


def lane_budgets(cfg: Optional[dict] = None) -> dict[str, int]:
    """``admission_lane_budgets`` as ``{lane: int}``; ``{}`` when unset.

    THE per-lane override surface (§5.1). A malformed entry, or a value that is
    not a mapping at all, is dropped rather than allowed to cap a lane at a
    number nobody wrote — a lane with no entry keeps its own derivation.
    Negative values clamp to 0, which reads as "no lane bound" everywhere else.
    """
    cfg = kanban_config() if cfg is None else cfg
    raw = cfg.get("admission_lane_budgets", DEFAULTS["admission_lane_budgets"])
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for lane, val in raw.items():
        try:
            out[str(lane).strip()] = max(0, int(val))
        except (TypeError, ValueError):
            continue
    return out


def admission_enabled_at(cfg: Optional[dict] = None) -> int:
    """Epoch the mechanism was switched on, or 0 when it is OFF (the default)."""
    return max(0, _int_cfg(cfg if cfg is not None else kanban_config(), "admission_enabled_at"))


@dataclass(frozen=True)
class Admission:
    """One decision, with the snapshot that produced it."""

    admitted: bool
    reason: str
    lane: Optional[str] = None
    depth: int = 0
    budget: int = 0
    budget_source: str = ""
    lane_depth: int = 0
    lane_budget: int = 0
    exempt: bool = False

    def as_payload(self) -> dict:
        """The ``admit`` block recorded on the transition event."""
        return {
            "state": ADMITTED if self.admitted else DEFERRED,
            "reason": self.reason,
            "lane": self.lane,
            "depth": int(self.depth),
            "budget": int(self.budget),
            "lane_depth": int(self.lane_depth),
            "lane_budget": int(self.lane_budget),
            "exempt": bool(self.exempt),
        }


# --- Counting ---------------------------------------------------------------


def _ready_depth(conn, lane: Optional[str] = None) -> int:
    if lane:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE status = 'ready' AND assignee = ?",
            (lane,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE status = 'ready'"
        ).fetchone()
    return int((row["n"] if row else 0) or 0)


def completed_in_window(conn, hours: float, *, lane: Optional[str] = None) -> int:
    """Distinct cards completed in the trailing ``hours``.

    Counted from the ``completed`` event log rather than ``tasks.completed_at``
    so the cohort a board has been closing with is not retroactively shrunk
    when an old ``done`` card is archived (the auto-archive ticker would
    otherwise silently starve every lane).
    """
    cutoff = int(time.time()) - int(max(0.0, float(hours)) * 3600)
    sql = (
        "SELECT COUNT(DISTINCT e.task_id) AS n FROM task_events e "
        "JOIN tasks t ON t.id = e.task_id "
        "WHERE e.kind = 'completed' AND e.created_at >= ?"
    )
    params: list[Any] = [cutoff]
    if lane:
        sql += " AND t.assignee = ?"
        params.append(lane)
    row = conn.execute(sql, tuple(params)).fetchone()
    return int((row["n"] if row else 0) or 0)


def ready_queue_budget(conn, *, lane: Optional[str] = None, cfg: Optional[dict] = None) -> tuple[int, str]:
    """``(budget, budget_source)`` — the ONE derivation, cohort-derived.

    ``lane=None`` is the board budget; a lane gets the SAME derivation over its
    OWN completions, so a lane's throughput sets its own ceiling. §5.1: the
    cohort is multiplied by nothing — a hidden multiplier is a deviation
    ``budget_source`` cannot report, and the sanctioned deviation is
    ``admission_budget``, which reports ``pin_override``.

    The pin is the BOARD's deviation and stops there: a lane's bound is its own
    derivation (A5), overridden only by the explicit ``admission_lane_budgets``
    map — otherwise a board pin would silently erase the per-lane structure.
    """
    cfg = kanban_config() if cfg is None else cfg
    if lane is None:
        pin = _int_cfg(cfg, "admission_budget")
        if pin > 0:
            return pin, SOURCE_PIN
    hours = _float_cfg(cfg, "admission_window_hours")
    cohort = completed_in_window(conn, hours, lane=lane)
    floor = _int_cfg(cfg, "admission_budget_floor")
    if cohort < floor:
        return floor, SOURCE_FLOOR
    return cohort, SOURCE_COHORT


def lane_budget(conn, lane: Optional[str], *, cfg: Optional[dict] = None) -> int:
    """A lane's ready budget; 0 = NO lane bound.

    A5/§5.1: the lane's bound is the lane's OWN derivation (its completions in
    the window, floored), never a share of the board's — a lane that closed
    nothing gets the floor, not the busy lane's number. ``admission_lane_budgets
    [lane]`` overrides it explicitly, and an UNASSIGNED demand (``lane``
    None/empty) has no lane bound at all: the board bound alone applies.
    """
    cfg = kanban_config() if cfg is None else cfg
    name = (str(lane).strip() or "") if lane else ""
    if not name:
        return 0
    overrides = lane_budgets(cfg)
    if name in overrides:
        return overrides[name]
    budget, _ = ready_queue_budget(conn, lane=name, cfg=cfg)
    return budget


def is_p0_fault(priority: Any, *, cfg: Optional[dict] = None) -> bool:
    cfg = kanban_config() if cfg is None else cfg
    try:
        return int(priority or 0) >= _int_cfg(cfg, "admission_p0_priority")
    except (TypeError, ValueError):
        return False


# --- The predicate ----------------------------------------------------------


def decide(
    conn, *, lane: Optional[str] = None, priority: Any = 0,
    admit_reason: Optional[str] = None, re_entry: bool = False,
    cfg: Optional[dict] = None,
) -> Admission:
    """Would this demand be admitted right now? Read-only; the ONE predicate.

    Precedence: admission OFF, then re-entry, then the two exemptions, then the
    lane bucket, then the board depth (the lane binds first on purpose).

    ``Admission`` carries the BOARD budget in ``budget`` and the demand's own
    lane bound in ``lane_budget`` (0 = no lane bound), so a lane-scoped number
    is never reported as the board's.
    """
    cfg = kanban_config() if cfg is None else cfg
    lane = (str(lane).strip() or None) if lane else None
    depth = _ready_depth(conn)
    lane_depth = _ready_depth(conn, lane) if lane else 0
    budget, source = ready_queue_budget(conn, cfg=cfg)
    lbudget = lane_budget(conn, lane, cfg=cfg)

    if not admission_enabled_at(cfg):
        return Admission(True, REASON_DISABLED, lane, depth, budget, source, lane_depth, lbudget)
    if re_entry:
        return Admission(True, REASON_RE_ENTRY, lane, depth, budget, source, lane_depth, lbudget, exempt=True)
    if is_p0_fault(priority, cfg=cfg):
        return Admission(True, REASON_EXEMPT_P0, lane, depth, budget, source, lane_depth, lbudget, exempt=True)
    if str(admit_reason or "").strip().lower() == "consent":
        return Admission(True, REASON_EXEMPT_CONSENT, lane, depth, budget, source, lane_depth, lbudget, exempt=True)
    if lbudget and lane_depth >= lbudget:
        return Admission(False, REASON_OVER_BUDGET, lane, depth, budget, source, lane_depth, lbudget)
    if depth >= budget:
        return Admission(False, REASON_OVER_BUDGET, lane, depth, budget, source, lane_depth, lbudget)
    return Admission(True, REASON_UNDER_BUDGET, lane, depth, budget, source, lane_depth, lbudget)


# --- The writer -------------------------------------------------------------


def _task_admission_row(conn, task_id: str):
    return conn.execute(
        "SELECT assignee, priority, status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()


def admit_to_ready(
    conn, task_id: str, *, entry_kind: str, lane: Optional[str] = None,
    priority: Any = None, admit_reason: Optional[str] = None, re_entry: bool = False,
    now: Optional[int] = None, cfg: Optional[dict] = None,
    from_status: Optional[str] = None,
) -> Admission:
    """Decide AND write the landing — the ONE writer of ``status='ready'``.

    MUST be called with the caller's write txn already open. On refusal the
    task keeps its current status, gets ``admit_state='deferred'`` and loses
    ``ready_since``: the demand stays parked where it is rather than entering
    the lane. Returns the :class:`Admission` so the caller can record the
    snapshot on the transition event it appends.

    ``from_status`` constrains the optimistic write to the phase the caller
    read (e.g. ``'todo'``), so a landing cannot overwrite a status change it
    did not make.
    """
    row = _task_admission_row(conn, task_id)
    if row is not None:
        if lane is None:
            lane = row["assignee"]
        if priority is None:
            priority = row["priority"]
    adm = decide(conn, lane=lane, priority=priority, admit_reason=admit_reason,
                 re_entry=re_entry, cfg=cfg)
    ts = int(time.time()) if now is None else int(now)
    if adm.admitted:
        sql = ("UPDATE tasks SET status = 'ready', admit_state = ?, ready_since = ? "
               "WHERE id = ? AND status != 'archived'")
        params: list[Any] = [ADMITTED, ts, task_id]
        if from_status:
            sql += " AND status = ?"
            params.append(from_status)
        conn.execute(sql, tuple(params))
    else:
        conn.execute(
            "UPDATE tasks SET admit_state = ?, ready_since = NULL WHERE id = ?",
            (DEFERRED, task_id),
        )
    return adm


def reenter_ready(conn, task_ids: Iterable[str], *, now: Optional[int] = None) -> int:
    """Stamp ready-queue RE-ENTRY for writers that flip status directly.

    Re-entry is exempt from the budget by construction (the demand was already
    admitted), so these sites only need the ageing stamp — a card that comes
    back into the lane after a crash must read as "waiting again", not as
    "waiting since it first landed". ``WHERE status = 'ready'`` keeps the call
    safe on the ``review`` branch of the same retry. Returns the number of rows
    stamped.
    """
    ts = int(time.time()) if now is None else int(now)
    stamped = 0
    for tid in task_ids:
        cur = conn.execute(
            "UPDATE tasks SET admit_state = COALESCE(admit_state, ?), ready_since = ? "
            "WHERE id = ? AND status = 'ready'",
            (ADMITTED, ts, tid),
        )
        stamped += cur.rowcount or 0
    return stamped


def record_refusal_on_origin(
    conn, origin_id: str, adm: "Admission", *, lane: Optional[str] = None,
    entry_kind: str = "create", title: Optional[str] = None,
) -> int:
    """Hand a refusal back to the card the filing came from.

    An ordinary filing that names its origin creates NO card when the lane is
    over budget: the refusal lands on the origin card as a comment (plus its
    ``commented`` event), so the demand is visible where the work happened
    rather than becoming an orphan row. Returns the comment id.
    """
    from hermes_cli import kanban_db as _kb
    payload = adm.as_payload()
    quoted = f'"{title.strip()}"' if title and title.strip() else "a filing"
    body = (
        f"Ready-queue admission refused {quoted} ({entry_kind}): "
        f"{adm.reason} — depth {adm.depth}/{adm.budget} "
        f"(source {adm.budget_source}), lane {lane or '-'} "
        f"{adm.lane_depth}/{adm.lane_budget}. No card was created; the work stays "
        f"here until the lane drains. Re-file it later, or file with "
        f"admit_reason=consent if it must not wait."
    )
    if payload.get("exempt"):
        body += " (exempt)"
    return _kb.add_comment(conn, origin_id, "kanban_admission", body)


# --- Reporting --------------------------------------------------------------


def queue_state(conn) -> dict:
    """The read surface for the board's admission state (§5.2, frozen keys)."""
    cfg = kanban_config()
    now = int(time.time())
    budget, source = ready_queue_budget(conn)  # board budget + its real source
    hours = _float_cfg(cfg, "admission_window_hours")
    window_completed = completed_in_window(conn, hours)
    drain_per_hour = round(window_completed / hours, 3) if hours > 0 else 0.0
    depth = _ready_depth(conn)
    lanes: dict[str, dict] = {}
    rows = conn.execute(
        "SELECT assignee, COUNT(*) AS n FROM tasks "
        "WHERE status = 'ready' AND assignee IS NOT NULL GROUP BY assignee"
    ).fetchall()
    for row in rows:
        lane = str(row["assignee"])
        lanes[lane] = {
            "depth": int(row["n"] or 0),
            "budget": lane_budget(conn, lane, cfg=cfg),
        }
    deferred_row = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE admit_state = ?", (DEFERRED,)
    ).fetchone()
    admitted_row = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE admit_state = ?", (ADMITTED,)
    ).fetchone()
    enabled_at = admission_enabled_at(cfg)
    bypass_sql = "SELECT COUNT(*) AS n FROM tasks WHERE status = 'ready' AND admit_state IS NULL"
    backlog_sql = bypass_sql
    if enabled_at:
        bypass_sql += " AND ready_since IS NOT NULL AND ready_since >= ?"
        backlog_sql += " AND (ready_since IS NULL OR ready_since < ?)"
        bypass = int((conn.execute(bypass_sql, (enabled_at,)).fetchone() or {"n": 0})["n"] or 0)
        backlog = int((conn.execute(backlog_sql, (enabled_at,)).fetchone() or {"n": 0})["n"] or 0)
    else:
        bypass = 0
        backlog = int((conn.execute(bypass_sql).fetchone() or {"n": 0})["n"] or 0)
    # A filing with no origin has nowhere to hand its refusal back to, so it
    # parks on its own card; the count is what tells the operator how much of
    # the deferred population is at risk of being forgotten.
    fallback_row = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks t WHERE t.admit_state = ? "
        "AND NOT EXISTS (SELECT 1 FROM task_links l WHERE l.child_id = t.id)",
        (DEFERRED,),
    ).fetchone()
    age = ageing(conn)
    return {
        "depth": depth,
        "budget": budget,
        "budget_source": source,
        "drain_per_hour": drain_per_hour,
        "window_hours": hours,
        "lanes": lanes,
        "deferred_count": int((deferred_row or {"n": 0})["n"] or 0),
        "admitted_count": int((admitted_row or {"n": 0})["n"] or 0),
        "bypass_count": bypass,
        "pre_mechanism_backlog": backlog,
        "fallback_count": int((fallback_row or {"n": 0})["n"] or 0),
        "oldest_ready_seconds": age["oldest_ready_seconds"],
        "ageing_warn_count": age["ageing_warn_count"],
        "ageing_escalate_count": age["ageing_escalate_count"],
        "ageing_oldest": age["ageing_oldest"],
        "now": now,
    }


def ageing(conn, *, cfg: Optional[dict] = None) -> dict:
    """Wait-time report from ``ready_since`` (a column read, not an event replay).

    §5.1 units: the tiers are HOURS in ready (``ageing_warn_hours`` /
    ``ageing_escalate_hours``), compared against seconds-in-ready * 3600. A tier
    of 0 (or less) is off.
    """
    cfg = kanban_config() if cfg is None else cfg
    warn = int(_float_cfg(cfg, "ageing_warn_hours") * 3600)
    escalate = int(_float_cfg(cfg, "ageing_escalate_hours") * 3600)
    now = int(time.time())
    rows = conn.execute(
        "SELECT id, ready_since FROM tasks "
        "WHERE status = 'ready' AND ready_since IS NOT NULL "
        "ORDER BY ready_since ASC"
    ).fetchall()
    waits = [(str(r["id"]), max(0, now - int(r["ready_since"]))) for r in rows]
    return {
        "oldest_ready_seconds": waits[0][1] if waits else 0,
        "ageing_warn_count": sum(1 for _, w in waits if warn > 0 and w >= warn),
        "ageing_escalate_count": sum(1 for _, w in waits if escalate > 0 and w >= escalate),
        "ageing_oldest": [
            {"id": tid, "ready_since": now - w} for tid, w in waits[:3]
        ],
    }
