"""
gap.py — Deterministic current-vs-desired gap computation.

Turns a GoalDoc into a GapResult: how far along a goal is, whether it is on
pace for its target date, and a one-line human summary. Kept as precise
Python (not left to model reasoning) so "you're at 9%, target 15%, behind
pace" is the same every run.

Progress semantics:
- increase:  progress = (current - baseline) / (target - baseline)
- decrease:  progress = (baseline - current) / (baseline - target)
- maintain:  in-band -> 1.0, else scaled by distance from target
When baseline_value is absent it defaults to 0 for `increase` and to the
current value's starting side for `decrease` (progress = target / current).

Pace: with both start_date and target_date, elapsed_frac is the fraction of
the window that has passed; a goal is `ahead` / `on_track` / `behind` by
comparing progress against elapsed_frac (±10% band).

Stdlib-only. Python 3.11+.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

from _common import _as_number, parse_date, utcnow

if TYPE_CHECKING:
    from _common import GoalDoc

_PACE_BAND = 0.10
_MILESTONE_RE = re.compile(r"^\s*[-*]\s+\[([ xX])\]", re.MULTILINE)


@dataclass
class GapResult:
    """Computed state of a single goal.

    `progress` is 0..1 (may exceed 1 on overshoot). `remaining` is the
    direction-aware distance still to cover: positive = not there yet,
    negative = past target.
    """

    status: str
    quantifiable: bool
    progress: float | None = None
    remaining: float | None = None
    unit: str | None = None
    direction: str = "increase"
    pace: str = "unknown"          # ahead | on_track | behind | met | unknown
    elapsed_frac: float | None = None
    days_left: int | None = None
    milestones_done: int = 0
    milestones_total: int = 0
    summary: str = ""

    @property
    def progress_pct(self) -> int | None:
        return None if self.progress is None else round(self.progress * 100)

    @property
    def milestone_progress(self) -> float | None:
        if self.milestones_total == 0:
            return None
        return self.milestones_done / self.milestones_total


def count_milestones(body: str) -> tuple[int, int]:
    """Return (done, total) for `- [ ]` / `- [x]` checkboxes in the body."""
    marks = _MILESTONE_RE.findall(body or "")
    done = sum(1 for m in marks if m in ("x", "X"))
    return done, len(marks)


def _remaining_to_go(direction: str, target: float, current: float) -> float:
    """Distance still to cover *in the goal direction*.

    Positive means not there yet; negative means overshot past target. Kept
    direction-aware so a decrease goal at 190 toward 180 reports 10 to go
    (not -10), which is what any JSON consumer or agent wording relies on.
    """
    if direction == "decrease":
        return current - target
    if direction == "maintain":
        return abs(current - target)
    return target - current  # increase


def _raw_progress(direction: str, target: float, current: float, baseline: float | None) -> float:
    if direction == "maintain":
        if target == 0:
            return 1.0 if current == 0 else 0.0
        return max(0.0, 1.0 - abs(current - target) / abs(target))
    if direction == "decrease":
        if baseline is not None and baseline != target:
            return (baseline - current) / (baseline - target)
        return (target / current) if current else 0.0
    # increase
    base = baseline if baseline is not None else 0.0
    denom = target - base
    if denom == 0:
        return 1.0 if current >= target else 0.0
    return (current - base) / denom


def _elapsed_frac(start: date | None, target: date | None, today: date) -> float | None:
    if start is None or target is None or target <= start:
        return None
    span = (target - start).days
    gone = (today - start).days
    return max(0.0, min(1.0, gone / span))


def compute_gap(doc: GoalDoc, now: datetime | None = None) -> GapResult:
    """Compute the gap for a single goal. Pure; pass `now` to pin the clock."""
    today = utcnow(now).date()
    done, total = count_milestones(doc.body)

    # Terminal statuses report themselves; no pace math needed.
    if doc.status in ("achieved", "dropped", "paused"):
        res = GapResult(
            status=doc.status,
            quantifiable=doc.is_quantifiable(),
            milestones_done=done,
            milestones_total=total,
            unit=doc.unit,
        )
        res.summary = _summarize(doc, res)
        return res

    direction = doc.effective_direction()
    target = _as_number(doc.target_value)
    current = _as_number(doc.current_value)
    baseline = _as_number(doc.baseline_value)
    quantifiable = target is not None and current is not None

    progress: float | None = None
    remaining: float | None = None
    # Direct None-guard (not the `quantifiable` bool) so the type checker
    # narrows target/current to float for the arithmetic below.
    if target is not None and current is not None:
        progress = round(_raw_progress(direction, target, current, baseline), 4)
        remaining = round(_remaining_to_go(direction, target, current), 6)
    elif total:
        progress = round(done / total, 4)

    start = parse_date(doc.start_date) or parse_date(doc.created_at)
    target_date = parse_date(doc.target_date)
    elapsed = _elapsed_frac(start, target_date, today)
    days_left = (target_date - today).days if target_date else None

    pace = _classify_pace(progress, elapsed)

    res = GapResult(
        status=doc.status,
        quantifiable=quantifiable,
        progress=progress,
        remaining=remaining,
        unit=doc.unit,
        direction=direction,
        pace=pace,
        elapsed_frac=None if elapsed is None else round(elapsed, 4),
        days_left=days_left,
        milestones_done=done,
        milestones_total=total,
    )
    res.summary = _summarize(doc, res)
    return res


def _classify_pace(progress: float | None, elapsed: float | None) -> str:
    if progress is not None and progress >= 1.0:
        return "met"
    if progress is None or elapsed is None:
        return "unknown"
    if progress >= elapsed + _PACE_BAND:
        return "ahead"
    if progress < elapsed - _PACE_BAND:
        return "behind"
    return "on_track"


def _fmt_num(value: float | int | None) -> str:
    """Human number with thousands separators; trailing zeros trimmed on floats."""
    if value is None:
        return "?"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _summarize(doc: GoalDoc, res: GapResult) -> str:
    unit = f"{doc.unit}" if doc.unit else ""
    sep = "" if unit in ("%", "") else " "
    if res.status == "achieved":
        return "achieved"
    if res.status in ("paused", "dropped"):
        tail = ""
        if res.milestones_total:
            tail = f" ({res.milestones_done}/{res.milestones_total} milestones)"
        return f"{res.status}{tail}"
    # Milestones (the persisted "path") ride alongside the numeric gap so a
    # quantifiable goal's proposed next steps stay visible in gap/report.
    steps = f" · {res.milestones_done}/{res.milestones_total} milestones" if res.milestones_total else ""
    if res.quantifiable:
        cur = f"{_fmt_num(_as_number(doc.current_value))}{sep}{unit}"
        tgt = f"{_fmt_num(_as_number(doc.target_value))}{sep}{unit}"
        head = f"{res.progress_pct}% — {cur} → {tgt}"
        if res.pace == "met":
            return f"{head} — target met{steps}"
        if res.days_left is not None:
            when = f"{res.days_left}d left" if res.days_left >= 0 else f"{-res.days_left}d overdue"
            return f"{head}, {res.pace.replace('_', ' ')} ({when}){steps}"
        return f"{head}, {res.pace.replace('_', ' ')}{steps}"
    if res.milestones_total:
        return f"{res.milestones_done}/{res.milestones_total} milestones ({res.progress_pct}%)"
    return "active — no measurable target set"


# ---------------------------------------------------------------------------
# Roll-up across many goals
# ---------------------------------------------------------------------------


@dataclass
class DomainRollup:
    domain: str
    total: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    behind: list[str] = field(default_factory=list)      # goal titles behind pace
    ready_to_close: list[str] = field(default_factory=list)  # active but target met


def rollup(pairs: list[tuple[GoalDoc, GapResult]]) -> list[DomainRollup]:
    """Aggregate (goal, gap) pairs into per-domain summaries, sorted by domain."""
    domains: dict[str, DomainRollup] = {}
    for doc, res in pairs:
        r = domains.setdefault(doc.domain, DomainRollup(domain=doc.domain))
        r.total += 1
        r.by_status[res.status] = r.by_status.get(res.status, 0) + 1
        if res.status == "active" and res.pace == "behind":
            r.behind.append(doc.goal)
        if res.status == "active" and res.pace == "met":
            r.ready_to_close.append(doc.goal)
    return [domains[k] for k in sorted(domains)]
