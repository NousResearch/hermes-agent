"""Pure step-rollup helpers for live tool-progress surfaces (no I/O beyond the locale catalog, no platform imports).

A *run* is a stretch of consecutive tool steps between two pieces of assistant text. While a run is
live its summary is the current step plus a compact trail of the earlier ones
(``3 earlier: ✓ read_file · ✓ terminal ×2``); once text (or the end of the turn) closes it, the run
reads ``Ran N steps · K failed``. Slack's native task cards render these today; the helpers take
plain dicts so Discord, Telegram or Feishu text progress can share the same wording.

A step is a mapping with ``title``, ``status`` (``in_progress`` / ``complete`` / ``error``), and
optionally ``name`` (the short tool name used in the compact trail) and ``stopped`` (the step was
still running when the user stopped the turn).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, Optional, Sequence

from agent.i18n import t

TITLE_LIMIT = 250
DETAILS_LIMIT = 250

RUNNING, COMPLETE, ERROR = "in_progress", "complete", "error"


def mark(step: Mapping[str, Any]) -> str:
    return "✗" if step.get("status") == ERROR else "✓"


def short_name(step: Mapping[str, Any]) -> str:
    name = str(step.get("name") or "").strip()
    if name:
        return name
    return str(step.get("title") or "step").split(" - ", 1)[0].strip() or "step"


def compact(steps: Iterable[Mapping[str, Any]]) -> list[str]:
    """``['✓ read_file', '✓ terminal ×3', '✗ terminal']``: consecutive repeats merged."""
    out: list[list[Any]] = []
    for step in steps:
        label = f"{mark(step)} {short_name(step)}"
        if out and out[-1][0] == label:
            out[-1][1] += 1
        else:
            out.append([label, 1])
    return [f"{label} ×{n}" if n > 1 else label for label, n in out]


def fit_tail(items: Sequence[str], budget: int, sep: str = " · ") -> str:
    """Join ``items`` within ``budget`` chars, dropping from the FRONT (newest steps matter most)."""
    kept: list[str] = []
    length = 0
    for item in reversed(items):
        add = len(item) + (len(sep) if kept else 0)
        if length + add > budget - 2:
            return sep.join(["…", *kept])
        kept.insert(0, item)
        length += add
    return sep.join(kept)


@dataclasses.dataclass(frozen=True)
class RunSummary:
    """What one run's row says. ``closed`` runs carry ``tool_count``/``failed`` for the wording."""

    title: str
    status: str
    details: Optional[str]
    closed: bool
    tool_count: int
    failed: int
    earlier: int = 0
    earlier_trail: str = ""


def summarize_run(steps: Sequence[Mapping[str, Any]], *, closed: bool) -> RunSummary:
    """One row for a run of steps.

    Live (or still running): the newest running step is the title, earlier finished steps fold
    into ``earlier_trail``. Closed: counts for ``Ran N steps · K failed`` and the full trail as
    details. A closed run is an error only when a step was cut off by a stop: any other failure
    was followed by more work or by text, so the agent moved past it and Slack's red badge would
    read as a problem still going on. The failure count still shows in the title.
    """
    if not steps:
        return RunSummary(title="", status=COMPLETE, details=None, closed=closed, tool_count=0, failed=0)
    running = [s for s in steps if s.get("status") == RUNNING]
    failed = sum(1 for s in steps if s.get("status") == ERROR)
    if not closed or running:
        current = running[-1] if running else steps[-1]
        earlier = [s for s in steps if s is not current and s.get("status") != RUNNING]
        trail = fit_tail(compact(earlier), DETAILS_LIMIT) if earlier else ""
        return RunSummary(
            title=str(current.get("title") or short_name(current)), status=RUNNING if running else str(current.get("status")),
            details=None, closed=False, tool_count=len(steps), failed=failed, earlier=len(earlier),
            earlier_trail=trail,
        )
    status = ERROR if any(s.get("stopped") for s in steps) else COMPLETE
    full = " · ".join(f"{mark(s)} {s.get('title') or short_name(s)}" for s in steps)
    details = full if len(full) <= DETAILS_LIMIT else fit_tail(compact(steps), DETAILS_LIMIT)
    return RunSummary(
        title=str(steps[0].get("title") or short_name(steps[0])) if len(steps) == 1 else "",
        status=status, details=details, closed=True, tool_count=len(steps), failed=failed,
    )


@dataclasses.dataclass
class RunTracker:
    """Which steps belong to which run. ``close()`` at each text boundary; pure bookkeeping."""

    runs: list = dataclasses.field(default_factory=list)  # [[step_id, ...], ...]
    run_of: dict = dataclasses.field(default_factory=dict)  # step_id -> run index
    open: bool = False

    def add(self, step_id: str) -> int:
        if step_id in self.run_of:
            return self.run_of[step_id]
        if not self.open:
            self.runs.append([])
            self.open = True
        self.runs[-1].append(step_id)
        self.run_of[step_id] = len(self.runs) - 1
        return self.run_of[step_id]

    def close(self) -> Optional[int]:
        """Close the live run; returns its index when one was open."""
        if not self.open:
            return None
        self.open = False
        return len(self.runs) - 1

    def summaries(self, steps: Mapping[str, Mapping[str, Any]], *, final: bool = False):
        """``(run_index, RunSummary)`` per run; ``final`` treats every run as closed."""
        for index, ids in enumerate(self.runs):
            closed = final or not (self.open and index == len(self.runs) - 1)
            yield index, summarize_run([steps[i] for i in ids if i in steps], closed=closed)


def steps_title(count: int, failed: int) -> str:
    """``Ran N steps · K failed`` (the failure suffix only when K > 0)."""
    title = t("gateway.progress.task_card_done_one" if count == 1 else "gateway.progress.task_card_done", count=count)
    return title + (t("gateway.progress.task_card_failed_suffix", failed=failed) if failed else "")


def turn_totals(steps: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
    """``(steps, failed)`` across a whole turn, for the card's terminal title."""
    total = failed = 0
    for step in steps:
        total += 1
        failed += step.get("status") == ERROR
    return total, failed


__all__ = [
    "RunSummary", "RunTracker", "compact", "fit_tail", "mark", "short_name", "steps_title", "summarize_run", "turn_totals",
]
