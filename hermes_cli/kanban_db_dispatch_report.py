"""Why a dispatch tick held ready work back: the one-line suppression summary.

Split out of ``hermes_cli.kanban_db_dispatch`` so that façade does not grow. The
``DispatchResult`` shape is imported for typing only — this module is reached from the
dispatcher's own tail import, so there is no runtime cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from hermes_cli.kanban_db_dispatch import DispatchResult


def describe_suppression(results: Iterable[Optional[DispatchResult]]) -> str:
    """One line naming why the tick(s) held ready work back, or ``""``.

    ``active_pr=1, recent_success=2, rate_limited=1, skipped_locked=1,
    memory_pressure=critical`` — the respawn-guard reasons counted per task
    plus the tick-level holds. Feeds the "dispatcher stuck" warnings of the
    CLI daemon and the embedded gateway dispatcher, which otherwise report a
    bare zero-spawn count while ``hermes kanban tail`` is the only place the
    guard reason is written (#111910).
    """
    counts: dict[str, int] = {}
    pressure: Optional[str] = None
    for res in results:
        if res is None:
            continue
        for _task_id, reason in res.respawn_guarded:
            counts[reason] = counts.get(reason, 0) + 1
        if res.rate_limited:
            counts["rate_limited"] = counts.get("rate_limited", 0) + len(res.rate_limited)
        if res.skipped_locked:
            counts["skipped_locked"] = counts.get("skipped_locked", 0) + 1
        if res.skipped_board_disabled:
            # Board-level refusal: the board is an estate/scratch board and the
            # tick touched nothing (card t_17c9c847). Without this the refusal is
            # invisible to the "dispatcher stuck" line.
            counts["board_not_dispatch_enabled"] = counts.get("board_not_dispatch_enabled", 0) + 1
        if res.memory_pressure:
            pressure = res.memory_pressure
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    if pressure:
        parts.append(f"memory_pressure={pressure}")
    return ", ".join(parts)
