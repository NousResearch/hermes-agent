"""Whether anything is attached to the Hermes board that will actually run the work.

NOVA submits work and never executes it. Execution is the dispatcher's job: it claims
``ready`` tasks and shells out to ``hermes -p <assignee> chat -q``
(``hermes_cli/kanban_db_dispatch.py::_default_spawn``), which needs the full runtime and
the model credentials NOVA deliberately does not hold. So the dispatcher lives in another
process — inside the gateway by default (``kanban.dispatch_in_gateway``), or standalone
via ``hermes kanban daemon --force``.

That separation is the design. The defect it produced is that a control plane with no
dispatcher behind it accepts an objective, writes four durable tasks, answers 200, and
looks exactly like one that is about to start working.

**Why this is measured from the board and not from a process.**

Three other signals were considered and rejected, each for a reason worth keeping:

``kanban/.dispatcher.lock``
    ``flock(2)``, which has no read-only probe — ``F_GETLK`` is for POSIX record locks.
    Testing it means taking it, and a gateway booting in that instant would read
    ``contended`` and refuse to dispatch for its whole process lifetime. A health check
    that can stop the thing it measures is not a health check.
``gateway.pid`` / ``gateway_state.json`` PID liveness
    The dispatcher may run in a different container from this one. PIDs are not
    comparable across PID namespaces, so a live gateway reads as dead.
``gateway_state.json`` freshness
    Written on state transitions, not on a timer, so a healthy gateway running for hours
    carries a stale ``updated_at``. Freshness would report a working deployment as broken.

The board is the one artifact both processes genuinely share, over a bind mount, with no
namespace between them. And it carries the fact that actually matters: **a dispatcher's
entire job is that claimable work does not sit**. Work that has sat past its poll interval
is therefore proof, not inference — it is the same observation the operator makes by
refreshing the screen, done in SQL.

Read-only throughout, through the same URI mode ``work.py`` uses.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from nova.runtime.base import WorkExecutionHealth
from nova.runtime.hermes.work import _readonly, work_store_path

#: The dispatcher's default poll interval, from ``gateway/kanban_watchers_dispatcher.py``
#: (``kanban.dispatch_interval_seconds``, default 60).
DEFAULT_POLL_INTERVAL_SECONDS = 60.0

#: Multiple of the poll interval a claimable task must sit before absence is called
#: proven. Three ticks, so one slow tick, one restart or one contended lock is not an
#: alarm; the floor keeps a deployment that has tuned the interval down to seconds from
#: alarming on normal jitter.
STALL_INTERVALS = 3
MIN_STALL_SECONDS = 180.0

#: How recently a claim must have happened to count as positive evidence. Two ticks: one
#: to notice the work, one for the spawn to land.
ATTACHED_INTERVALS = 2
MIN_ATTACHED_SECONDS = 150.0

#: The lane a dispatcher claims from. ``todo`` items are waiting on dependencies and are
#: not claimable, so counting them would manufacture an alarm on a healthy board.
CLAIMABLE_STATUS = "ready"

_REMEDY = (
    "Run a dispatcher against this state directory: start the Hermes gateway (it hosts "
    "the dispatcher by default, config kanban.dispatch_in_gateway), or run "
    "`hermes kanban daemon --force --interval 60` standalone. The NOVA control-plane "
    "image does not contain one — it has neither the runtime's dependencies nor the "
    "model credentials a worker needs."
)


def _poll_interval(home: Path) -> float:
    """``kanban.dispatch_interval_seconds`` from this home's config, or the default.

    Read from the file rather than through ``hermes_cli.config`` so the answer describes
    the home NOVA governs, not whichever home this process happens to have inherited.
    """
    path = Path(home) / "config.yaml"
    if not path.is_file():
        return DEFAULT_POLL_INTERVAL_SECONDS
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — an unreadable config is a default, not a crash
        return DEFAULT_POLL_INTERVAL_SECONDS
    if not isinstance(loaded, Mapping):
        return DEFAULT_POLL_INTERVAL_SECONDS
    kanban = loaded.get("kanban")
    raw: Any = kanban.get("dispatch_interval_seconds") if isinstance(kanban, Mapping) else None
    try:
        interval = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_POLL_INTERVAL_SECONDS
    # The dispatcher applies the same floor before using it.
    return max(interval, 1.0)


def _scalar(connection: sqlite3.Connection, sql: str, params: tuple = ()) -> Any:
    try:
        row = connection.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def work_execution_health(home: Path, *, now: Optional[float] = None) -> WorkExecutionHealth:
    """Is a dispatcher attached to this home's board?

    ``now`` is injectable so a test can age a board without sleeping through a poll
    interval.
    """
    moment = time.time() if now is None else now
    interval = _poll_interval(home)
    stall_after = max(STALL_INTERVALS * interval, MIN_STALL_SECONDS)
    attached_within = max(ATTACHED_INTERVALS * interval, MIN_ATTACHED_SECONDS)

    path = work_store_path(Path(home))
    with _readonly(path) as connection:
        if connection is None:
            # No board yet is the normal state of a fresh deployment, and says nothing
            # about whether a dispatcher is running. Unobserved, not unhealthy.
            return WorkExecutionHealth(
                determined=False,
                poll_interval_seconds=interval,
                detail=(
                    f"No work store at {path}, so there is nothing to observe. This is "
                    "normal before the first objective is submitted."
                ),
            )

        waiting = _scalar(
            connection,
            "SELECT COUNT(*) FROM tasks WHERE status = ? AND claim_lock IS NULL",
            (CLAIMABLE_STATUS,),
        )
        oldest_created = _scalar(
            connection,
            "SELECT MIN(created_at) FROM tasks WHERE status = ? AND claim_lock IS NULL",
            (CLAIMABLE_STATUS,),
        )
        # A claim is the one thing only a dispatcher does. ``started_at`` is set when the
        # worker is spawned, so the most recent one dates the last time anything ran.
        last_claim = _scalar(connection, "SELECT MAX(started_at) FROM tasks")

    ready_waiting = int(waiting or 0)
    claim_age = (moment - float(last_claim)) if last_claim else None
    oldest_age = (moment - float(oldest_created)) if oldest_created else None

    if claim_age is not None and claim_age <= attached_within:
        return WorkExecutionHealth(
            attached=True,
            determined=True,
            mechanism="board-activity",
            ready_waiting=ready_waiting,
            oldest_ready_age_seconds=oldest_age,
            poll_interval_seconds=interval,
            detail=(
                f"A dispatcher claimed work {int(claim_age)}s ago, within the "
                f"{int(attached_within)}s window, so something is attached to this board."
            ),
        )

    if ready_waiting and oldest_age is not None and oldest_age >= stall_after:
        never = " Nothing has ever been claimed on this board." if not last_claim else ""
        return WorkExecutionHealth(
            attached=False,
            determined=True,
            mechanism="",
            ready_waiting=ready_waiting,
            oldest_ready_age_seconds=oldest_age,
            poll_interval_seconds=interval,
            detail=(
                f"{ready_waiting} task(s) have been ready and unclaimed for up to "
                f"{int(oldest_age)}s, past the {int(stall_after)}s a dispatcher polling "
                f"every {int(interval)}s would take. Nothing is claiming this board's "
                f"work.{never}"
            ),
            remedy=_REMEDY,
        )

    # Claimable work that has not waited long enough yet, or no claimable work at all.
    # Either way the board has not said anything, and saying it has would be the same
    # mistake in the other direction.
    if ready_waiting:
        detail = (
            f"{ready_waiting} task(s) are ready and have waited up to "
            f"{int(oldest_age or 0)}s, under the {int(stall_after)}s needed to conclude "
            "anything. Ask again after a poll interval."
        )
    else:
        detail = (
            "No claimable work on the board, so whether a dispatcher is attached cannot "
            "be observed from it."
        )
    return WorkExecutionHealth(
        determined=False,
        ready_waiting=ready_waiting,
        oldest_ready_age_seconds=oldest_age,
        poll_interval_seconds=interval,
        detail=detail,
    )
