"""Goal, loop + queued-prompt rows for the classic CLI live-work dock.

Sibling of ``cli_process_dock``: the dock paints an active ``/goal``, a standing ``/loop`` with its
countdown, and the prompts waiting in ``/queue`` around the subagent and process blocks, so they are
visible while the agent works instead of only through ``/goal status``, ``/loop status`` and
``/queue list``.
"""
from __future__ import annotations

import math
import time

# Queued prompts drain one per turn; past this many the dock shows a ``+N more`` line.
QUEUE_ROWS = 3


def goal_line(cli) -> str:
    """``GoalManager.status_line()`` for an active or paused goal, else ``''``.

    Reads only the manager the CLI already holds for its current session: the dock refreshes on
    the spinner thread every second and must never be the thing that opens state.db.
    """
    mgr = getattr(cli, "_goal_manager", None)
    if mgr is None or mgr.session_id != getattr(cli, "session_id", None) or not mgr.has_goal():
        return ""
    return mgr.status_line()


def countdown(seconds: float) -> str:
    """Time left until the next wakeup, rounded UP: whole minutes until the last one, then seconds.

    Minute steps keep an idle prompt's dock to one repaint a minute (the spinner thread repaints only
    when the dock text changes); per-second repaints of a 45-minute wait would fight terminal viewport
    restoration for nothing.
    """
    left = max(0, math.ceil(seconds))
    if left < 60:
        return f"{left}s"
    hours, minutes = divmod(math.ceil(left / 60), 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def loop_row(cli, now: float | None = None) -> tuple[str, str, str]:
    """``(badge, line, prompt)`` for an active or paused /loop, else ``('', '', '')``.

    ``line`` is the expanded dock row (``↻ loop · every 5m · 2 runs · next in 4m · <prompt>``),
    ``badge`` the collapsed heading's count fragment (``↻ loop next in 4m``). Same rule as :func:`goal_line`:
    only the manager the CLI already holds is read, never state.db.
    """
    mgr = getattr(cli, "_loop_manager", None)
    # One read of ``state``: the input thread's ``/loop stop`` sets it to None under this spinner-thread poll.
    s = mgr.state if mgr is not None and mgr.session_id == getattr(cli, "session_id", None) else None
    if s is None or s.status not in {"active", "paused"}:
        return "", "", ""
    runs = f"{s.ticks_fired}/{s.times} runs" if s.times else f"{s.ticks_fired} run{'s' if s.ticks_fired != 1 else ''}"
    if s.status == "paused":
        reason = f" · {s.paused_reason}" if s.paused_reason else ""
        return "⏸ loop paused", f"⏸ loop paused{reason} · {runs} · {s.prompt}", s.prompt
    if s.awaiting_response:
        when = "running now"
    else:
        remaining = s.next_due_at - (time.time() if now is None else now)
        when = f"next in {countdown(remaining)}" if remaining > 0 else "due, waiting for idle"
    return f"↻ loop {when}", f"↻ loop · {s.cadence_label()} · {runs} · {when} · {s.prompt}", s.prompt


def _prompt_text(item) -> str:
    if isinstance(item, tuple):  # (text, images) from an image-attached submit
        item = item[0]
    # TimelineNotification carries a compact title; voice/seeded sentinels wrap their text.
    text = getattr(item, "display_text", None) or getattr(item, "text", None) or str(item)
    return " ".join(str(text).split())


def queued_prompts(cli) -> list[str]:
    """One-line previews of the turns waiting in ``_pending_input``, oldest first."""
    pending = getattr(cli, "_pending_input", None)
    if pending is None:
        return []
    with pending.mutex:
        items = list(pending.queue)
    return [text for text in map(_prompt_text, items) if text]
