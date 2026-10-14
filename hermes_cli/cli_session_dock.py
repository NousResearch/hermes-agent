"""Goal + queued-prompt rows for the classic CLI live-work dock.

Sibling of ``cli_process_dock``: the dock paints an active ``/goal`` and the prompts waiting in
``/queue`` under the subagent and process blocks, so both are visible while the agent works
instead of only through ``/goal status`` and ``/queue list``.
"""
from __future__ import annotations

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
