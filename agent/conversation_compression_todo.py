"""Todo snapshot authority at a compression boundary."""

import contextlib
from typing import Any


def compaction_todo_snapshot(agent: Any) -> tuple[str | None, bool]:
    """Return the live rendering or an opaque fallback, plus whether it supersedes older snapshots."""
    snapshot = agent._todo_store.format_for_injection()
    fallback = getattr(agent, "_compaction_todo_snapshot", None)
    fallback = fallback if isinstance(fallback, str) else None
    # Non-empty stores (even all done), explicit structured clears, and opaque carry-forward blocks
    # supersede old snapshots. An unhydrated empty store must leave an existing snapshot alone.
    authoritative = fallback is not None or getattr(agent, "_compaction_todo_authoritative", False) is True
    has_items = getattr(agent._todo_store, "has_items", None)
    with contextlib.suppress(Exception):
        authoritative = authoritative or (bool(has_items()) if callable(has_items) else False)
    return snapshot or fallback, authoritative
