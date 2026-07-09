"""Phase 3 — Archive lifecycle listener (registered on ``on_session_end``).

This module is the ONLY place archive-lifecycle logic lives. It is wired to
the existing plugin hook ``on_session_end(session_id, completed, interrupted,
model, platform)`` — it is NOT called from any close path. Close paths stay
clean ("Session ended.") and merely emit the hook.

Lifecycle, on purpose, is boring:

- On ``on_session_end`` we resolve the raw transcript path
  (``<HERMES_HOME>/sessions/<session_id>.jsonl``), and if it exists, enqueue it
  into the index's ``index_pending`` table. The actual indexing is decoupled.
- A fire-and-forget background thread drains the pending queue
  (``MemoryIndex.refresh_pending``). It NEVER blocks the caller and swallows
  any error. Memory can lag slightly; memory can never interrupt work.
- ``search()`` and ``archive_stats()`` also call ``refresh_pending()`` lazily,
  so even if the background thread never runs, the next query makes closed
  sessions searchable. No daemon required.

The memory subsystem only READS raw transcripts. It never mutates them
(ownership rule in docs/memory-archive-contract.md §0).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..plugins import get_plugin_manager
from .indexer import MemoryIndex

logger = logging.getLogger("hermes.memory.archive_lifecycle")

_HOOK_NAME = "on_session_end"
_FLUSH_LOCK = threading.Lock()


def _resolve_transcript(session_id: str) -> Path:
    """Resolve a session id to its raw transcript path (read-only)."""
    home = MemoryIndex().hermes_home
    if session_id.endswith(".jsonl"):
        candidate = Path(session_id)
        if candidate.is_absolute():
            return candidate
        return home / session_id
    return home / "sessions" / f"{session_id}.jsonl"


def _flush_pending() -> None:
    """Drain the pending queue in the background. Never blocks, never raises."""
    try:
        with _FLUSH_LOCK:
            MemoryIndex().refresh_pending()
    except Exception:  # noqa: BLE001 — background safety net must be silent
        logger.debug("archive lifecycle pending flush failed", exc_info=True)


def _on_session_end(session_id: str, **_kwargs: object) -> None:
    """Hook callback: enqueue the closed session's transcript, then flush.

    Must be fast and non-blocking. The heavy indexing happens in a thread.
    """
    try:
        path = _resolve_transcript(session_id)
        if not path.is_file():
            return
        MemoryIndex().enqueue(str(path))
        # Non-blocking flush: fire-and-forget. If it fails, the next search
        # or status call lazily drains pending anyway.
        t = threading.Thread(target=_flush_pending, daemon=True, name="mem-archive-flush")
        t.start()
    except Exception:  # noqa: BLE001 — a hook callback must never break finalize
        logger.debug("archive lifecycle enqueue failed for %s", session_id, exc_info=True)


def register_listener() -> None:
    """Register the archive-lifecycle hook with the plugin manager.

    Idempotent: registering twice is harmless because ``enqueue`` is an
    UPSERT and the flush is locked. Safe to call at import time.
    """
    try:
        pm = get_plugin_manager()
    except Exception:  # noqa: BLE001
        logger.debug("plugin manager unavailable; archive listener not registered")
        return
    # `register_hook` is the plugin-facing API; core code plugs directly into
    # the manager's hook registry (same dict the hook API writes to).
    pm._hooks.setdefault(_HOOK_NAME, []).append(_on_session_end)
    logger.debug("archive lifecycle listener registered on %s", _HOOK_NAME)


# Auto-register when this module is imported. Keeps wiring out of close paths.
try:
    register_listener()
except Exception:  # noqa: BLE001
    logger.debug("archive lifecycle auto-register failed", exc_info=True)
