"""Archive-watermark selection shared by the compressor and its prune/micro-compaction callers.

Split out of ``context_compressor`` (which is far over the file-length target) — the helper has no
dependencies on the compressor itself.
"""

from __future__ import annotations

from typing import Any, Optional


def _archive_watermark_for(session_db: Any, session_id: str, held: list[dict[str, Any]],
                           start_watermark: Optional[int] = None) -> Optional[int]:
    """The archive watermark for a commit that rewrites the history this process holds.

    Without one, ``archive_and_compact`` archives every active row, including turns another surface appended
    to the same session since this process loaded it (a Desktop session continued from Telegram) and rows
    that arrived while the commit was being built. Those never reached this process, so they would be marked
    summarized away with no summary holding them: still displayed and searchable, but gone from the model's
    history. Capping at the newest row the process held sends them down the
    concurrent-append path instead (cloned after the new set), the same rule the in-place compaction commit
    applies. *start_watermark* is the store's watermark from before any slow step; it defaults to now.
    A store without the watermark API keeps today's archive-everything commit.

    Raises :class:`StaleHeldHistory` when the newest held exact row is no longer active. The in-place commit
    falls back to the lease watermark there because its lease rules out an overlapping compaction; prune and
    micro-compaction hold no lease, so for them that fallback would publish a stale generation beside the one
    that won.
    """
    watermark_of = getattr(session_db, "get_active_message_watermark", None)
    if not callable(watermark_of) or not callable(getattr(session_db, "get_message_role", None)):
        return None
    if start_watermark is None:
        start_watermark = watermark_of(session_id)
    from agent.conversation_compression import held_archive_watermark
    return held_archive_watermark(session_db, session_id, start_watermark, held, stale_raises=True)
