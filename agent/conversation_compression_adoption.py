"""Rotation-path adoption of a durable parent that outgrew the in-memory transcript.

Split out of :mod:`agent.conversation_compression` (FILE_LINES ratchet). The facade late-imports
:func:`_adopt_grown_durable_parent` at its only caller, the rotation-mode lease setup in
``compress_context``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent.conversation_compression import _CompressionLease

logger = logging.getLogger("agent.conversation_compression")


def _model_facing_rows(rows: Any) -> Any:
    """Durable rows as the gateway hands them to the agent: without the ``session_meta``/``system``
    bookkeeping rows (gateway/run.py), so a length comparison with the live transcript is like for like."""
    if not isinstance(rows, list):
        return rows
    return [m for m in rows if not (isinstance(m, dict) and m.get("role") in ("session_meta", "system"))]


def _adopt_grown_durable_parent(agent: Any, lease: _CompressionLease, messages: list) -> Optional[list]:
    """Return the durable parent transcript when it outgrew the in-memory snapshot.
    Rotation only (in-place never loses rows). The snapshot predates the lease: if durable grew, a writer
    committed a turn — ADOPT it (aborting wedged busy sessions forever). Length check only: in-memory edits of
    past turns are legal. Bookkeeping rows are left out of both the comparison and the adopted list: a gateway
    session's first-turn ``session_meta`` row is never in the live transcript, so counting it would adopt a
    snapshot with every later message one index off. Adoption advances ``lease.watermark`` to the snapshot's
    newest row, so publication clones only rows the snapshot does not already carry."""
    if lease.db is None or not lease.sid:
        return None
    durable_loader = getattr(type(lease.db), "get_messages_as_conversation", None)
    if not callable(durable_loader):
        return None
    durable_parent = _model_facing_rows(durable_loader(lease.db, lease.sid))
    if not (isinstance(durable_parent, list) and len(durable_parent) > len(messages)):
        return None
    # In-memory carries this turn's un-persisted user tail; flush it via the normal
    # rotation-boundary path before adopting, else skip adoption (would drop input).
    # The in-memory transcript carries the CURRENT turn's un-persisted user tail (anchored by
    # _persist_user_message_idx) that the durable snapshot read above does not contain yet. Flush that tail
    # through the normal rotation-boundary path (conversation_history = the already-durable prefix, #68196
    # boundary) BEFORE adopting, then re-read the durable parent so the adopted snapshot includes the live
    # input. If the flush fails (or the anchor is unknown), skip adoption entirely: replacing the in-memory
    # transcript with a snapshot that lacks the user's input would silently drop it from the summarized and
    # rotated history (#adopt-live-tail).
    _preflush_idx = getattr(agent, "_persist_user_message_idx", None)
    # No un-persisted tail means the transcript is fully durable: adopting the longer parent cannot drop input.
    _preflush_ok = True
    if isinstance(_preflush_idx, int) and 0 <= _preflush_idx < len(messages):
        _preflush_ok = False
        with contextlib.suppress(Exception):
            _preflush_ok = agent._flush_messages_to_session_db(messages, conversation_history=messages[:_preflush_idx])
    if not _preflush_ok:
        logger.warning(
            "compression: session=%s grew before lease (%d → %d msgs) but the pre-adoption flush of the "
            "live tail failed; skipping durable-snapshot adoption so un-persisted user input is kept",
            lease.sid, len(messages), len(durable_parent),
        )
        return None
    # Re-read after the flush so the adopted snapshot carries the just-persisted tail.
    durable_rows = durable_loader(lease.db, lease.sid, include_row_ids=True)
    if not isinstance(durable_rows, list):
        return None
    # Row ids from every snapshot row, bookkeeping included: the watermark covers what this read saw.
    adopted_row_ids = [rid for m in durable_rows if isinstance(m, dict) and isinstance(rid := m.pop("_row_id", None), int)]
    durable_parent = _model_facing_rows(durable_rows)
    if len(durable_parent) <= len(messages):
        return None
    if lease.watermark is not None and adopted_row_ids:
        lease.watermark = max(lease.watermark, *adopted_row_ids)
    logger.info(
        "compression: session=%s grew before lease (%d → %d msgs); adopting durable snapshot", lease.sid, len(messages),
        len(durable_parent),
    )
    return durable_parent
