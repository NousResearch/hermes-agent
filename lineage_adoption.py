"""Adoption of a session's compression lineage from another profile's store: the stranded-session heal for a profile bot whose rows accumulated in the default profile's store before session calls were routed by their target session."""

from __future__ import annotations

import logging
from typing import Any

from persistence.conversation.messages import TranscriptRead, TranscriptView
from persistence.conversation.sessions import END_REASON_ADOPTED
from persistence.housekeeping.maintenance import ImportResult, SessionExport

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

_EVERY_ROW = TranscriptRead(view=TranscriptView.ALL)


def _row_count(store, session_id: str) -> int:
    return len(store.messages.read_transcript(session_id, _EVERY_ROW))


def _import_entry(segment: SessionExport) -> dict[str, Any]:
    return {**segment.session.as_dict(), "messages": [message.as_wire_row() for message in segment.messages]}


def _result_dict(result: ImportResult) -> dict[str, Any]:
    """The ``import_sessions`` dict: the id lists only on success."""
    out: dict[str, Any] = {"ok": result.ok, "imported": result.imported, "skipped": result.skipped, "detached": result.detached}
    if result.ok:
        out.update(imported_ids=list(result.imported_ids), skipped_ids=list(result.skipped_ids))
    out["errors"] = list(result.errors)
    return out


def adopt_session_lineage(store, donor, session_id: str, *, retire_donor: bool = True) -> dict[str, Any]:
    """Adopt *session_id*'s full compression lineage from the *donor* store into *store*: the donor's lineage export with every row, imported here (runtime fields reset, already-present ids skipped, so it is idempotent). Every message row is carried with its archived state, so turns in-place compaction archived stay visible here and stay out of model context. With ``retire_donor`` and a complete adoption, donor rows are ARCHIVED (never deleted) with ``end_reason=END_REASON_ADOPTED``, deliberately not a recoverable reason, so resurrection cannot undo an adoption. Returns the ``import_sessions`` dict plus ``adopted`` and ``donor_retired`` (True only when EVERY segment retired)."""
    # Every row, not just live ones: the retired donor is unrecoverable, so any turn left behind
    # (compaction-archived history included) would be visible nowhere.
    payload = donor.maintenance.export_session_lineage(session_id, include_inactive=True)
    if payload is None:
        return {"ok": False, "adopted": False, "donor_retired": False,
                "error": f"session {session_id!r} not found in donor store"}
    segments = payload.segments

    # Divergence guard: a segment we will SKIP (already here) may have kept growing in
    # the donor after a partial adoption; retiring it would strand those messages
    # behind a non-recoverable archive. Still import, but refuse to retire.
    donor_ahead = False
    for segment in segments:
        seg_id = segment.session.id
        if store.sessions.get(seg_id) is None:
            continue
        donor_count = len(segment.messages)
        local_count = _row_count(store, seg_id)
        if donor_count > local_count:
            donor_ahead = True
            logger.warning("adoption divergence: donor segment %s has %d messages, "
                           "local copy has %d — donor will NOT be retired", seg_id, donor_count, local_count)

    result = _result_dict(store.maintenance.import_sessions([_import_entry(segment) for segment in segments]))
    imported = result["imported"]
    skipped = result["skipped"]
    adopted = result["ok"] and (imported + skipped) == len(segments)
    if not adopted:
        logger.warning("adoption of %s did not complete: imported=%s skipped=%s of %s segment(s); errors=%s",
                       session_id, imported, skipped, len(segments), result["errors"])

    donor_retired = False
    if adopted and retire_donor and not donor_ahead:
        donor_retired = all(_retire_donor_segment(store, donor, segment.session.id) for segment in segments)
    return {**result, "adopted": adopted, "donor_retired": donor_retired}


def _retire_donor_segment(store, donor, seg_id: str) -> bool:
    """Archive one adopted donor segment; False when skipped or failed. The divergence guard used export-time counts, so both stores are re-read right before stamping and donor growth never lands behind a non-recoverable archive (equal-count content divergence is accepted: the bytes stay in the donor either way). A retirement failure does not fail the adoption (a later resume retries idempotently), but never claims success it did not have."""
    try:
        donor_now = _row_count(donor, seg_id)
        local_now = _row_count(store, seg_id)
        if donor_now > local_now:
            logger.warning(
                "adoption divergence at retire time: donor segment %s grew to %d messages (local %d) — "
                "leaving donor unretired", seg_id, donor_now, local_now,
            )
            return False
        # The first end reason wins; reopen so the adoption boundary is stamped even on an ended segment.
        donor.sessions.reopen(seg_id)
        donor.sessions.end(seg_id, END_REASON_ADOPTED)
        # The session lives on in the adopting profile: not a delete for plugins.
        donor.sessions.set_archived(seg_id, True, notify=False)
        return True
    except Exception:
        logger.warning("failed to retire donor segment %s after adoption", seg_id, exc_info=True)
        return False
