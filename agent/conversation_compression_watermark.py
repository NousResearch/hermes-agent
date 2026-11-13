"""The held-row watermark cap and its lease-less staleness gate (#124102).

Split out of :mod:`agent.conversation_compression` (FILE_LINES ratchet): the watermark math the
in-place compaction commit, the proactive prune and micro-compaction share. The origin module
imports :func:`held_archive_watermark` from here; ``_held_watermark`` (the agent-bound wrapper)
stays with its only caller in the origin module's commit path.
"""

from __future__ import annotations

from typing import Any, Optional


def held_archive_watermark(
    session_db: Any, session_id: str, watermark: Optional[int], messages: list, verbatim_tail: Optional[list] = None,
    *, stale_raises: bool = False,
) -> Optional[int]:
    """The in-place archive watermark, capped at the newest durable row the compressor was handed.

    The lease watermark is the newest row in state.db, but a surface compacts the history it holds, and that
    can be older: a Desktop/TUI or CLI ``/compress`` does not hold turns another surface appended to the same
    session since it loaded. Archived under the watermark, those rows would leave every surface's
    history and search, and the summary never saw them. Above the cap they take the concurrent-append path
    instead (cloned after the compacted set).

    Only while the held history is a live prefix of the session: its LAST row names an exact ``_row_id``, and
    that row is still active (after another surface compacted, the held rows are archived and every live row
    would be cloned beside the new summary). An id is exact while its dict is unchanged since it was loaded: a
    dict loaded from the DB is born carrying both the id and the persist marker, and a pass that rewrote its
    content drops the marker and keeps the id (the user/assistant merges in ``repair_message_sequence``, or a
    context engine that rewrites in place). Such a row absorbed later durable rows whose ids are gone from the
    held set, so its id no longer names what the summary covered: capped there, those rows would be cloned
    live beside a summary that already contains them. A trailing row of unknown provenance may be durable
    under the lease watermark without any stamp (the TUI model-switch marker, written with a bare
    ``append_message``); capped below it, the clone would land beside its own carried copy. A ``here N`` tail
    is marker-swept copies, so ``compress_now`` keeps a copy's id only when its source still carried the
    marker; their ids are trusted as given.

    *stale_raises*: the newest exact held row being inactive means another compaction already committed.
    Under the in-place lease that cannot overlap a live compaction, so the lease watermark is returned; a
    lease-less caller (prune, micro-compaction) passes ``True`` and gets :class:`StaleHeldHistory` instead,
    because for it the fallback would publish a stale generation beside the winner. It also catches the
    generation change an id/activity check cannot see: an in-place prune rewrites a row's body under its
    own id, so ``stale_raises`` additionally compares each exact persisted held row against its stored
    payload (through the store's row-version CAS / loader-lens compare — never raw storage bytes against
    a loaded view). That comparison runs for every exact persisted held row, so an unstamped trailing
    message (which caps the watermark through the early return below) cannot suppress it for the durable
    prefix.
    """
    if watermark is None:
        return None
    from agent.context_compressor import _DB_PERSISTED_MARKER, StaleHeldHistory

    def _exact_id(m: dict, copied: bool) -> Optional[int]:
        rid = m.get("_row_id")
        if not isinstance(rid, int) or isinstance(rid, bool) or rid <= 0:
            return None
        return rid if copied or m.get(_DB_PERSISTED_MARKER) else None

    ids = [_exact_id(m, False) for m in messages if isinstance(m, dict)]
    ids += [_exact_id(m, True) for m in (verbatim_tail or ()) if isinstance(m, dict)]
    held = [rid for rid in ids if rid is not None]
    if stale_raises and held:
        # An IN-PLACE commit (the proactive prune) keeps every row id and activity flag, so the liveness
        # check below cannot see it. The stored payloads are the only trace: a held row whose active row
        # now holds something else belongs to a generation this caller never saw, and publishing the held
        # copy would republish the pre-rewrite text over the winner (#124102). Runs before the
        # trailing-row early return: an unstamped trailing message may cap the watermark, but it must not
        # suppress the version check for the durable prefix this caller does hold exactly.
        rewritten_of = getattr(session_db, "rewritten_held_row_ids", None)
        if callable(rewritten_of):
            rewritten = rewritten_of(session_id, [
                m for m in messages
                if isinstance(m, dict) and _exact_id(m, False) is not None])
            if rewritten:
                raise StaleHeldHistory(
                    f"held row {min(rewritten)} of session {session_id} was rewritten in place")
    if not ids or ids[-1] is None:
        return watermark
    newest_held = max(held)
    # A lease-less caller checks liveness even when nothing was appended: a commit-time re-check against a
    # watermark read before the slow step never sees newest_held < watermark, yet the winner may have landed.
    if newest_held >= watermark and not stale_raises:
        return watermark
    if session_db.get_message_role(session_id, newest_held) is None:
        if stale_raises:
            raise StaleHeldHistory(f"held row {newest_held} of session {session_id} is no longer active")
        return watermark
    return min(newest_held, watermark)
