"""``session.transcript``: a stored conversation read without attaching (``gateway/session_stored_reads.py``).

A picker previews a conversation before it opens it: ``session.resume`` would subscribe the caller's
pane, and ``session.events.since`` replays only what an attached viewer missed. A poller follows a
conversation, or one of its delegated children (``subagent_id``), from the cursor it holds.
"""
from typing import Literal

from .base import Result
from .canonical_sessions import CanonicalSessionParams
from .common import TranscriptMessage
from .registry import canonical_method


class StoredTranscriptParams(CanonicalSessionParams):
    # A delegated child of ``session_id`` (a ``subagent.list`` row's ``subagent_id``): read its transcript.
    subagent_id: str | None = None
    # ``next_cursor`` of a previous page of this session (and child) in this order; omitted for the first page.
    cursor: str | None = None
    limit: int = 100
    # ``latest``: the first page is the newest rows and each next page is older.
    order: Literal['oldest', 'latest'] = 'oldest'
    # Image parts as stored (data URLs); by default they read ``[image]``, as a byte-bounded resume shows them.
    inline_images: bool = False


class StoredTranscriptRow(TranscriptMessage):
    """``row_kind: 'message'`` is a ``session.resume`` ``display_messages`` row. ``row_kind: 'compaction'``
    is a compaction handoff at its position: ``role: 'system'``, ``text`` the summary a reader sees (not
    the parts written for the model), ``compacted_messages``, ``timestamp`` and the stored ``row_id``."""

    row_kind: Literal['message', 'compaction']
    # How many messages the compaction summarized, when its handoff names it (compaction rows only).
    compacted_messages: int | None = None
    # A row too large for a page alone keeps only its role, kind, time, ids and the head of its text.
    truncated: bool | None = None


class StoredTranscriptResult(Result):
    session_id: str
    # Oldest first in either order.
    rows: list[StoredTranscriptRow]
    # Opaque, bound to this session, child and order. Oldest order: never null while the transcript is
    # available; pass it back for the next page and, once a page is empty (caught up), to poll for the
    # rows written since. Latest order: null after the oldest page.
    next_cursor: str | None
    # The rows start the transcript again from its first page and replace what the caller holds: on a
    # read without a cursor, when an oldest-order cursor's rows changed shape (an in-place
    # compaction, a rotation, a rewind, a rewrite or deletion of a covered row or of an earlier segment's row),
    # and in either order when appends took a rotated lineage past ``sessions.max_resume_messages``.
    reset: bool
    # Only with ``subagent_id``: the child read, whether it was found (a foreign or unknown child is
    # ``available: false`` with no rows and a null cursor), whether it is still running, and its own
    # stored session (the newest segment).
    subagent_id: str | None = None
    available: bool | None = None
    running: bool | None = None
    child_session_id: str | None = None


canonical_method('session.transcript', params=StoredTranscriptParams, result=StoredTranscriptResult,
                 doc="One page of a stored conversation's display rows across its compression lineage "
                     "(live rows and rows a compaction archived, each logical message once, and each "
                     "compaction handoff as a `row_kind: 'compaction'` row at its position), from the "
                     "oldest or the newest. Never attaches the caller. A page holds at most `limit` "
                     "(1-500) rows, compaction rows included, plus up to 64 tool results of the call at its "
                     "edge, and at most 2 MiB "
                     "of JSON, cut between tool groups (a single larger row keeps only its role, kind, time, "
                     "ids and the head of its text, and carries `truncated`). Oldest order: read on until "
                     "a page is empty, then poll with the same `next_cursor` for the rows written since; "
                     "`reset: true` means the rows the cursor covered changed shape, so the reply starts "
                     "again from the first page. Latest order: read on until `next_cursor` is null; its "
                     "cursor stays valid across later compactions and turns. A rotated lineage past "
                     "`sessions.max_resume_messages` pages only its current segment (a cursor from before it "
                     "crossed, or a latest-order cursor from before a compaction rotated that segment, answers "
                     "`reset: true`). With `subagent_id`: "
                     "a delegated child of the session (running: through the parent's roster; finished: "
                     "through its stored row, when its ancestry reaches this conversation); a foreign or "
                     "unknown child is `available: false`. Refusals: invalid_params (a cursor issued for "
                     "another session, child or order, or not by this read), not_found, permission_denied, "
                     "profile_mismatch.")
