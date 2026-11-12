"""The ``api_content`` sidecar backfill writers for SessionDB (#124102 FILE_LINES split).

Split out of :mod:`hermes_state_messages`: the two row-addressed sidecar UPDATEs the turn prologue
uses when the current turn's user row was materialized before the sidecar could be composed. Bound
into ``SessionDB`` via this mixin, so every existing ``session_db.set_message_api_content(...)``
call site and test patch seam keeps working unchanged.
"""

from __future__ import annotations

from typing import Any


class SessionApiContentMixin:
    """``SessionDB`` mixin: the ``api_content`` sidecar backfills."""

    def set_latest_user_api_content(self, session_id: str, content: Any, api_content: str) -> int:
        """Backfill the ``api_content`` sidecar onto the newest ACTIVE user row (0/1 rows). Preflight compaction
        inserts that row BEFORE the sidecar exists and the later persist identity-skips compacted dicts;
        without this a reload reopens the prompt-cache divergence. ``content`` match guards a racing rewrite.

        POSITIONAL, and only safe when the caller already knows the newest
        active user row IS the message it stamped. The content match is NOT
        sufficient on its own: repeated identical user turns ("ok", "y",
        "continue") make an OLDER row compare equal, so calling this before
        the current turn's row exists overwrites the previous turn's sidecar
        with this turn's bytes — durable wrong-bytes replay, a worse cache
        break than the missing sidecar. When the caller holds the durable row
        id (``_row_id``, synced onto the live dict by
        ``sync_flushed_message_markers`` and stamped by
        :meth:`_insert_message_rows`), use :meth:`set_message_api_content`
        instead — it addresses the exact row and cannot land on a neighbour.
        """
        from hermes_state_messages import _scrub_surrogates
        return self._write_rowcount(
            "UPDATE messages SET api_content = ? WHERE id = (SELECT id FROM messages "
            "WHERE session_id = ? AND role = 'user' AND active = 1 ORDER BY id DESC LIMIT 1"
            ") AND content IS ?",
            (_scrub_surrogates(api_content), session_id, self._encode_content(content)))

    def set_message_api_content(
        self, session_id: str, row_id: int, content: Any, api_content: str
    ) -> int:
        """Backfill the ``api_content`` sidecar onto ONE known durable row.

        Row-addressed counterpart to :meth:`set_latest_user_api_content`: the
        caller passes the ``_row_id`` the write path stamped on the live
        message dict, so the update cannot drift onto a neighbouring row that
        merely carries the same text.

        Used by the turn prologue whenever the current turn's user row was
        already materialized before the sidecar could be composed (in-place
        preflight compaction, a close/early flush that raced the prologue).
        The crash persist then marker-skips that message, so this is the only
        way the stamped bytes reach the store.

        ``active = 1`` and the ``content`` match stay as defensive guards: a
        row the compaction archived, or one a racing rewrite changed, is left
        untouched.
        """
        from hermes_state_messages import _scrub_surrogates
        if not session_id or isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
            return 0
        return self._write_rowcount(
            "UPDATE messages SET api_content = ? WHERE id = ? AND session_id = ? "
            "AND role = 'user' AND active = 1 AND content IS ?",
            (_scrub_surrogates(api_content), row_id, session_id, self._encode_content(content)))
