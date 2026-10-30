"""In-place row-rewrite persistence for the proactive prune (#124102).

Split out of :mod:`hermes_state_messages` (FILE_LINES ratchet): the prune's own commit primitive and
its lease-less staleness probe. Both are ``SessionMessagesMixin`` methods; this sibling defines a
``SessionPruneRewriteMixin`` the facade binds into the same MRO slot, so every existing
``session_db.rewrite_pruned_rows(...)`` / ``rewritten_held_row_ids(...)`` call site and test patch
seam keeps working unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.context_compressor import _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY
from agent.message_metadata import DB_ROW_SNAPSHOT
from hermes_state_common import _id_chunks, _placeholders
from hermes_state_messages import _INSERT_MESSAGE_SQL, _parse_tool_calls

# _INSERT_MESSAGE_SQL's columns in bind order, so a row rewrite can diff _message_row_params by name.
_MESSAGE_ROW_COLUMNS = tuple(
    column.strip() for column in _INSERT_MESSAGE_SQL.split("(", 1)[1].split(")", 1)[0].split(","))
# A prune only changes what a message says; who said it, when, and in which row never move.
_PRUNE_FIXED_COLUMNS = frozenset({"session_id", "role", "tool_call_id", "timestamp", "active", "display_identity"})


class SessionPruneRewriteMixin:
    """``SessionDB`` mixin: the proactive prune's in-place row rewrite and held-row staleness probe."""

    def rewritten_held_row_ids(self, session_id: str, held: list[dict[str, Any]]) -> list[int]:
        """Ids among *held* message dicts whose ACTIVE row no longer holds the payload the dict holds.

        An in-place commit (the proactive prune's ``rewrite_pruned_rows``) changes what a row says and
        keeps its id and its ``active`` flag, so an id/activity staleness check cannot see it. A lease-less
        writer holding the pre-rewrite bodies must not publish them: that archives rows it never compared
        and republishes the stale text over the newer generation (#124102).

        The comparison is representation-safe, never raw storage against a loaded view. A dict's content
        passed through the loader lens (``_loaded_view_content`` sanitizes and strips user/assistant
        strings), so the stored bytes are compared through the same lens; when the dict carries the
        stored-row CAS digest (``DB_ROW_SNAPSHOT``, see ``agent.transcript_repair``) that wins instead,
        covering every owned column including assistant ``tool_calls`` arguments. Absent or inactive
        rows are NOT reported — that is the activity check's job and it has its own fallback rules.
        Every model-transcript role is compared (user/assistant/tool): the callers gate the FULL
        writer (``archive_and_compact`` via the prune fallback and micro-compaction), which re-inserts
        held rows of those roles verbatim — a concurrently rewritten user row (``set_user_message_content``,
        the repair batch's in-place rewrite) is as much a lost race as a rewritten tool result (#124102).
        Bookkeeping roles (``session_meta``/``system``) never reach the model-facing list and stay excluded.
        """
        return self._read_retrying_ioerr(
            lambda conn: self.held_row_mismatches(conn, session_id, held))

    def held_row_mismatches(self, conn, session_id: str, held: list[dict[str, Any]]) -> list[int]:
        """``rewritten_held_row_ids`` against *conn*: the same CAS compare, on a connection the
        caller already holds (inside a ``BEGIN IMMEDIATE``, the full writer's commit window — a
        precheck outside that lock misses a write that lands between check and commit, and the
        full writer would then archive that newer row and republish the stale held text, #124102).

        Chunked like ``stamp_inserted_row_snapshots``: one ``id IN (...)`` per ``_id_chunks`` batch,
        so a long history stays under SQLite's host-parameter limit instead of failing the pass.
        Rows without an exact ``_row_id`` but with both the persist marker and a CAS snapshot are
        resolved by their logical ``message_uid`` (``idx_messages_session_uid``): the unaddressed
        live-replay loaders stamp exactly that pair, and a rewritten-in-place row is as lost a race
        there as a row-addressed one (#124102).
        """
        from agent.message_metadata import message_uid_or_none
        from agent.transcript_repair import transcript_row_snapshot

        def _exact_id(message: dict[str, Any]) -> Optional[int]:
            rid = message.get("_row_id")
            if isinstance(rid, int) and not isinstance(rid, bool) and rid > 0:
                return rid
            return None

        exact = [
            (message, _exact_id(message))
            for message in held
            if isinstance(message, dict) and message.get(_DB_PERSISTED_MARKER_KEY)
            and message.get("role") in ("user", "assistant", "tool")]
        by_id = {row_id: message for message, row_id in exact if row_id is not None}
        # The unaddressed replay shape: snapshot + marker, no _row_id. Only rows the caller can
        # name with durable identity join the fence; anything else keeps the loader-lens behavior
        # of the activity check (absence of evidence is not permission to publish, but a dict the
        # store cannot name at all has always belonged to the caller's fallback rules).
        by_uid = {}
        for message, row_id in exact:
            if row_id is None and isinstance(message.get(DB_ROW_SNAPSHOT), str):
                uid = message_uid_or_none(message)
                if uid is not None:
                    by_uid[uid] = message
        if not session_id or not (by_id or by_uid):
            return []
        stored: dict[int, Any] = {}
        for chunk in _id_chunks(by_id):
            for row in conn.execute(
                f"SELECT * FROM messages WHERE session_id = ? AND id IN ({_placeholders(chunk)})",
                (session_id, *chunk)).fetchall():
                stored[int(row["id"])] = row
        by_uid_row: dict[str, Any] = {}
        for uid, message in by_uid.items():
            # Newest active row, same rule as the loader's _active_logical_message_row: a logical
            # message_uid can have several active physical rows (a carried or concurrent compaction
            # cloning one back beside its archived twin), and the loader stamps the held dict from
            # the newest. A bare fetchone() could return an older copy whose payload still matches
            # the held digest — the newest row was rewritten by another writer, and the lease-less
            # fallback would republish the stale held payload over that newer generation (#124102).
            row = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND active = 1 AND role = ? AND message_uid = ? "
                "ORDER BY id DESC LIMIT 1",
                (session_id, message.get("role"), uid)).fetchone()
            if row is not None:
                by_uid_row[uid] = row
                stored[int(row["id"])] = row
        mismatches: list[int] = []
        for message, row_id in exact:
            row = stored.get(row_id) if row_id is not None else by_uid_row.get(message_uid_or_none(message) or "")
            if row is None or not int(row["active"] or 0):
                continue  # the activity staleness check owns absent/inactive rows
            snapshot = message.get(DB_ROW_SNAPSHOT)
            if isinstance(snapshot, str):
                if transcript_row_snapshot(row) != snapshot:
                    mismatches.append(int(row["id"]))
                continue
            held_identity = self._row_identity(
                message.get("role", "unknown"), message.get("content"), message.get("tool_call_id"),
                _parse_tool_calls(message.get("tool_calls")))
            stored_identity = self._row_identity(
                row["role"], self._decode_content(row["content"]), row["tool_call_id"],
                _parse_tool_calls(row["tool_calls"]))
            if held_identity != stored_identity:
                mismatches.append(int(row["id"]))
        return mismatches

    def rewrite_pruned_rows(self, session_id: str, changes: list[tuple[dict[str, Any], dict[str, Any]]], *,
        model_config_patch: Optional[dict[str, Any]] = None, turn_lease_holder: Optional[str] = None,
        turn_lease_ttl_seconds: float = 300.0) -> int:
        """Proactive-prune commit that rewrites only the changed rows, in place (#124102).

        ``archive_and_compact`` is a generation writer: it archives every active row and re-inserts the whole
        list, which is right for summary compaction (the row set changes) but makes each prune grow the store
        by the full live transcript although a prune only edits a few rows' content. Each
        ``(original, replacement)`` in *changes* is a persisted message the prune edited. Its row is named by
        ``original["_row_id"]``, else by its tool call id (tool results) or tool-call ids (assistant turns),
        else by the single active row with the same role and stored content. That row gets one archived twin
        of its pre-prune self (``active=0, compacted=1``: still searchable, like the originals
        ``archive_and_compact`` keeps), then takes the replacement's changed columns under its own id.
        Unchanged rows, model order, row ids and concurrently appended rows are untouched; the FTS and
        display-identity triggers keep both indexes consistent; the counters are unchanged because no row
        enters or leaves the live set.

        Raises :class:`PruneRowUnresolvedError` before writing anything when a row cannot be named exactly (a
        merged dict, zero or several matches, an inactive row or a user row), so the caller can commit
        through ``archive_and_compact``. A named row whose stored payload moved on raises the
        :class:`PruneRowStaleError` subclass instead: that is another writer's generation, not ambiguity,
        and the full writer must not be handed the stale transcript. The staleness check is the repo's
        row-version CAS (``DB_ROW_SNAPSHOT`` / ``transcript_row_snapshot``, see ``agent/transcript_repair``)
        when the held dict carries the digest — it covers every owned column including assistant
        ``tool_calls`` arguments — and the representation-safe loader-lens identity compare otherwise
        (raw storage bytes are never compared against a loaded/normalized view; #124102).
        ``turn_lease_holder`` (a writer inside a turn) refuses with :class:`SessionTurnLeaseLostError`, before
        writing anything, when another process owns the session's turn lease, as ``append_message`` does.
        Returns the rows rewritten. Each rewritten dict is stamped with its row's new digest, so a second
        prune of the same transcript compares against what this commit wrote, not what it replaced.
        """
        from agent.conversation_compression_archive import ABSORBED_ROW_IDS
        from agent.message_metadata import DB_ROW_SNAPSHOT
        from agent.transcript_repair import transcript_row_snapshot
        from hermes_state_errors import PruneRowStaleError, PruneRowUnresolvedError

        def _call_ids(tool_calls: Any) -> list[Any]:
            return [call.get("id") if isinstance(call, dict) else None for call in (_parse_tool_calls(tool_calls) or [])]

        def _resolve(conn, original: dict[str, Any]) -> Any:
            role = original.get("role")
            if role not in ("assistant", "tool") or original.get(ABSORBED_ROW_IDS):
                raise PruneRowUnresolvedError(f"a {role} message is not rewritten in place")
            row_id = original.get("_row_id")
            if isinstance(row_id, int) and not isinstance(row_id, bool) and row_id > 0:
                ids = [row_id]
            elif role == "tool" and original.get("tool_call_id"):
                ids = [int(row["id"]) for row in conn.execute(
                    "SELECT id FROM messages WHERE session_id = ? AND active = 1 AND role = 'tool' AND tool_call_id = ?",
                    (session_id, original["tool_call_id"])).fetchall()]
            elif role == "assistant" and _call_ids(original.get("tool_calls")):
                wanted = _call_ids(original.get("tool_calls"))
                ids = [int(row["id"]) for row in conn.execute(
                    "SELECT id, tool_calls FROM messages WHERE session_id = ? AND active = 1 AND role = 'assistant' "
                    "AND tool_calls IS NOT NULL", (session_id,)).fetchall() if _call_ids(row["tool_calls"]) == wanted]
            else:
                ids = self._matching_active_ids(conn, session_id, original)
            if len(ids) != 1:
                raise PruneRowUnresolvedError(f"{len(ids)} live rows match a pruned {role} message")
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ? AND active = 1", (ids[0], session_id)).fetchone()
            if row is None or row["role"] != role:
                raise PruneRowStaleError(f"row {ids[0]} no longer holds the pruned {role} message")
            snapshot = original.get(DB_ROW_SNAPSHOT)
            if isinstance(snapshot, str):
                # The stored-row CAS digest: covers every owned column (tool_calls arguments included), and
                # is computed from the read-back row, so storage affinity can never fake a mismatch.
                if transcript_row_snapshot(row) != snapshot:
                    raise PruneRowStaleError(f"row {ids[0]} no longer holds the pruned {role} message")
            elif (self._row_identity(role, original.get("content"), original.get("tool_call_id"),
                                     _parse_tool_calls(original.get("tool_calls")))
                    != self._row_identity(row["role"], self._decode_content(row["content"]),
                                           row["tool_call_id"], _parse_tool_calls(row["tool_calls"]))):
                # No digest (a resumed dict): compare through the loader lens instead, the same
                # representation both sides of _stamp_kept_live_prefix use.
                raise PruneRowStaleError(f"row {ids[0]} no longer holds the pruned {role} message")
            return row

        def _do(conn):
            if turn_lease_holder:
                # A writer inside a turn commits only while it still owns the turn, like its appends.
                self._check_transcript_write_guards(conn, session_id, None, turn_lease_holder=turn_lease_holder,
                                                    turn_lease_ttl_seconds=turn_lease_ttl_seconds)
            patch = model_config_patch is not None
            patched_model_config = self._merge_model_config_json(
                conn, session_id, model_config_patch, on_missing="raise") if patch else None
            updates: list[tuple[int, list[str], list[Any], Any, Any]] = []
            for original, replacement in changes:
                row = _resolve(conn, original)
                role, timestamp = row["role"], row["timestamp"]
                if replacement.get("role") != role:
                    raise PruneRowUnresolvedError("a prune replacement changed the message role")
                before = self._message_row_params(session_id, role, original, _parse_tool_calls(
                    original.get("tool_calls")), timestamp, keep_reasoning=role == "assistant")
                after = self._message_row_params(session_id, role, replacement, _parse_tool_calls(
                    replacement.get("tool_calls")), timestamp, keep_reasoning=role == "assistant")
                changed = [column for column, old, new in zip(_MESSAGE_ROW_COLUMNS, before, after)
                           if old != new and column not in _PRUNE_FIXED_COLUMNS]
                if changed:
                    updates.append((int(row["id"]), changed, [
                        value for column, value in zip(_MESSAGE_ROW_COLUMNS, after) if column in changed],
                        row["display_identity"],
                        int(row["display_order"]) if row["display_order"] is not None else int(row["id"]),
                        replacement))
            twin_columns = ", ".join(c for c in self._message_column_names(conn)
                                     if c not in ("id", "active", "compacted", "display_order"))
            for row_id, columns, values, identity, order, replacement in updates:
                twin_id = conn.execute(f"INSERT INTO messages ({twin_columns}, active, compacted) "
                             f"SELECT {twin_columns}, 0, 1 FROM messages WHERE id = ?", (row_id,)).lastrowid
                # The content UPDATE fires the identity trigger, which nulls the slot of the live row and of
                # every visible row sharing its old identity: the twin and any earlier generation's copies.
                peers = conn.execute(
                    "SELECT id, display_order FROM messages WHERE session_id = ? AND id <> ? "
                    "AND display_identity = ? AND (active = 1 OR compacted = 1)",
                    (session_id, row_id, identity)).fetchall() if identity is not None else [(twin_id, order)]
                conn.execute(f"UPDATE messages SET {', '.join(f'{column} = ?' for column in columns)} WHERE id = ?",
                             [*values, row_id])
                # The superseded bodies keep their identity and slot. The live row takes the identity of what it
                # now stores, so a later compaction's copy of it joins the same slot, and keeps the slot.
                # The page shows one row per display_order, preferring the live one: the stub, in place.
                conn.executemany("UPDATE messages SET display_identity = ?, display_order = ? WHERE id = ?",
                                 [(identity, order if peer_order is None else peer_order, peer_id)
                                  for peer_id, peer_order in peers])
                live = conn.execute(
                    "SELECT role, content, timestamp, tool_call_id, tool_calls, tool_name, display_kind, "
                    "display_metadata FROM messages WHERE id = ?", (row_id,)).fetchone()
                conn.execute("UPDATE messages SET display_identity = ?, display_order = ? WHERE id = ?",
                             (self._display_identity(self._display_dedupe_key(live)), order, row_id))
                # Stamp the replacement with the stored row's new CAS digest (read back, not the bind values):
                # a second prune of this transcript must compare against what this commit wrote, not what it
                # replaced, or it would false-stale against its own output (#124102).
                final_row = conn.execute(
                    "SELECT * FROM messages WHERE id = ?", (row_id,)).fetchone()
                replacement[DB_ROW_SNAPSHOT] = transcript_row_snapshot(final_row)
            if patch:
                conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?", (patched_model_config, session_id))
            return len(updates)
        self._ensure_display_order(session_id)  # every row holds its stored slot before one is carried over
        # _execute_transcript_write restores each dict's row state (id/digest/timestamp) on rollback, so a
        # failed or retried commit never leaves the replacements stamped with an uncommitted digest.
        return self._execute_transcript_write(
            _do, [replacement for _, replacement in changes])
