"""Held-row coverage resolution for in-place compaction: which durable rows the compressor actually held.

Names carried-forward dicts, id-less held dicts and the rows an alternation repair merged, dropped or
folded, so ``archive_and_compact`` archives exactly those (or falls back to the watermark when a durable
held row cannot be named). Mixin bound via the MRO next to SessionMessagesMixin, which owns the writes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from agent.context_compressor import MODEL_ONLY_DISPLAY_METADATA_KEY
from hermes_cli.timefmt import coerce_epoch
from hermes_state_common import _id_chunks, _placeholders
from hermes_state_messages import _DISPLAY_ACTIVE_CLAUSE, _parse_tool_calls


class SessionCoverageMixin:
    """Coverage helpers for SessionDB; relies on SessionMessagesMixin's codec/identity helpers."""

    def _resolve_carried_row_ids(
        self, conn, session_id: str, carried_messages: list[dict[str, Any]],
    ) -> list[int]:
        """Resolve byte-identical carried-forward live dicts to their ACTIVE durable originals.

        _row_id is authoritative when the message carries it and the stored identity still matches.
        Resume surfaces that intentionally omit row ids fall back to a UNIQUE
        (role/content/tool identity, timestamp) match. Ambiguous or timestamp-less fallbacks are left
        as compacted history rather than risking a false rewind classification.
        """
        if not carried_messages:
            return []
        carried: list[tuple[tuple[Any, ...], Any, Any]] = []
        for message in carried_messages:
            if not isinstance(message, dict):
                continue
            identity = self._row_identity(
                message.get("role", "unknown"), message.get("content"), message.get("tool_call_id"),
                _parse_tool_calls(message.get("tool_calls")))
            row_id = message.get("_row_id")
            if not (isinstance(row_id, int) and not isinstance(row_id, bool) and row_id > 0):
                row_id = None
            carried.append((identity, row_id, message.get("timestamp")))

        def _index(ids: Optional[list[int]]):
            by_id: dict[int, tuple[Any, ...]] = {}
            by_key: dict[tuple[Any, ...], list[int]] = {}
            narrow = f" AND id IN ({_placeholders(ids)})" if ids else ""
            for row in conn.execute(
                "SELECT id, role, content, tool_call_id, tool_calls, timestamp FROM messages "
                f"WHERE session_id = ? AND active = 1{narrow} ORDER BY id",
                (session_id, *(ids or ())),
            ).fetchall():
                rid = int(row["id"])
                by_id[rid] = self._row_identity(
                    row["role"], self._decode_content(row["content"]), row["tool_call_id"],
                    _parse_tool_calls(row["tool_calls"]))
                ts = coerce_epoch(row["timestamp"], field="message timestamp")
                if ts is not None:
                    by_key.setdefault((*by_id[rid], ts), []).append(rid)
            return by_id, by_key

        # The common micro pass carries dicts that all hold a matching _row_id, so the identity
        # check only needs those rows; a full active-row scan is reserved for the fallbacks.
        row_ids = [row_id for _, row_id, _ in carried if row_id is not None]
        by_id, by_key = _index(row_ids if len(row_ids) == len(carried) else None)
        if len(row_ids) == len(carried) and any(by_id.get(rid) != ident for ident, rid, _ in carried):
            by_id, by_key = _index(None)

        resolved: list[int] = []
        for identity, row_id, raw_timestamp in carried:
            if row_id is not None and by_id.get(row_id) == identity:
                resolved.append(row_id)
                continue
            timestamp = coerce_epoch(raw_timestamp, field="message timestamp")
            if timestamp is None:
                continue
            matches = by_key.get((*identity, timestamp), [])
            if len(matches) == 1:
                resolved.append(matches[0])
        return list(dict.fromkeys(resolved))

    def _mark_superseded_display_copies(
        self, conn, session_id: str, messages: list[dict[str, Any]],
    ) -> None:
        """Flag carried copies whose durable original stays in display history (#135655).

        A commit re-inserts ``compacted_messages`` as fresh live rows. A copy whose content was
        rewritten since its durable original was loaded (a pruned tool-result digest, a merge
        carrier) no longer shares the original's display identity, so
        :meth:`_dedupe_display_generations` cannot fold the two and the display read paints the
        logical message twice: the archived original plus its rewrite. The archived original is the
        generation the user should see, so the copy is marked model-only (mirrors
        ``micro_compaction._merge_adjacent_user_turns``): the model still reads it, display
        projections skip it. The rule is uniform, not rewrite-conditional: a copy that kept its
        original's display identity is folded by ``_dedupe_display_generations`` anyway, so marking
        it only makes the archived original the representative of the single painted generation. An
        original that is itself model-only still supersedes the copy - its lineage keeps its own
        display generation (a suppressed merge join keeps the pre-merge rows), and a visible copy
        would resurrect what the suppression removed.

        Call AFTER the rewind/archive step of the commit transaction: it leaves exactly the
        display-visible originals as ``active = 1 OR compacted = 1``. A copy whose original was
        rewound (``active = 0, compacted = 0`` - the carried tail) is the only visible generation
        and stays unmarked, as does a copy with no ``_row_id``.
        """
        candidates = [
            msg for msg in messages
            if isinstance(msg, dict) and type(row_id := msg.get("_row_id")) is int and row_id > 0
            and not (isinstance(msg.get("display_metadata"), dict)
                     and msg["display_metadata"].get(MODEL_ONLY_DISPLAY_METADATA_KEY))]
        if not candidates:
            return
        row_ids = list(dict.fromkeys(msg["_row_id"] for msg in candidates))
        visible: set[int] = set()
        for chunk in _id_chunks(row_ids):
            visible.update(
                int(row["id"]) for row in conn.execute(
                    f"SELECT id FROM messages WHERE session_id = ? AND id IN ({_placeholders(chunk)})"
                    f"{_DISPLAY_ACTIVE_CLAUSE}", (session_id, *chunk)))
        for msg in candidates:
            if msg["_row_id"] not in visible:
                continue
            metadata = msg.get("display_metadata")
            msg["display_metadata"] = {
                **(metadata if isinstance(metadata, dict) else {}),
                MODEL_ONLY_DISPLAY_METADATA_KEY: True}

    def _matching_active_ids(self, conn, session_id: str, message: dict[str, Any]) -> list[int]:
        """Active row ids whose stored role and content equal *message*. Empty when it was never persisted."""
        content = message.get("content")
        if not isinstance(content, str):
            return []
        stored = self._encode_content(self._loaded_view_content(message.get("role", "unknown"), content))
        return [int(row["id"]) for row in conn.execute(
            "SELECT id FROM messages WHERE session_id = ? AND active = 1 AND role = ? AND content = ?",
            (session_id, message.get("role"), stored)).fetchall()]

    def _matching_retired_ids(
        self, conn, session_id: str, retired: dict[str, Any], watermark: Optional[int] = None,
    ) -> list[int]:
        """Active rows equal to a row the alternation repair retired without an id.

        Matched on role, loaded-view content (an empty assistant stores ``None`` or ``""``), tool_call_id
        and the ids of its tool calls. Only rows at or below *watermark*: the repair ran on a load taken
        before the snapshot, so a later equal row is another surface's append.
        """
        role = str(retired.get("role") or "")
        bound = watermark is not None
        rows = conn.execute(
            "SELECT id, content, tool_call_id, tool_calls FROM messages WHERE session_id = ? AND active = 1 "
            f"AND role = ?{' AND id <= ?' if bound else ''} ORDER BY id",
            (session_id, role, *((int(watermark),) if bound else ()))).fetchall()
        want_calls = list(retired.get("tool_call_ids") or ())
        matches = []
        for row in rows:
            content = self._loaded_view_content(role, self._decode_content(row["content"]))
            if (content or None) != (retired.get("content") or None):
                continue
            if (row["tool_call_id"] or None) != (retired.get("tool_call_id") or None):
                continue
            calls = _parse_tool_calls(row["tool_calls"]) or []
            ids = [call.get("id") for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
            if ids != want_calls:
                continue
            matches.append(int(row["id"]))
        return matches

    def _merged_user_run(
        self, conn, session_id: str, message: dict[str, Any], watermark: Optional[int] = None,
    ) -> Optional[list[int]]:
        """Active rows an alternation repair merged into *message*: ``[]`` for none, None when ambiguous.

        A reload without row ids turns a durable ``user;user`` pair (a prompt that never got its reply)
        into one dict equal to neither row. Read as an unpersisted turn, its rows would be re-sequenced
        after the compacted set like concurrent appends, behind the turn that is running.
        Only a dict the repair stamped: a prompt that was never persisted can carry the same text as
        rows another surface appended, and those were never held.
        Only rows at or below *watermark*: the repair ran on a load taken before the snapshot, so a
        run appended later that joins to the same text is another surface's, not the merged one.
        """
        from agent.conversation_compression_archive import MERGED_DURABLE_ROWS, OWN_ROW, RETIRED_DURABLE_ROWS

        content, width = message.get("content"), message.get(MERGED_DURABLE_ROWS)
        if message.get("role") != "user" or not isinstance(content, str) or type(width) is not int or width < 2:
            return []
        from gateway.message_timestamps import strip_leading_message_timestamps

        content, _ = strip_leading_message_timestamps(content)
        bound = watermark is not None
        rows = [
            (int(row["id"]), row["role"], self._loaded_view_content(row["role"], self._decode_content(row["content"])))
            for row in conn.execute(
                f"SELECT id, role, content FROM messages WHERE session_id = ? AND active = 1"
                f"{' AND id <= ?' if bound else ''} ORDER BY id",
                (session_id, *((int(watermark),) if bound else ()))).fetchall()]
        # Dropping an orphan can make user rows adjacent only in the repaired view.
        # Skip only uniquely proved retired originals, never arbitrary intervening rows.
        retired_ids = set()
        for retired in message.get(RETIRED_DURABLE_ROWS) or ():
            if retired.get(OWN_ROW):
                continue
            matches = self._matching_retired_ids(conn, session_id, retired, watermark)
            if len(matches) != 1:
                return None
            retired_ids.update(matches)
        rows = [row for row in rows if row[0] not in retired_ids]
        runs: list[list[int]] = []
        for start in range(len(rows)):
            merged = ""
            for end in range(start, len(rows)):
                _row_id, role, part = rows[end]
                if role != "user" or not isinstance(part, str):
                    break
                merged = f"{merged}\n\n{part}" if merged and part else (merged or part)
                if not content.startswith(merged):
                    break
                if end - start + 1 == width:
                    if merged == content:
                        runs.append([rows[i][0] for i in range(start, end + 1)])
                    break
        if len(runs) > 1:
            return None
        return runs[0] if runs else []

    def _proved_coverage(
        self, conn, session_id: str, covered_ids: Optional[list[int]],
        unresolved_held: Optional[list[dict[str, Any]]], watermark: Optional[int] = None,
    ) -> Optional[tuple[list[int], set[int]]]:
        """``(ids safe to archive as summarized, ids merged into another held dict)``, or None when
        a durable held row cannot be named.

        An unresolved dict that still carries the persist marker was loaded from the DB.
        Failing to name it means the watermark path, which archives the rows the compressor
        saw, including ones whose ids were stripped. A marker-less miss is an unpersisted
        turn: it names nothing, and it is not a reason to abandon the ids we do have.
        Several active rows with the same content are ambiguous, so that also abandons,
        and so does a merged dict whose run cannot be named.
        The second set holds only merges the caller could not count: a dict that lists its
        own ``_absorbed_row_ids`` is already counted in ``tail_count``.
        """
        if covered_ids is None:
            return None
        from agent.context_compressor import _DB_PERSISTED_MARKER
        from agent.conversation_compression_archive import (
            ABSORBED_ROW_IDS, MERGED_DURABLE_ROWS, OWN_ROW, RETIRED_DURABLE_ROWS, RETIRED_ROW)

        proved = [int(row_id) for row_id in covered_ids if isinstance(row_id, int) and row_id > 0]
        merged_away: set[int] = set()
        for message in unresolved_held or ():
            if not isinstance(message, dict):
                continue
            if message.get(RETIRED_ROW):
                # A row the repair dropped behind another held dict: that dict stands for it in the tail.
                # The dict's own row, recorded before a fold rewrote its text, is its own tail slot.
                matches = self._matching_retired_ids(conn, session_id, message, watermark)
                if len(matches) != 1:
                    return None
                proved.extend(matches)
                if not message.get(OWN_ROW):
                    merged_away.update(matches)
                continue
            # A folded assistant's own row is named by its OWN_ROW record above. Its combined text never
            # existed in storage, so matching it by content could only claim another surface's row.
            if any(isinstance(row, dict) and row.get(OWN_ROW) for row in message.get(RETIRED_DURABLE_ROWS) or ()):
                continue
            # The stamp is provenance and text is not: a surface may have re-rendered the dict since the
            # repair (gateway timestamps), and a later row can equal the merged text. So the run is
            # resolved first, and a stamped dict that names none is still durable, never an unpersisted turn.
            run = self._merged_user_run(conn, session_id, message, watermark)
            if message.get(MERGED_DURABLE_ROWS) and not run:
                return None
            matches = [] if run else self._matching_active_ids(conn, session_id, message)
            if len(matches) > 1 or (
                    message.get(_DB_PERSISTED_MARKER) and len(matches) != 1 and not run):
                return None
            proved.extend(matches or run)
            merged_away.update(set(run[1:]) - set(message.get(ABSORBED_ROW_IDS) or ()))
        return list(dict.fromkeys(proved)), merged_away

    @staticmethod
    def _uncounted_merged_rows(tail: list[dict[str, Any]]) -> int:
        """Durable rows behind the carried *tail* beyond one per dict, for the positional rewind.

        The watermark path cannot name a merged dict's run, so it widens by the stamp, less the
        rows the dict lists in ``_absorbed_row_ids``: those are already counted in ``tail_count``.
        Rows the repair dropped behind a dict, or folded into an assistant turn, on a reload
        without ids are in neither, so their count is added. A dict that holds a ``_row_id`` was
        written back as a single row by an earlier compaction.
        """
        from agent.conversation_compression_archive import ABSORBED_ROW_IDS, MERGED_DURABLE_ROWS, UNNAMED_DURABLE_ROWS

        def behind(message: dict[str, Any]) -> int:
            merged, unnamed = message.get(MERGED_DURABLE_ROWS), message.get(UNNAMED_DURABLE_ROWS)
            unlisted = merged - 1 - len(message.get(ABSORBED_ROW_IDS) or ()) if type(merged) is int else 0
            return max(0, unlisted) + (unnamed if type(unnamed) is int else 0)

        return sum(
            behind(message) for message in tail
            if isinstance(message, dict) and not isinstance(message.get("_row_id"), int))

    @staticmethod
    def _tail_originals(covered_active: list[int], tail_count: int, merged_away: set[int]) -> list[int]:
        """Newest rows behind *tail_count* carried dicts; a dict that merged rows stands for each of them."""
        width = int(tail_count)
        while True:
            window = covered_active[-width:]
            need = int(tail_count) + sum(1 for row_id in window if row_id in merged_away)
            if need <= width or width >= len(covered_active):
                return window
            width = need
