"""Held-row coverage resolution for in-place compaction: which durable rows the compressor actually held.

Names carried-forward dicts, id-less held dicts and the rows an alternation repair merged, dropped or
folded, so ``archive_and_compact`` archives exactly those (or falls back to the watermark when a durable
held row cannot be named). Mixin bound via the MRO next to SessionMessagesMixin, which owns the writes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_cli.timefmt import coerce_epoch
from hermes_state_common import _placeholders
from hermes_state_messages import _parse_tool_calls, _tool_calls_len


class SessionCoverageMixin:
    """Coverage helpers for SessionDB; relies on SessionMessagesMixin's codec/identity helpers."""

    def _tail_rows_after_watermark(self, conn, sql: str, params) -> Tuple[List[int], int]:
        """``(ids, tool_call_count)`` of the concurrent-tail rows selected by *sql* (``SELECT id, tool_calls``)."""
        rows = conn.execute(sql, params).fetchall()
        return [int(r["id"]) for r in rows], sum(_tool_calls_len(r["tool_calls"]) for r in rows)

    def _resolve_carried_row_ids(
        self, conn, session_id: str, carried_messages: List[Dict[str, Any]],
    ) -> List[int]:
        """Resolve byte-identical carried-forward live dicts to their ACTIVE durable originals.

        _row_id is authoritative when the message carries it and the stored identity still matches.
        Resume surfaces that intentionally omit row ids fall back to a UNIQUE
        (role/content/tool identity, timestamp) match. Ambiguous or timestamp-less fallbacks are left
        as compacted history rather than risking a false rewind classification.
        """
        if not carried_messages:
            return []
        carried: List[Tuple[Tuple[Any, ...], Any, Any]] = []
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

        def _index(ids: Optional[List[int]]):
            by_id: Dict[int, Tuple[Any, ...]] = {}
            by_key: Dict[Tuple[Any, ...], List[int]] = {}
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

        resolved: List[int] = []
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

    def _matching_active_ids(self, conn, session_id: str, message: Dict[str, Any]) -> List[int]:
        """Active row ids whose stored role and content equal *message*. Empty when it was never persisted."""
        content = message.get("content")
        if not isinstance(content, str):
            return []
        stored = self._encode_content(self._loaded_view_content(message.get("role", "unknown"), content))
        return [int(row["id"]) for row in conn.execute(
            "SELECT id FROM messages WHERE session_id = ? AND active = 1 AND role = ? AND content = ?",
            (session_id, message.get("role"), stored)).fetchall()]

    def _matching_retired_ids(
        self, conn, session_id: str, retired: Dict[str, Any], watermark: Optional[int] = None,
    ) -> List[int]:
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
        self, conn, session_id: str, message: Dict[str, Any], watermark: Optional[int] = None,
    ) -> Optional[List[int]]:
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
        runs: List[List[int]] = []
        for start in range(len(rows)):
            merged = ""
            for end in range(start, len(rows)):
                row_id, role, part = rows[end]
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
        self, conn, session_id: str, covered_ids: Optional[List[int]],
        unresolved_held: Optional[List[Dict[str, Any]]], watermark: Optional[int] = None,
    ) -> Optional[Tuple[List[int], Set[int]]]:
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
            MERGED_DURABLE_ROWS, OWN_ROW, RETIRED_DURABLE_ROWS, RETIRED_ROW)

        proved = [int(row_id) for row_id in covered_ids if isinstance(row_id, int) and row_id > 0]
        merged_away: Set[int] = set()
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
            merged_away.update(set(run[1:]) - set(message.get("_absorbed_row_ids") or ()))
        return list(dict.fromkeys(proved)), merged_away

    @staticmethod
    def _uncounted_merged_rows(tail: List[Dict[str, Any]]) -> int:
        """Durable rows behind the carried *tail* beyond one per dict, for the positional rewind.

        The watermark path cannot name a merged dict's run, so it widens by the stamp, less the
        rows the dict lists in ``_absorbed_row_ids``: those are already counted in ``tail_count``.
        Rows the repair dropped behind a dict, or folded into an assistant turn, on a reload
        without ids are in neither, so their count is added. A dict that holds a ``_row_id`` was
        written back as a single row by an earlier compaction.
        """
        from agent.conversation_compression_archive import MERGED_DURABLE_ROWS, UNNAMED_DURABLE_ROWS

        def behind(message: Dict[str, Any]) -> int:
            merged, unnamed = message.get(MERGED_DURABLE_ROWS), message.get(UNNAMED_DURABLE_ROWS)
            unlisted = merged - 1 - len(message.get("_absorbed_row_ids") or ()) if type(merged) is int else 0
            return max(0, unlisted) + (unnamed if type(unnamed) is int else 0)

        return sum(
            behind(message) for message in tail
            if isinstance(message, dict) and not isinstance(message.get("_row_id"), int))

    @staticmethod
    def _tail_originals(covered_active: List[int], tail_count: int, merged_away: Set[int]) -> List[int]:
        """Newest rows behind *tail_count* carried dicts; a dict that merged rows stands for each of them."""
        width = int(tail_count)
        while True:
            window = covered_active[-width:]
            need = int(tail_count) + sum(1 for row_id in window if row_id in merged_away)
            if need <= width or width >= len(covered_active):
                return window
            width = need
