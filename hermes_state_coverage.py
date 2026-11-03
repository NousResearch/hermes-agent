"""Held-row coverage resolution for in-place compaction: which durable rows the compressor actually held.

Names carried-forward dicts, id-less held dicts and the rows an alternation repair merged, dropped or
folded, so ``archive_and_compact`` archives exactly those (or falls back to the watermark when a durable
held row cannot be named). Mixin bound via the MRO next to SessionMessagesMixin, which owns the writes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.message_metadata import DB_ROW_SNAPSHOT, MESSAGE_UID, message_uid_or_none
from hermes_cli.timefmt import coerce_epoch
from hermes_state_common import _id_chunks, _placeholders
from hermes_state_messages import _parse_tool_calls


def _text_of(content: Any) -> str:
    """The text a message carries: a string, or its text parts joined; image parts carry none."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "") for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"})
    return ""


def _holds_as_block(inner: str, outer: str) -> bool:
    """True when *inner* sits in *outer* as whole lines: the compressor adds only whole blocks around a carried
    row's text (header, markers, summary), so a hit that starts or ends mid-line is the row losing text instead.
    Blanks beside the hit count as line edges: the load lens strips the stored text, never the copy inside."""
    start = outer.find(inner)
    while start != -1:
        end = start + len(inner)
        if not outer[:start].rpartition("\n")[2].strip() and not outer[end:].partition("\n")[0].strip():
            return True
        start = outer.find(inner, start + 1)
    return False


def _images_replaced(stored: Any, held: Any) -> bool:
    """True when *held* is *stored* with at least one image part swapped for a text part, all else equal."""
    from agent.context_compressor import _is_image_part

    if not (isinstance(stored, list) and isinstance(held, list)) or len(stored) != len(held):
        return False
    swapped = False
    for was, now in zip(stored, held):
        if was == now:
            continue
        if not (_is_image_part(was) and isinstance(now, dict) and now.get("type") == "text"):
            return False
        swapped = True
    return swapped


class SessionCoverageMixin:
    """Coverage helpers for SessionDB; relies on SessionMessagesMixin's codec/identity helpers."""

    def _carried_tail_ids(
        self, conn, tail_ids: list[int], compacted_messages: list[dict[str, Any]],
    ) -> tuple[dict[int, int], set[int]]:
        """Rows of *tail_ids* that *compacted_messages* already carry -> ``({row id: carrying dict index}, weak)``.

        A row above the watermark is normally a concurrent append the summary never saw, so the commit
        clones it after the compacted set. A row the compressor ALSO held (flushed after the watermark
        was read, or left unnamed so coverage fell back to the watermark) is already in the compacted
        set as a carried copy; cloning it too stores the message twice, with a fresh id and timestamp
        each (#137197). A claim must prove the occurrence AND the version, so the next request sees exactly
        what it would have seen with the clone minus the second copy: only a dict that names the row
        (``_row_id``, else ``message_uid``) may claim it, never matching text, which another surface can
        append independently. The first pass claims a named dict holding the row's stored content and its
        replayed sidecars (``_replays_as``). The second lets a still-unclaimed named dict with other content
        claim the row only when the compressor's own rewrite of the row's CURRENT version explains the
        difference (see ``_rewrite_explains``), exactly one tail row carries that uid and no active row
        outside *tail_ids* does (there the dict's original is that row, so the tail row is another version).
        Anything else is cloned: a duplicate, never a stale or missing model input. Each dict claims at most
        one row and each row is claimed once.

        ``weak`` holds the rows claimed by a rewritten carry: it is not the stored text, so the caller archives
        those as compacted history, never rewinds them out of display.
        """
        if not tail_ids or not compacted_messages:
            return {}, set()
        by_row, by_uid = self._index_held_dicts(compacted_messages)
        rows = conn.execute(
            # Every column: the claims compare replayed sidecars and the row's version digest (DB_ROW_SNAPSHOT).
            f"SELECT * FROM messages WHERE id IN ({_placeholders(tail_ids)}) ORDER BY id", tail_ids,
        ).fetchall()
        carried: dict[int, int] = {}
        for row in rows:
            index = self._claim_stored_copy(row, compacted_messages, by_row, by_uid, set(carried.values()))
            if index is not None:
                carried[int(row["id"])] = index
        named_uids = {*by_uid, *(named[2] for named in by_row.values() if named[2] is not None)}
        held_elsewhere = self._uids_active_outside(conn, rows, tail_ids, named_uids)
        tail_uids = Counter(row[MESSAGE_UID] for row in rows if row[MESSAGE_UID])
        weak: set[int] = set()
        for row in rows:
            row_id, row_uid = int(row["id"]), row[MESSAGE_UID] or None
            if row_id in carried or row_uid is None or tail_uids[row_uid] != 1 or row_uid in held_elsewhere:
                continue
            index = self._rewritten_carry(row, row_uid, compacted_messages, by_row, by_uid, set(carried.values()))
            if index is not None:
                carried[row_id] = index
                weak.add(row_id)
        return carried, weak

    def _index_held_dicts(self, compacted_messages: list[dict[str, Any]]) -> tuple[dict, dict]:
        """Held dicts by the name they carry: ``_row_id`` -> (index, identity, uid); uid -> [(index, identity)].
        A dict naming neither is never a claim: equal text is not the same occurrence."""
        by_row: dict[int, tuple[int, tuple[Any, ...], Optional[str]]] = {}
        by_uid: dict[str, list[tuple[int, tuple[Any, ...]]]] = {}
        for index, message in enumerate(compacted_messages):
            if not isinstance(message, dict):
                continue
            identity = self._row_identity(
                message.get("role", "unknown"), message.get("content"), message.get("tool_call_id"),
                _parse_tool_calls(message.get("tool_calls")))
            row_id = message.get("_row_id")
            uid = message_uid_or_none(message)
            if isinstance(row_id, int) and not isinstance(row_id, bool) and row_id > 0:
                # Authoritative only while the stored identity still matches (a stale stamp names another row).
                by_row.setdefault(row_id, (index, identity, uid))
            elif uid is not None:
                by_uid.setdefault(uid, []).append((index, identity))
        return by_row, by_uid

    def _claim_stored_copy(self, row, compacted_messages, by_row, by_uid, claimed) -> Optional[int]:
        """First pass: the unclaimed dict naming *row* (``_row_id``, else uid) that replays exactly as it does."""
        identity = self._row_identity(
            row["role"], self._decode_content(row["content"]), row["tool_call_id"], _parse_tool_calls(row["tool_calls"]))
        named = by_row.get(int(row["id"]))
        candidates = [named[0]] if named is not None and named[1] == identity else []
        row_uid = row[MESSAGE_UID] or None
        candidates += [index for index, held in by_uid.get(row_uid, ()) if held == identity] if row_uid else []
        return next((index for index in candidates
                     if index not in claimed and self._replays_as(row, compacted_messages[index], True)), None)

    def _rewritten_carry(self, row, row_uid, compacted_messages, by_row, by_uid, claimed) -> Optional[int]:
        """Second pass: the one unclaimed dict naming *row* (``_row_id`` with the row's uid, else its uid alone)
        whose different content the compressor's rewrite of that row explains; None means clone."""
        named = by_row.get(int(row["id"]))
        if named is not None and named[2] == row_uid:
            candidates = [named[0]]
        else:
            candidates = [index for index, _identity in by_uid.get(row_uid, ())]
        candidates = [index for index in candidates if index not in claimed]
        if len(candidates) != 1:
            return None
        if self._rewrite_explains(row, compacted_messages[candidates[0]]):
            return candidates[0]
        return None

    def _replays_as(self, row, held: dict[str, Any], same_content: bool) -> bool:
        """Whether *held* replays the sidecars the next request would read from *row* (``api_content`` and the
        reasoning carriers in ``REPLAY_SIDECARS``), compared through the writer's own serializer.

        A sidecar the copy carries must equal the row's: a different one is an older version (a later
        ``set_message_api_content``, a re-stamped reasoning item). A copy WITHOUT the row's ``api_content``
        stands for it only when its own content is those bytes (the flush stores sanitize-changed text as the
        sidecar), or when the compressor rewrote the content, which drops the stale sidecar by design."""
        from agent.transcript_repair import REPLAY_SIDECARS

        serialized = self._serialized_message_row(row["session_id"], held, float(row["timestamp"] or 0))
        content = self._decode_content(row["content"])
        row_api, held_api = (None if api == text else api for api, text in (
            (row["api_content"], content), (serialized["api_content"], held.get("content"))))
        if any(serialized[column] is not None and serialized[column] != row[column] for column in REPLAY_SIDECARS):
            return False
        if held_api is not None or row_api is None:
            return held_api == row_api
        return not same_content or held.get("content") == row_api

    @staticmethod
    def _copy_of_current_row(row, held: dict[str, Any]) -> bool:
        """Whether *held* was copied from *row*'s current version, so extra text in it is the compressor's.

        Containment alone cannot tell "the compressor added text" from "the row lost text after the copy"
        (an in-place edit that shortens it). Only the stored-row digest a flushed dict carries settles it: a
        mismatch means the row was rewritten after the copy. A copy without one (a bare uid copy, or a
        row-addressed load, which omits the digest by design) may predate any number of rewrites, so the row
        is cloned: a duplicate, never an older instruction in place of the current one."""
        # Runtime import avoids the transcript_repair -> hermes_state_messages module cycle.
        from agent.transcript_repair import transcript_row_snapshot

        snapshot = held.get(DB_ROW_SNAPSHOT)
        return isinstance(snapshot, str) and snapshot == transcript_row_snapshot(row)

    def _rewrite_explains(self, row, held: dict[str, Any]) -> bool:
        """True when *held* differs from stored *row* only the way the compressor rewrites a carry it keeps:
        image parts replaced by text placeholders, or, for a copy of the row's current version, the row's
        text kept verbatim as whole lines inside a longer text (a summary merged in, prior-context header and
        markers around it); and its sidecars still replay the row's. The summary flag alone is not enough: a
        flagged copy taken before an in-place rewrite of the row would put the older text back."""
        role = row["role"]
        if (held.get("role"), held.get("tool_call_id") or None) != (role, row["tool_call_id"] or None):
            return False
        if (_parse_tool_calls(held.get("tool_calls")) or None) != (_parse_tool_calls(row["tool_calls"]) or None):
            return False
        stored = self._loaded_view_content(role, self._decode_content(row["content"]))
        content = self._loaded_view_content(role, held.get("content"))
        if not self._replays_as(row, held, stored == content):
            return False
        if _images_replaced(stored, content):
            return True
        if not self._copy_of_current_row(row, held):
            return False
        stored_text = _text_of(stored)
        return bool(stored_text.strip()) and _holds_as_block(stored_text, _text_of(content))

    @staticmethod
    def _uids_active_outside(conn, rows, tail_ids: list[int], named_uids) -> set[str]:
        """uids of tail *rows* a held dict names that an ACTIVE row outside *tail_ids* (a covered row, below the
        watermark) also carries: there the dict's own original is that row, so the tail row is another version."""
        wanted = sorted({row[MESSAGE_UID] for row in rows if row[MESSAGE_UID] in named_uids})
        if not wanted:
            return set()
        tail, session_id = set(tail_ids), rows[0]["session_id"]
        return {
            found[MESSAGE_UID]
            for chunk in _id_chunks(wanted)
            for found in conn.execute(
                f"SELECT id, {MESSAGE_UID} FROM messages WHERE session_id = ? AND active = 1 "
                f"AND {MESSAGE_UID} IN ({_placeholders(chunk)})", (session_id, *chunk)).fetchall()
            if int(found["id"]) not in tail
        }

    @staticmethod
    def _carried_tail_slots(carried: dict[int, int], compacted_count: int, tail_count: int) -> int:
        """How many of the carried tail's *tail_count* dicts name a row in *carried*: those originals sit
        above the watermark, so the positional rewind below it must take that many fewer rows."""
        if tail_count <= 0:
            return 0
        return sum(1 for index in carried.values() if index >= compacted_count - int(tail_count))

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
