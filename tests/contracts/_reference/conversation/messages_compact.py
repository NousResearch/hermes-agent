"""In-place compaction of the in-memory message reference: the compaction commit, the held-row staleness compare and the in-place prune rewrite."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import fields, replace
from typing import Any, Mapping, Optional, Sequence

from persistence import HeldRowRewritten, PruneRowStale, PruneRowUnresolved, RowVisibility
from persistence.conversation.messages import (
    CONTENT_JSON_PREFIX,
    CompactionResult,
    HeldCoverage,
    HeldMessage,
    NewMessage,
    PrunedRowChange,
    PruneRewriteResult,
    Watermark,
)
from persistence.core.identity import JsonValue, MessageId, SessionId
from gateway.message_timestamps import strip_leading_message_timestamps

from . import _turn_policy as policy
from .._lease_fence import verify_lease, verify_turn_lease
from .messages_base import MessageBase
from .transcript_store import _payload, _row_snapshot, _StoredRow, _Transcript, clone_rows, insert_rows, mint_row_id, row_stamp


def row_identity(role: str, content: Any, tool_call_id: Optional[str], tool_calls: Sequence[Any]) -> tuple[Any, ...]:
    """The columns a kept-prefix or held-row match compares (``SessionMessagesMixin._row_identity``): role, content through the load lens as stored, tool call id and tool calls."""
    viewed = policy.loaded_view(role, content)
    if isinstance(viewed, (dict, list)):
        viewed = CONTENT_JSON_PREFIX + json.dumps(viewed)
    return (role, viewed, tool_call_id, json.dumps(list(tool_calls)) if tool_calls else None)


def _stored_identity(row: _StoredRow) -> tuple[Any, ...]:
    m = row.message
    return row_identity(m.role, m.content, m.tool_call_id, m.tool_calls)


def _held_identity(held: HeldMessage) -> tuple[Any, ...]:
    return row_identity(held.role, held.content, held.tool_call_id, held.tool_calls)


def _carried_ids(active: Sequence[_StoredRow], carried: Sequence[HeldMessage]) -> list[int]:
    """Carried originals (``SessionCoverageMixin._resolve_carried_row_ids``): a ``row_id`` whose row still matches wins, else a unique identity-plus-timestamp match, else nothing."""
    by_id = {row.id: row for row in active}
    resolved: list[int] = []
    for held in carried:
        identity = _held_identity(held)
        direct = by_id.get(int(held.row_id)) if held.row_id is not None else None
        if direct is not None and _stored_identity(direct) == identity:
            resolved.append(direct.id)
            continue
        stamp = policy.coerce_epoch(held.timestamp) if held.timestamp is not None else None
        if stamp is None:
            continue
        matches = [row.id for row in active if _stored_identity(row) == identity and policy.coerce_epoch(row.message.timestamp) == stamp]
        if len(matches) == 1:
            resolved.append(matches[0])
    return list(dict.fromkeys(resolved))


def _matching_active_ids(active: Sequence[_StoredRow], held: HeldMessage) -> list[int]:
    """Active rows whose stored role and content equal the held text (``SessionCoverageMixin._matching_active_ids``); a non-text held message matches nothing."""
    if not isinstance(held.content, str):
        return []
    stored = policy.loaded_view(held.role, held.content)
    return [row.id for row in active if row.message.role == held.role and row.message.content == stored]


def _retired_matches(active: Sequence[_StoredRow], retired: HeldMessage, floor: Optional[int]) -> list[int]:
    """Active rows equal to a row the alternation repair retired without an id (``SessionCoverageMixin._matching_retired_ids``): role, content through the load lens (empty and absent alike), tool call id and the ids of its tool calls, only at or below ``floor`` when one is given."""
    matches = []
    for row in active:
        m = row.message
        if m.role != retired.role or (floor is not None and row.id > floor):
            continue
        if (policy.loaded_view(m.role, m.content) or None) != (retired.content or None) or (m.tool_call_id or None) != (retired.tool_call_id or None):
            continue
        if [call.get("id") for call in m.tool_calls if isinstance(call, dict)] == list(retired.tool_call_ids):
            matches.append(row.id)
    return matches


def _merged_run(active: Sequence[_StoredRow], held: HeldMessage, floor: Optional[int]) -> Optional[list[int]]:
    """The user rows an alternation repair merged into ``held`` (``SessionCoverageMixin._merged_user_run``): ``[]`` for none, None when ambiguous. The run is the ``merged_durable_rows`` adjacent user rows at or below ``floor`` whose texts joined by blank lines equal the held text without its leading timestamps; retired rows the message records are skipped between them."""
    content, width = held.content, held.merged_durable_rows
    if held.role != "user" or not isinstance(content, str) or width < 2:
        return []
    content, _ = strip_leading_message_timestamps(content)
    retired_ids: set[int] = set()
    for retired in held.retired_durable_rows:
        if retired.own_row:
            continue
        matches = _retired_matches(active, retired, floor)
        if len(matches) != 1:
            return None
        retired_ids.update(matches)
    rows = [(row.id, row.message.role, policy.loaded_view(row.message.role, row.message.content))
            for row in active if (floor is None or row.id <= floor) and row.id not in retired_ids]
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


def _proved_ids(active: Sequence[_StoredRow], coverage: Optional[HeldCoverage], floor: Optional[int]) -> Optional[tuple[list[int], set[int]]]:
    """``(ids a coverage proof archives as summarized, ids merged into another held message)`` (``SessionCoverageMixin._proved_coverage``), or None to fall back to the watermark: a persisted held message that names no single row, any ambiguous one, a retired row that matches no single row and a merged message whose run cannot be named abandon the proof. A message that records its own row is named by that record alone."""
    if coverage is None:
        return None
    proved = [int(i) for i in coverage.covered_ids]
    merged_away: set[int] = set()
    for held in coverage.unresolved:
        if held.retired_row:
            matches = _retired_matches(active, held, floor)
            if len(matches) != 1:
                return None
            proved.extend(matches)
            if not held.own_row:
                merged_away.update(matches)
            continue
        if any(row.own_row for row in held.retired_durable_rows):
            continue
        run = _merged_run(active, held, floor)
        if held.merged_durable_rows and not run:
            return None
        matches = [] if run else _matching_active_ids(active, held)
        if len(matches) > 1 or (held.persisted and len(matches) != 1 and not run):
            return None
        proved.extend(matches or run or ())
        merged_away.update(set((run or [])[1:]) - {int(i) for i in held.absorbed_row_ids})
    return list(dict.fromkeys(proved)), merged_away


def _tail_originals(covered_active: list[int], tail_count: int, merged_away: set[int]) -> list[int]:
    """The newest covered rows behind the ``tail_count`` carried rows (``_tail_originals``): a row merged into another held message widens the window by one."""
    width = tail_count
    while True:
        window = covered_active[-width:]
        need = tail_count + sum(1 for row_id in window if row_id in merged_away)
        if need <= width or width >= len(covered_active):
            return window
        width = need


def held_mismatches(transcript: Optional[_Transcript], held: Sequence[HeldMessage]) -> list[int]:
    """Ids of persisted held messages whose active row no longer holds what they hold (``SessionPruneRewriteMixin.held_row_mismatches``): the ownership version when the message carries one, else the load-lens identity; a message without a row id is named by its uid when it carries a version. Absent and inactive rows are not reported."""
    if transcript is None:
        return []
    by_id = {row.id: row for row in transcript.rows}
    mismatches: list[int] = []
    for message in held:
        if not message.persisted or message.role not in ("user", "assistant", "tool"):
            continue
        if message.row_id is not None:
            row = by_id.get(int(message.row_id))
        elif message.db_row_snapshot is not None and message.message_uid:
            row = next((r for r in reversed(transcript.active()) if r.message.role == message.role and r.message.message_uid == message.message_uid), None)
        else:
            continue
        if row is None or row.visibility is not RowVisibility.ACTIVE:
            continue
        if message.db_row_snapshot is not None:
            stale = _row_snapshot(row.message) != message.db_row_snapshot
        else:
            stale = _stored_identity(row) != _held_identity(message)
        if stale:
            mismatches.append(row.id)
    return mismatches


def _uncounted_merged_rows(tail: Sequence[NewMessage]) -> int:
    """Durable rows behind the carried tail beyond one per message, which the watermark path's tail rewind widens by: a merged run's rows less the first and the ids it lists, plus the rows it stands for unnamed; a message with a ``repair_target`` is one row."""
    return sum(max(0, m.merged_durable_rows - 1 - len(m.absorbed_row_ids)) + m.unnamed_durable_rows
               for m in tail if m.repair_target is None)


# Image part types a compressor may replace by a text placeholder in a carried row.
_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})
# The reasoning carriers the next request replays from a row.
_REPLAY_SIDECARS = ("reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items")


def _text_of(content: Any) -> str:
    """The text a message carries: a string, or its text parts joined; image parts carry none."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content
                         if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"})
    return ""


def _holds_as_block(inner: str, outer: str) -> bool:
    """Whether ``inner`` sits in ``outer`` as whole lines, blanks beside the hit counting as line edges."""
    start = outer.find(inner)
    while start != -1:
        end = start + len(inner)
        if not outer[:start].rpartition("\n")[2].strip() and not outer[end:].partition("\n")[0].strip():
            return True
        start = outer.find(inner, start + 1)
    return False


def _images_replaced(stored: Any, held: Any) -> bool:
    """Whether ``held`` is ``stored`` with at least one image part swapped for a text part, all else equal."""
    if not (isinstance(stored, list) and isinstance(held, list)) or len(stored) != len(held):
        return False
    swapped = False
    for was, now in zip(stored, held):
        if was == now:
            continue
        if not (isinstance(was, dict) and was.get("type") in _IMAGE_PART_TYPES and isinstance(now, dict) and now.get("type") == "text"):
            return False
        swapped = True
    return swapped


def _replays_as(row: _StoredRow, held: NewMessage, same_content: bool) -> bool:
    """Whether ``held`` replays the sidecars the next request reads from ``row`` (``SessionCoverageMixin._replays_as``): a sidecar the copy carries equals the row's, and a copy without the row's ``api_content`` stands for it only when its own content is those bytes or the compressor rewrote the content."""
    stored = row.message
    written = _payload(held, keep_reasoning=held.role == "assistant")
    if any(getattr(written, name) is not None and getattr(written, name) != getattr(stored, name) for name in _REPLAY_SIDECARS):
        return False
    row_api = None if stored.api_content == stored.content else stored.api_content
    held_api = None if written.api_content == held.content else written.api_content
    if held_api is not None or row_api is None:
        return held_api == row_api
    return not same_content or held.content == row_api


def _rewrite_explains(row: _StoredRow, held: NewMessage) -> bool:
    """Whether ``held`` differs from ``row`` only as the compressor rewrites a carry it keeps (``SessionCoverageMixin._rewrite_explains``): image parts replaced by text, or, for a copy carrying the row's current ownership version, the row's text kept as whole lines inside a longer text; its sidecars still replay the row's."""
    m = row.message
    if (held.role, held.tool_call_id or None) != (m.role, m.tool_call_id or None):
        return False
    if (list(held.tool_calls) or None) != (list(m.tool_calls) or None):
        return False
    stored, content = policy.loaded_view(m.role, m.content), policy.loaded_view(m.role, held.content)
    if not _replays_as(row, held, stored == content):
        return False
    if _images_replaced(stored, content):
        return True
    if held.db_row_snapshot is None or held.db_row_snapshot != _row_snapshot(m):
        return False
    stored_text = _text_of(stored)
    return bool(stored_text.strip()) and _holds_as_block(stored_text, _text_of(content))


def _carried_tail(active: Sequence[_StoredRow], tail: Sequence[_StoredRow], replacement: Sequence[NewMessage]) -> tuple[dict[int, int], set[int]]:
    """Rows of ``tail`` the replacement already carries, as ``({row id: replacement index}, weak)`` (``SessionCoverageMixin._carried_tail_ids``).

    Only a replacement naming the row (``repair_target``, else ``message_uid``) claims it. The first pass claims for the row a named replacement with its identity that replays as it. The second lets a still-unclaimed named replacement (``repair_target`` with the row's uid, else that uid alone) claim a row whose uid is unique in the tail and on no active row outside it when the compressor's rewrite explains the difference; those claims are ``weak``. Each replacement claims at most one row.
    """
    if not tail or not replacement:
        return {}, set()
    by_row: dict[int, tuple[int, tuple[Any, ...], Optional[str]]] = {}
    by_uid: dict[str, list[tuple[int, tuple[Any, ...]]]] = {}
    for index, message in enumerate(replacement):
        identity = row_identity(message.role, message.content, message.tool_call_id, message.tool_calls)
        if message.repair_target is not None and int(message.repair_target) > 0:
            by_row.setdefault(int(message.repair_target), (index, identity, message.message_uid or None))
        elif message.message_uid:
            by_uid.setdefault(message.message_uid, []).append((index, identity))
    carried: dict[int, int] = {}
    for row in tail:
        identity, uid = _stored_identity(row), row.message.message_uid or None
        named = by_row.get(row.id)
        candidates = [named[0]] if named is not None and named[1] == identity else []
        candidates += [index for index, held in by_uid.get(uid, ()) if held == identity] if uid else []
        claimed = set(carried.values())
        index = next((i for i in candidates if i not in claimed and _replays_as(row, replacement[i], True)), None)
        if index is not None:
            carried[row.id] = index
    named_uids = {*by_uid, *(named[2] for named in by_row.values() if named[2] is not None)}
    tail_ids = {row.id for row in tail}
    wanted = {row.message.message_uid for row in tail if row.message.message_uid in named_uids}
    held_elsewhere = {row.message.message_uid for row in active if row.id not in tail_ids and row.message.message_uid in wanted}
    tail_uids = Counter(row.message.message_uid for row in tail if row.message.message_uid)
    weak: set[int] = set()
    for row in tail:
        uid = row.message.message_uid or None
        if row.id in carried or uid is None or tail_uids[uid] != 1 or uid in held_elsewhere:
            continue
        named = by_row.get(row.id)
        candidates = [named[0]] if named is not None and named[2] == uid else [index for index, _identity in by_uid.get(uid, ())]
        claimed = set(carried.values())
        candidates = [index for index in candidates if index not in claimed]
        if len(candidates) == 1 and _rewrite_explains(row, replacement[candidates[0]]):
            carried[row.id] = candidates[0]
            weak.add(row.id)
    return carried, weak


def _carried_tail_slots(carried: Mapping[int, int], replacement_count: int, tail_count: int) -> int:
    """How many of the last ``tail_count`` replacement messages carry a tail row: the positional rewind below the tail takes that many fewer rows."""
    if tail_count <= 0:
        return 0
    return sum(1 for index in carried.values() if index >= replacement_count - tail_count)


def _superseded(active: Sequence[_StoredRow], replacement: Sequence[NewMessage], replacement_tail: int, carried: list[int],
                proved: Optional[tuple[list[int], set[int]]], watermark: Optional[Watermark]) -> tuple[set[int], list[_StoredRow]]:
    """The rows a compaction rewinds as superseded copies, and the rows it clones after the replacement (``_archive_named_rows`` with a holding proof, else the watermark split of ``_archive_and_compact_on_conn``). A tail row the replacement already carries is not cloned: it is rewound, or archived when its claim is weak, and the positional rewind of the carried tail's originals shrinks by the carried slots."""
    floor = None if watermark is None else int(watermark.last_active or 0)
    if proved is not None:
        covered = set(proved[0]) | set(carried)
        covered_ids = [row.id for row in active if row.id in covered]
        tail = [row for row in active if row.id not in covered]
    else:
        tail = [] if floor is None else [row for row in active if row.id > floor]
    claimed, weak = _carried_tail(active, tail, replacement)
    below = replacement_tail - _carried_tail_slots(claimed, len(replacement), replacement_tail)
    if proved is not None:
        tail_ids = set(_tail_originals(covered_ids, below, proved[1])) if below > 0 else set()
        originals = [row for row in active if row.id in tail_ids]
    else:
        bounded = [row for row in active if floor is None or row.id <= floor]
        width = below + _uncounted_merged_rows(replacement[-replacement_tail:]) if below > 0 else 0
        originals = bounded[-width:] if width > 0 else []
    rewound = {*carried, *(row.id for row in originals), *(row.id for row in tail if row.id not in weak)}
    return rewound, [row for row in tail if row.id not in claimed]


# Fields a prune never rewrites: the row keeps its role, tool call id and stored time, and the repair's markers name no column.
_PRUNE_FIXED = frozenset({"role", "tool_call_id", "timestamp", "merged_durable_rows", "unnamed_durable_rows", "absorbed_row_ids"})


def _pruned_target(transcript: _Transcript, original: HeldMessage, session_id: SessionId) -> _StoredRow:
    """The one live row a pruned message names (``rewrite_pruned_rows``'s resolver): its row id, else its tool call id (results) or call ids (assistant turns), else the single active row with its stored text; refusals before any write."""
    if original.role not in ("assistant", "tool") or original.absorbed_row_ids:
        raise PruneRowUnresolved(session_id, f"a {original.role} message is not rewritten in place")
    active = transcript.active()

    def call_ids(calls: Sequence[Any]) -> list[Any]:
        return [call.get("id") if isinstance(call, dict) else None for call in calls]

    if original.row_id is not None:
        ids = [int(original.row_id)]
    elif original.role == "tool" and original.tool_call_id:
        ids = [row.id for row in active if row.message.role == "tool" and row.message.tool_call_id == original.tool_call_id]
    elif original.role == "assistant" and call_ids(original.tool_calls):
        ids = [row.id for row in active if row.message.role == "assistant" and row.message.tool_calls and call_ids(row.message.tool_calls) == call_ids(original.tool_calls)]
    else:
        ids = _matching_active_ids(active, original)
    if len(ids) != 1:
        raise PruneRowUnresolved(session_id, f"{len(ids)} live rows match a pruned {original.role} message")
    row = transcript.find(ids[0])
    stale = row is None or row.visibility is not RowVisibility.ACTIVE or row.message.role != original.role
    if not stale and row is not None:
        if original.db_row_snapshot is not None:
            stale = _row_snapshot(row.message) != original.db_row_snapshot
        else:
            stale = _stored_identity(row) != _held_identity(original)
    if stale or row is None:
        raise PruneRowStale(session_id, f"row {ids[0]} no longer holds the pruned {original.role} message")
    return row


class MessageCompactMixin(MessageBase):
    """The compaction commit and the prune rewrite."""

    def compact(self, session_id: SessionId, replacement: Sequence[NewMessage], *, watermark: Optional[Watermark] = None,
                carried_tail_count: int = 0, lock_holder: Optional[str] = None, model_config_patch: Optional[Mapping[str, JsonValue]] = None,
                carried: Sequence[HeldMessage] = (), coverage: Optional[HeldCoverage] = None,
                held_for_cas: Optional[Sequence[HeldMessage]] = None) -> CompactionResult:
        scope = self._backend.scope
        verify_lease(self._backend, scope, session_id, lock_holder)
        if model_config_patch is not None and str(session_id) not in self._sessions(scope):
            raise ValueError(f"Session not found: {session_id}")
        with self._write_transaction(scope):
            transcript = self._created(scope, session_id)
            if held_for_cas and held_mismatches(transcript, held_for_cas):
                raise HeldRowRewritten(session_id)
            if replacement:
                self._require_session(scope, session_id)
            active = transcript.active()
            floor = None if watermark is None else int(watermark.last_active or 0)
            superseded, tail = _superseded(active, replacement, carried_tail_count, _carried_ids(active, carried), _proved_ids(active, coverage, floor), watermark)
            for row in active:
                row.visibility = RowVisibility.REWOUND if row.id in superseded else RowVisibility.COMPACTION_ARCHIVED
            inserted = insert_rows(self._ns(scope), transcript, replacement)
            clone_rows(self._ns(scope), transcript, tail)
            transcript.recount()
            if model_config_patch is not None:
                self._merge_model_config(scope, session_id, model_config_patch)
        return CompactionResult(active_count=len(transcript.active()), rows=tuple(row_stamp(row) for row in inserted))

    def rewritten_held_row_ids(self, session_id: SessionId, held: Sequence[HeldMessage]) -> tuple[MessageId, ...]:
        scope = self._backend.scope
        return tuple(MessageId(i) for i in held_mismatches(self._transcript(scope, session_id), held))

    def rewrite_pruned_rows(self, session_id: SessionId, changes: Sequence[PrunedRowChange], *,
                            model_config_patch: Optional[Mapping[str, JsonValue]] = None, turn_lease_holder: Optional[str] = None,
                            turn_lease_ttl_seconds: float = 300.0) -> PruneRewriteResult:
        scope = self._backend.scope
        # The turn lease is checked first and renewed only once every refusal has passed.
        verify_turn_lease(self._backend, scope, session_id, turn_lease_holder, turn_lease_ttl_seconds, renew=False)
        if model_config_patch is not None and str(session_id) not in self._sessions(scope):
            raise ValueError(f"Session not found: {session_id}")
        transcript = self._created(scope, session_id)
        planned: list[Optional[tuple[_StoredRow, NewMessage]]] = []
        for change in changes:
            row = _pruned_target(transcript, change.original, session_id)
            if change.replacement.role != row.message.role:
                raise PruneRowUnresolved(session_id, "a prune replacement changed the message role")
            # Only what the prune changed against the row as written is overwritten; the rest stays as stored.
            keep = row.message.role == "assistant"
            before, after = _payload(change.before, keep_reasoning=keep), _payload(change.replacement, keep_reasoning=keep)
            changed = {f.name: getattr(after, f.name) for f in fields(NewMessage)
                       if f.name not in _PRUNE_FIXED and getattr(after, f.name) != getattr(before, f.name)}
            planned.append((row, replace(row.message, **changed)) if changed else None)
        verify_turn_lease(self._backend, scope, session_id, turn_lease_holder, turn_lease_ttl_seconds)
        snapshots: list[Optional[str]] = []
        for entry in planned:
            if entry is None:
                snapshots.append(None)
                continue
            row, rewritten = entry
            # The archived twin keeps the pre-prune body; both keep the logical message's display slot, the live row showing the stub in place.
            slot = row.display_slot if row.display_slot is not None else self._display_slot(transcript, row)
            transcript.rows.append(_StoredRow(id=mint_row_id(self._ns(scope)), message=row.message, visibility=RowVisibility.COMPACTION_ARCHIVED, display_slot=slot))
            row.message, row.display_slot = rewritten, slot
            self._bump_transcript_revision(scope, session_id)
            snapshots.append(_row_snapshot(rewritten))
        if model_config_patch is not None:
            self._merge_model_config(scope, session_id, model_config_patch)
        return PruneRewriteResult(rewritten=len(changes) - snapshots.count(None), snapshots=tuple(snapshots))

    @staticmethod
    def _display_slot(transcript: _Transcript, row: _StoredRow) -> int:
        """The display slot ``row`` holds now: the first row of its logical message."""
        from .transcript_store import display_slots

        pairs = [("", r) for r in transcript.rows]
        return next((slot for slot, (_sid, chosen) in display_slots(pairs) if chosen is row), row.id)
