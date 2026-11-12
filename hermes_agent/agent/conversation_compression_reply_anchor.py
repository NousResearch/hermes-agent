"""Keep the just-delivered assistant reply live across a compaction commit (#118900).

Sibling of ``agent/conversation_compression.py`` (the facade). The facade
late-imports this module at its single call site in ``compress_context``; this
module must never import the facade at module level (import cycle).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.message_sanitization import coalesce_tool_call_id
from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)

def _ensure_compressed_keeps_last_assistant_reply(
    original_messages: list, compressed: list, *, session_id: Optional[str] = None,
) -> Optional[dict]:
    """Keep the latest visible assistant reply live across compaction (#118900).

    A reply that just finished streaming is the row the user is reading; when an
    engine's fold drops it into the summary region, the commit archives its row
    (active=0) and surfaces that render it from the active set drop it on the
    next refresh — while the content sits intact on disk. The built-in
    compressor keeps this row in the tail (``_ensure_last_assistant_message_in_tail``),
    but plugin engines implement their own ``compress()`` without that guard, so
    it is enforced here, next to ``_ensure_compressed_has_user_turn``, where
    every engine and every path (threshold preflight, engine maintenance,
    manual /compress; in-place and rotation) routes through.

    Runs FIRST at the commit boundary — before the todo-snapshot fold and the
    user-turn anchor — so the follower rows still read as the engine returned
    them and the later passes see the reply as the tail they place around
    (``[summary, reply, U_new]`` / ``[..., reply, TODO-row]``). Placement is
    chronological (see ``_reply_insertion_index``); when the only slot would
    sit assistant-adjacent to a non-twin row the reinsertion is skipped, never
    forced: strict alternation beats recovery. Empty (reasoning-only) and
    tool-call rows are out of scope: the former is a different bug family, the
    latter must keep its atomic tool group.

    Returns the exact dict placed into ``compressed`` (the caller names it as a
    carried row at commit), or ``None`` when nothing was reinserted — every skip
    that is not "the reply is already there" is logged here with its reason.
    """
    from agent.context_compressor import (
        _DB_PERSISTED_MARKER, _fresh_compaction_message_copy, is_compaction_summary_message,
    )
    from agent.conversation_compression import _message_text

    reply, reply_text = None, ""
    for message in reversed(original_messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if message.get("tool_calls") or is_compaction_summary_message(message):
            continue
        reply_text = _message_text(message).strip()
        if not reply_text:
            continue
        reply = message
        break
    if reply is None:
        return None

    def _is_reply_twin(message: Any) -> bool:
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or message.get("tool_calls")
        ):
            return False
        return _same_visible_content(message, reply, reply_text)

    if any(_is_reply_twin(message) for message in compressed):
        return None
    anchor = _fresh_compaction_message_copy(reply)
    # Identity scan, not list.index(): duplicate rows with identical content
    # exist in the wild, and == would resolve to the older twin (#118900).
    reply_pos = next(i for i, m in enumerate(original_messages) if m is reply)
    index = _reply_insertion_index(
        original_messages[reply_pos + 1:], compressed, reused_ids=_reused_tool_call_ids(original_messages),
    )
    if index is None:
        logger.warning(
            "Compression: engine folded away the just-delivered assistant reply and the surviving tool "
            "rounds reuse tool-call ids, so its slot cannot be located; not reinserting it (session=%s).",
            session_id or "none",
        )
        return None
    # Either neighbour already being a non-tool assistant twin means the reply is
    # effectively present at this seam; inserting would create assistant;assistant.
    neighbours = [compressed[i] for i in (index - 1, index) if 0 <= i < len(compressed)]
    if any(_is_reply_twin(message) for message in neighbours):
        return None
    # A non-twin assistant neighbour (an older kept reply, the engine's own paraphrase of
    # this one, or a tool-call row) means the slot cannot take the row without
    # assistant;assistant adjacency, and sliding past it would only move the seam while
    # putting the newest reply before an older one. Strict alternation is the invariant
    # every provider path relies on; the reply stays readable on disk, so skip the recovery
    # rather than break the shape. Tool-call rows are NOT exempt on either side: to the left
    # the row would separate a call from its result; to the right the user-turn anchor is
    # only best-effort (it yields to a merged in-flight replay, a surviving real user row or
    # a busy steer), so the pair would reach the durable active set and the pre-call belt
    # would then merge the reply INTO the tool-call row, diverging live from DB.
    if any(isinstance(m, dict) and m.get("role") == "assistant" for m in neighbours):
        logger.warning(
            "Compression: engine folded away the just-delivered assistant reply and left a non-matching "
            "assistant row at its slot; not reinserting it to keep strict role alternation (session=%s).",
            session_id or "none",
        )
        return None
    # The built-in compressor deliberately lets a reply that alone dominates the window fold
    # (``_ensure_last_assistant_message_in_tail`` yields to ``head_end + 1``) so compaction can
    # make progress; forcing it back would make the candidate no smaller than the input, and
    # the commit-site no-growth guard would then refuse the attempt and add an ineffective
    # strike — a session that can never compact is worse than a folded reply. Measure with
    # the same rough estimate that guard uses and mirror its comparison (it refuses only on
    # `>`, so exact break-even is still accepted). The estimate is a per-row sum, so the
    # anchor's cost is added rather than re-estimating the whole candidate. The `<` pre-check
    # limits the yield to compactions that ARE shrinking: a candidate that already fails to
    # shrink is refused by the guard (or repaired by salvage) whether or not the reply is in
    # it, so dropping the reply there would buy nothing.
    rough_in = estimate_messages_tokens_rough(original_messages)
    rough_out = estimate_messages_tokens_rough(compressed)
    if rough_out < rough_in < rough_out + estimate_messages_tokens_rough([anchor]):
        logger.info(
            "Compression: not reinserting the just-delivered assistant reply; it would stop this "
            "compaction from shrinking the transcript (session=%s).", session_id or "none",
        )
        return None
    # Post-commit contract (#98450, mirrors _insert_real_user_anchor._place):
    # archive_and_compact durably writes every dict in `compressed` as the new
    # active set, so stamp the copy or the next flush re-INSERTs it as a duplicate.
    anchor[_DB_PERSISTED_MARKER] = True
    compressed.insert(index, anchor)
    return anchor


def _same_visible_content(left: dict, right: dict, right_text: str) -> bool:
    """Visible-content equality: normalized text when either side has any, else raw ``content``.

    A conforming engine may hand a row back whitespace-stripped or with parts-list
    content re-rendered as a string; raw ``==`` would treat that as a different row
    (the twin check would insert a duplicate, the follower scan would miss its slot).
    ``left`` is the per-candidate row and is flattened here; ``right_text`` is the
    caller's row flattened once per scan rather than once per candidate.
    """
    from agent.conversation_compression import _message_text

    left_text = _message_text(left).strip()
    if left_text or right_text:
        return left_text == right_text
    return left.get("content") == right.get("content")


def _tool_call_ids(message: dict) -> frozenset:
    """Pairing ids of an assistant row's tool calls, keyed the way the pre-call sanitizer
    keys them (``call_id`` before ``id``, composite ``call|item`` split): Codex rows carry
    ``id=fc_…`` AND ``call_id=call_…`` and their results pair on the latter."""
    return frozenset(filter(None, map(coalesce_tool_call_id, message.get("tool_calls") or ())))


def _tool_result_id(message: dict) -> str:
    """The ``tool_call_id`` of a ``tool`` row under the same normalisation as ``_tool_call_ids``."""
    return coalesce_tool_call_id({"id": message.get("tool_call_id")})


def _reused_tool_call_ids(messages: list) -> frozenset:
    """Tool-call ids that more than one assistant row in ``messages`` uses.

    ``tool_call_id`` is not unique in practice: llama.cpp emits one constant id for
    every call it ever returns, and other providers reuse ``call_0`` per turn (see
    ``_dedupe_tool_call_ids``). Such ids cannot locate a specific row.
    """
    seen: dict = {}
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            for call_id in _tool_call_ids(message):
                seen[call_id] = seen.get(call_id, 0) + 1
    return frozenset(call_id for call_id, count in seen.items() if count > 1)


def _reply_insertion_index(followers: list, compressed: list, *, reused_ids: frozenset = frozenset()) -> Optional[int]:
    """Chronologically correct slot for the dropped reply inside ``compressed``.

    ``followers`` are the ORIGINAL rows after the reply (the next user turn, its
    tool rounds, ...). The slot is just before the surviving row that originally
    followed it: the LAST match in ``compressed`` on role + normalized text
    (a first-match scan binds to an older identical "ok"/"thanks" twin and
    resurfaces the reply mid-history; raw ``==`` misses a follower the engine
    re-rendered from a parts list). When no follower survives but the original
    had a real user turn after the reply, the reply must still never land after
    a trailing REAL user row — that is the turn the loop is about to answer, and
    an ``assistant`` tail would present the old reply as its answer — so it goes
    right before that row instead (user-role scaffolding such as a todo snapshot
    or handoff row is not that turn; the reply goes after it and the user-turn
    anchor places the real turn behind the reply). No followers at all
    (idle/maintenance compaction right after the reply) → the reply is genuinely
    the tail.

    Tool rounds whose ids the transcript reuses (``reused_ids``) are not usable
    anchors: the id would bind to whichever round happens to sit last and put the
    reply mid-chain. Such followers are passed over in favour of the remaining
    real followers; when none locates the slot and a reused-id round survives in
    ``compressed``, the placement is unknown and ``None`` is returned (skip).
    """
    from agent.conversation_compression import _is_real_user_message, _message_text

    def _row_ids(message: dict) -> frozenset:
        return _tool_call_ids(message) | frozenset(filter(None, (_tool_result_id(message),)))

    followers = [f for f in followers if isinstance(f, dict)]
    ambiguous = False
    for follower in followers:
        follower_role, follower_text = follower.get("role"), _message_text(follower).strip()
        follower_call_ids = _tool_call_ids(follower)
        if _row_ids(follower) & reused_ids:
            ambiguous = ambiguous or any(
                isinstance(m, dict) and _row_ids(m) & reused_ids for m in compressed
            )
            continue
        for pos in range(len(compressed) - 1, -1, -1):
            message = compressed[pos]
            if not isinstance(message, dict) or message.get("role") != follower_role:
                continue
            # Tool rounds carry no visible text: a tool-call assistant has ``content None``
            # and tool results repeat ("ok", "{}"), so content equality binds to whichever
            # such row sits LAST (mid-turn compaction keeps the whole current tool chain).
            # The call ids are unique per round; match on those instead.
            message_call_ids = _tool_call_ids(message)
            if follower_call_ids or message_call_ids:
                if message_call_ids == follower_call_ids:
                    return pos
                continue
            if follower_role == "tool" and (_tool_result_id(follower) or _tool_result_id(message)):
                if _tool_result_id(message) == _tool_result_id(follower):
                    return pos
                continue
            if _same_visible_content(message, follower, follower_text):
                return pos
    if ambiguous:
        return None
    if (
        followers
        and any(_is_real_user_message(f) for f in followers)
        and compressed
        and _is_real_user_message(compressed[-1])
    ):
        return len(compressed) - 1
    return len(compressed)
