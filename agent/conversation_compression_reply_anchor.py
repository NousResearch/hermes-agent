"""Keep the just-delivered assistant reply live across a compaction commit (#118900).

Sibling of ``agent/conversation_compression.py`` (the facade). The facade
late-imports this module at its single call site in ``compress_context``; this
module must never import the facade at module level (import cycle).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.message_metadata import message_uid_or_none, record_absorbed_message
from agent.message_sanitization import coalesce_tool_call_id
from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)

# Live-only ordered carriers that replay INSTEAD of ``content`` (native Anthropic, Bedrock Converse). A merged
# row without them replays from ``content`` and the persisted ``reasoning_details``, as its durable copy does.
_ORDERED_REPLAY_CARRIERS = ("anthropic_content_blocks", "bedrock_content_blocks")

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
    sit assistant-adjacent to a non-twin assistant row the row is never added
    as a peer (strict alternation beats recovery) and its text is merged into
    that neighbour instead (``_absorb_into_assistant_neighbour``, #144546), so
    a reply the user has already read is not lost. Empty (reasoning-only) and
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
            "rounds reuse tool-call ids, so its slot cannot be located; not reinserting it "
            "(session=%s, reply=%s, neighbour=tool_call).",
            session_id or "none", message_uid_or_none(reply) or "none",
        )
        return None
    # Either neighbour already being a non-tool assistant twin means the reply is
    # effectively present at this seam; inserting would create assistant;assistant.
    neighbours = [compressed[i] for i in (index - 1, index) if 0 <= i < len(compressed)]
    if any(_is_reply_twin(message) for message in neighbours):
        return None
    # A non-twin assistant neighbour (an older reply the engine kept) means the slot cannot take the
    # row as a PEER without assistant;assistant adjacency, and sliding past it would only move the
    # seam while putting the newest reply before an older one. Strict alternation is the invariant
    # every provider path relies on, so the row is never inserted beside it — but the delivered text
    # is not dropped either: it is merged into that older reply, the row that already occupies the
    # slot (#144546). Tool-call and handoff neighbours are not mergeable and keep the old skip.
    if _absorb_into_assistant_neighbour(
        original_messages, compressed, index, reply_pos, anchor, reply_text, session_id=session_id,
    ):
        return None
    # The built-in compressor deliberately lets a reply that alone dominates the window fold
    # (``_ensure_last_assistant_message_in_tail`` yields to ``head_end + 1``) so compaction can
    # make progress; forcing it back would make the candidate no smaller than the input, and
    # the commit-site no-growth guard would then refuse the attempt and add an ineffective
    # strike — a session that can never compact is worse than a folded reply. The `<` pre-check
    # limits the yield to compactions that ARE shrinking: a candidate that already fails to
    # shrink is refused by the guard (or repaired by salvage) whether or not the reply is in it,
    # so dropping the reply there would buy nothing.
    if _reply_would_block_shrinking(original_messages, compressed, anchor):
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


def _absorb_into_assistant_neighbour(
    original_messages: list, compressed: list, index: int, reply_pos: int, anchor: dict,
    reply_text: str, *, session_id: Optional[str] = None,
) -> bool:
    """Carry the delivered reply INSIDE the assistant row that took its slot (#144546).

    The slot cannot hold the reply as a peer row: an assistant neighbour would make the compacted
    transcript assistant;assistant, and strict alternation is the invariant every provider path
    relies on. Dropping the reply instead loses text the user has already read, so the ONE row that
    can safely absorb it does: an OLDER reply the engine kept, sitting just before the slot, with
    the delivered text appended (the reply is the newest visible assistant row of the original
    transcript, so it belongs after that row). Row count, role shapes and chronology are unchanged.

    The neighbour must be an older reply in fact, not just by role: its visible text has to match an
    assistant row that precedes the reply in ``original_messages``. That is what keeps the compaction
    handoff row (the engine's summary standing in for the whole folded region) and any engine-invented
    text out of this path — folding a reply into the summary would rewrite the handoff itself.

    Not mergeable, and logged as before: a tool-call neighbour (that row owns an atomic call group,
    and it can sit on either side of the slot), a compaction summary row, and an assistant neighbour
    on the FOLLOWING side (nothing after the slot is older than the reply). The no-growth pre-check
    applies here too, charged what the merge adds to that row: a compaction that stops shrinking is
    refused at the commit site.

    The merge lands on a copy that replaces the neighbour in ``compressed``: an engine's kept row can be
    the live original, which must stay as stored if this compaction is refused later. The copy is
    ``model_only`` for display, as micro-compaction's user-turn join is: both source rows stay in
    display history, so showing the join too would paint the reply twice.

    Returns True when an assistant row holds the slot (absorbed, or skipped for a reason logged
    here), False when the slot is free and the caller may insert.
    """
    from agent.context_compressor import MODEL_ONLY_DISPLAY_METADATA_KEY
    from agent.turn_context import drop_stale_api_content

    pos = next(
        (
            candidate
            for candidate in (index - 1, index)
            if 0 <= candidate < len(compressed)
            and isinstance(compressed[candidate], dict)
            and compressed[candidate].get("role") == "assistant"
        ),
        None,
    )
    if pos is None:
        return False
    neighbour = compressed[pos]
    reply_row = original_messages[reply_pos]
    reply_uid = message_uid_or_none(reply_row) or "none"
    kind = _neighbour_kind(neighbour, pos != index - 1, original_messages[:reply_pos])
    if kind:
        logger.warning(
            "Compression: engine folded away the just-delivered assistant reply and left a non-matching "
            "assistant row at its slot; not reinserting it to keep strict role alternation "
            "(session=%s, reply=%s, neighbour=%s).",
            session_id or "none", reply_uid, kind,
        )
        return True
    merged = dict(neighbour)
    if not _append_text_to_row(merged, reply_text):
        logger.warning(
            "Compression: engine folded away the just-delivered assistant reply and the assistant row at "
            "its slot carries no visible text to merge into; not reinserting it "
            "(session=%s, reply=%s, neighbour=empty).",
            session_id or "none", reply_uid,
        )
        return True
    drop_stale_api_content(merged)
    for carrier in _ORDERED_REPLAY_CARRIERS:
        merged.pop(carrier, None)
    merged["display_metadata"] = {**(merged.get("display_metadata") or {}), MODEL_ONLY_DISPLAY_METADATA_KEY: True}
    if _reply_would_block_shrinking(original_messages, compressed, merged, replaced=neighbour):
        logger.info(
            "Compression: not carrying the just-delivered assistant reply into the compacted transcript; "
            "it would stop this compaction from shrinking it (session=%s).", session_id or "none",
        )
        return True
    record_absorbed_message(merged, reply_row)
    compressed[pos] = merged
    logger.info(
        "Compression: engine folded away the just-delivered assistant reply; merged it into the assistant "
        "row at its slot instead of adding a peer (session=%s, reply=%s).", session_id or "none", reply_uid,
    )
    return True


def _neighbour_kind(neighbour: dict, following: bool, older_messages: list) -> Optional[str]:
    """Name an unmergeable neighbour without changing its provider-owned replay state."""
    from agent.agent_runtime_helpers import _is_codex_interim
    from agent.context_compressor import is_compaction_summary_message
    from agent.conversation_compression import _message_text

    if following:
        return "following"
    if neighbour.get("tool_calls"):
        return "tool_call"
    if is_compaction_summary_message(neighbour):
        return "summary"
    if _is_codex_interim(neighbour):
        return "codex"
    older_reply_texts = {
        _message_text(message).strip()
        for message in older_messages
        if isinstance(message, dict) and message.get("role") == "assistant" and not message.get("tool_calls")
    }
    text = _message_text(neighbour).strip()
    if text not in older_reply_texts:
        return "paraphrase" if text else "empty"
    return None


def _append_text_to_row(row: dict, text: str) -> bool:
    """Append ``text`` to ``row``'s visible content, preserving its shape.

    Parts-list content stays a parts list (the engine's own rendering survives); a plain string
    gains a blank-line-separated block. Returns False — the caller logs and keeps the old skip —
    when the row carries nothing visible to merge into, so an unexpected content shape is never
    overwritten.
    """
    content = row.get("content")
    if isinstance(content, str) and content.strip():
        row["content"] = f"{content.rstrip()}\n\n{text}"
        return True
    if isinstance(content, list) and content:
        row["content"] = [*content, {"type": "text", "text": text}]
        return True
    return False


def _reply_would_block_shrinking(
    original_messages: list, compressed: list, anchor: dict, *, replaced: Optional[dict] = None,
) -> bool:
    """True when carrying the reply would stop this compaction from shrinking the transcript.

    Measured with the same rough estimate as the commit-site no-growth guard and mirroring its
    comparison (it refuses only on ``>``, so exact break-even is still accepted). The estimate is a
    per-row sum, so the reply's cost is added to the candidate rather than re-estimating the whole
    transcript; a merge passes the row its ``anchor`` replaces, so only the growth of that row counts.
    """
    rough_in = estimate_messages_tokens_rough(original_messages)
    rough_out = estimate_messages_tokens_rough(compressed)
    added = estimate_messages_tokens_rough([anchor]) - estimate_messages_tokens_rough([replaced] if replaced else [])
    return rough_out < rough_in < rough_out + added


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
