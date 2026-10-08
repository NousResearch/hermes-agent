"""Client-facing projection helpers for model-only compaction carriers."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from agent.context_compressor import ContextCompressor, is_compaction_summary_message
from agent.message_metadata import ABSORBED_MESSAGE_UIDS, message_uid_or_none, uid_list


_COMPACTION_INTERNAL_FIELDS = (
    "tool_calls",
    "finish_reason",
    "reasoning",
    # Provider replay/metadata fields that ride the wire on every request but are invisible to
    # ``msg["content"]``/``msg["tool_calls"]`` accounting. Codex Responses sessions in particular carry
    # ``codex_reasoning_items`` blobs of ``encrypted_content`` that can dominate the serialized session (a
    # measured 214-turn session held ~115K tokens / 27% of its payload there — #55572).
    # ``reasoning_details`` is handled separately (see ``_reasoning_details_text_chars``): its signed/base64
    # envelope is excluded from the budget, mirroring the preflight estimator's exclusion in
    # ``model_metadata._estimate_message_tokens_without_images`` (#73298).
    # An assistant turn may carry only reasoning/thinking content with no visible text (extended-thinking
    # turns, thinking-only recovery responses). Such a turn is persisted with its reasoning fields and is
    # recallable from the transcript, but dropping it here as "empty" makes it vanish from the
    # resumed/reloaded session view while the desktop's reasoning disclosure has nothing to render. Keep it
    # when it carries reasoning so the "Thinking…" block still shows. (#44022)
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
)


def _restated_uids(message: Dict[str, Any]) -> List[str]:
    """Uids of the unfinished request *message* re-states for display, or [] when it re-states none.

    A standalone restatement keeps the original's uid. A carrier whose only live content is the
    restatement records the original as absorbed. A carrier with real prior content shows that
    content, not the restatement, so it never duplicates the original.
    """
    if ContextCompressor._is_inflight_restatement(message):
        uid = message_uid_or_none(message)
        return [uid] if uid else []
    if not is_compaction_summary_message(message):
        return []
    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    if projected is None or not ContextCompressor._is_inflight_restatement({**projected, "role": "user"}):
        return []
    return _absorbed_uids(message)


def _absorbed_uids(message: Dict[str, Any]) -> List[str]:
    """The merge witness of a live dict, or of a stored row (its JSON ``absorbed_message_uids`` column)."""
    value = message.get(ABSORBED_MESSAGE_UIDS) or message.get("absorbed_message_uids")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return uid_list(value)


def restates_inflight_request(message: Dict[str, Any]) -> bool:
    """True when the row a transcript would paint is a compaction's re-statement of a request."""
    if not isinstance(message, dict):
        return False
    if ContextCompressor._is_inflight_restatement(message):
        return True
    projected = ContextCompressor._strip_context_summary_handoff_message(message) \
        if is_compaction_summary_message(message) else None
    return projected is not None and ContextCompressor._is_inflight_restatement({**projected, "role": "user"})


def restatements_shown_elsewhere(messages: List[Any], *, lineage: bool = False) -> Set[int]:
    """Indexes of re-stated requests whose original row the reader also sees (#131104).

    Lineage reads (archived rows included) carry both the original and its restatement; painting
    both shows the request twice. ``lineage=True`` covers a paged lineage read whose page holds
    only one of the pair: the original is still in the reader's transcript. Active-only reads
    carry just the restatement, the only copy, which stays (unframed, see
    ``project_compaction_message_for_display``).
    """
    if lineage:
        return {index for index, m in enumerate(messages) if isinstance(m, dict) and _restated_uids(m)}
    originals = {
        uid for m in messages
        if isinstance(m, dict) and m.get("role") == "user" and not restates_inflight_request(m)
        and not is_compaction_summary_message(m) and (uid := message_uid_or_none(m))
    }
    return {
        index for index, m in enumerate(messages)
        if isinstance(m, dict) and originals.intersection(_restated_uids(m))
    }


def project_compaction_message_for_display(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return authentic transcript content, or ``None`` for a pure handoff.

    Model-facing recovery history retains the complete carrier. Display
    projections instead remove the handoff, inherited tool state, and internal
    reasoning while preserving any real prior-tail content or live user ask
    embedded in the carrier. A re-stated in-flight request loses its model-only
    frame; a read that also carries the original row hides it via
    ``restatements_shown_elsewhere``.
    """
    if not isinstance(message, dict):
        return None
    if not is_compaction_summary_message(message):
        if ContextCompressor._is_inflight_restatement(message):
            return ContextCompressor._without_inflight_replay_header(message)
        return message.copy()

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    if projected is None:
        return None
    if ContextCompressor._is_inflight_restatement({**projected, "role": "user"}):
        projected = ContextCompressor._without_inflight_replay_header(projected)

    projected = projected.copy()
    for key in _COMPACTION_INTERNAL_FIELDS:
        projected.pop(key, None)
    projected.pop("display_kind", None)
    return projected
