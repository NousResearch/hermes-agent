"""Bedrock Converse reasoning-replay shaping and region-sealed blob recovery (issue #115865).

Sibling of ``agent/bedrock_adapter.py``: the facade owns wire calls and format conversion, this
module owns the two reasoning-replay rules that are pure functions of a payload.

**Tagged union.** ``ReasoningContentBlock`` is a botocore *tagged union*: a block carries exactly
one of ``reasoningText`` / ``redactedContent``. Hermes captures a turn's reasoning into one
JSON-safe sidecar dict (``{"text": ..., "redactedContentBase64": ...}``), so replay has to fan it
back out into separate blocks. Emitting the captured ``text`` verbatim produced
``reasoningContent: {"text": ...}`` and died client-side with ``ParamValidationError`` before the
request ever left the process — every Converse model with thinking enabled, Kimi K3 included.

**Region-sealed encrypted reasoning.** ``redactedContent`` is sealed to the model *and* the region
that minted it. Replaying it elsewhere — after an in-place model switch, or against a ``global.*``
cross-region inference profile that routed to a different region than the one that encrypted it —
is rejected with ``ValidationException`` ("Encrypted content cannot be used in a different region
from the one that created it."). The blob is opaque and cannot be re-derived, so the only recovery
is to drop those blocks and resend once, mirroring the ``cachePoint`` self-heal in the facade
(same-object return means a retry cannot help). Unsealed ``reasoningText`` is preserved, so a
stripped turn keeps its visible chain of thought instead of degrading into a bare tool call.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bedrock reports a region/model-sealed reasoning blob as a ValidationException whose body varies by
# model family: Kimi K3 wraps a Moonshot ``validation_error`` naming the region, Anthropic models say
# the redacted block's encrypted content is not valid for this model. Both mean the same thing —
# this payload cannot be replayed here — so both must reach the strip-and-resend recovery.
_SEALED_REASONING_REJECTION_RE = re.compile(
    r"(?=.*\bvalidation\w*\b)"
    r"(?=.*(?:encrypted|redacted))"
    r"(?=.*(?:different region|not valid for this model|created it|reformat your input))",
    re.IGNORECASE | re.DOTALL,
)


def is_sealed_reasoning_rejection(exc: BaseException) -> bool:
    """True when Bedrock refused a replayed encrypted-reasoning blob (wrong region or wrong model)."""
    return _SEALED_REASONING_REJECTION_RE.search(str(exc)) is not None


def replay_reasoning_blocks(reasoning: Any, *, decode_redacted) -> List[Dict[str, Any]]:
    """One captured reasoning sidecar -> the Converse blocks that replay it.

    Returns ``reasoningText`` and ``redactedContent`` as SEPARATE single-member blocks (the tagged
    union forbids packing both). An undecodable redacted payload is skipped without discarding the
    thinking text that accompanied it.
    """
    if not isinstance(reasoning, dict):
        return []
    blocks: List[Dict[str, Any]] = []
    text = reasoning.get("text")
    if isinstance(text, str):
        blocks.append({"reasoningContent": {"reasoningText": {"text": text}}})
    encoded = reasoning.get("redactedContentBase64")
    if isinstance(encoded, str) and encoded:
        redacted = decode_redacted(encoded)
        if redacted is not None:
            blocks.append({"reasoningContent": {"redactedContent": redacted}})
    return blocks


def _is_sealed_reasoning_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and isinstance(block.get("reasoningContent"), dict)
        and "redactedContent" in block["reasoningContent"]
    )


def _content_without_sealed_reasoning(content: Any) -> Optional[List[Any]]:
    """``content`` minus sealed reasoning blocks, or None when it is not a list / nothing was sealed."""
    if not isinstance(content, list):
        return None
    kept = [block for block in content if not _is_sealed_reasoning_block(block)]
    return None if len(kept) == len(content) else kept


def strip_sealed_reasoning(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of Converse ``kwargs`` without sealed reasoning blocks; the SAME object when nothing was
    sealed (callers use identity to decide that a retry cannot help).

    A turn left with no content at all is dropped: it carried nothing but an unreplayable blob, and
    Converse rejects an empty content list. Dropping it can leave two same-role neighbours, which
    Converse also rejects ("A conversation must alternate between user and assistant roles"), so
    neighbours that collide are merged — the same rule ``convert_messages_to_converse`` applies when
    it builds the turn list. Turns that still hold text/toolUse keep their position.
    """
    messages = kwargs.get("messages")
    if not isinstance(messages, list):
        return kwargs
    stripped: List[Any] = []
    changed = False
    for message in messages:
        kept = _content_without_sealed_reasoning(
            message.get("content") if isinstance(message, dict) else None
        )
        if kept is None:
            _append_merging_same_role(stripped, message)
            continue
        changed = True
        if kept:
            _append_merging_same_role(stripped, {**message, "content": kept})
    if not changed:
        return kwargs
    return {**kwargs, "messages": stripped}


def _append_merging_same_role(turns: List[Any], message: Any) -> None:
    """Append ``message``, folding its content into the previous turn when the roles collide."""
    role = message.get("role") if isinstance(message, dict) else None
    previous = turns[-1] if turns else None
    if (
        role is not None
        and isinstance(previous, dict)
        and previous.get("role") == role
        and isinstance(previous.get("content"), list)
        and isinstance(message.get("content"), list)
    ):
        turns[-1] = {**previous, "content": [*previous["content"], *message["content"]]}
        return
    turns.append(message)


def recover_from_sealed_reasoning_rejection(
    exc: BaseException, kwargs: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Retry kwargs with sealed reasoning dropped, or None when the error was something else / there
    was nothing sealed to drop (the caller then re-raises)."""
    if not is_sealed_reasoning_rejection(exc):
        return None
    retry_kwargs = strip_sealed_reasoning(kwargs)
    if retry_kwargs is kwargs:
        return None
    logger.warning(
        "bedrock: %s rejected replayed encrypted reasoning (sealed to the model/region that minted "
        "it) - dropping those blocks and resending once. Visible reasoning text is preserved.",
        str(kwargs.get("modelId", "")) or "model",
    )
    return retry_kwargs
