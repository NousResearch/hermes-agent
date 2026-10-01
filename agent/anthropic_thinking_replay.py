"""Durable request-local suppression for Anthropic thinking rejected by signature validation.

Canonical history stays untouched. We persist only fingerprints of rejected opaque
signature/data values, then filter those blocks from each rebuilt request copy. This is
needed beyond the immediate retry because context selection and process resume can rebuild
from canonical history later. The state belongs to one session (and is carried onto its
compression continuation); it applies only where Anthropic signs the blocks.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any


logger = logging.getLogger(__name__)

_MODEL_CONFIG_KEY = "_anthropic_rejected_thinking"
_THINKING_TYPES = frozenset({"thinking", "redacted_thinking"})
_CARRIERS = ("reasoning_details", "anthropic_content_blocks", "_anthropic_content_blocks")


def _fingerprint(block: Any) -> str | None:
    if not isinstance(block, dict) or block.get("type") not in _THINKING_TYPES:
        return None
    kind = block["type"]
    value = block.get("signature" if kind == "thinking" else "data")
    if value in (None, "", b""):
        return None
    payload = value if isinstance(value, str) else repr(value)
    return hashlib.sha256(f"{kind}\0{payload}".encode("utf-8", "replace")).hexdigest()


def tracks_rejected_thinking(agent: Any) -> bool:
    """Native Anthropic signatures only: Kimi, DeepSeek and third-party routes keep their own replay
    contract and the one-request ``reasoning_details`` repair."""
    from agent.anthropic_thinking_policy import anthropic_thinking_route

    return getattr(agent, "api_mode", None) == "anthropic_messages" and anthropic_thinking_route(
        getattr(agent, "base_url", None), getattr(agent, "model", None)
    ) == "native"


def rejected_thinking_fingerprints(session_db: Any, session_id: Any) -> set[str]:
    """The persisted rejection fingerprints of ``session_id``."""
    getter = getattr(session_db, "get_session_model_config_value", None)
    if not session_id or not callable(getter):
        return set()
    try:
        raw = getter(session_id, _MODEL_CONFIG_KEY, [])
    except Exception:
        logger.debug("Anthropic thinking suppression restore failed", exc_info=True)
        return set()
    return {value for value in raw if isinstance(value, str) and value} if isinstance(raw, list) else set()


def session_rejected_thinking(holder: Any, session_db: Any, session_id: Any) -> set[str]:
    """``holder``'s in-memory fingerprints for ``session_id``, else the persisted ones. The in-memory
    set is authoritative: it outlives a failed or disabled persist."""
    cached = getattr(holder, "_anthropic_rejected_thinking", None)
    if isinstance(cached, tuple) and cached[0] == session_id:
        return cached[1]
    return rejected_thinking_fingerprints(session_db, session_id)


def _bind(agent: Any, session_id: Any, rejected: set[str]) -> None:
    # The compressor's tail walk must price the same replay as the agent's preflight, so it shares
    # the agent's set object (later in-place updates reach both).
    agent._anthropic_rejected_thinking = (session_id, rejected)
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor._anthropic_rejected_thinking = agent._anthropic_rejected_thinking


def _rejected(agent: Any) -> set[str]:
    # Keyed by session: /new, /resume, /branch and compression rotate ``session_id`` on a live agent.
    session_id = getattr(agent, "session_id", None)
    rejected = session_rejected_thinking(agent, getattr(agent, "_session_db", None), session_id)
    _bind(agent, session_id, rejected)
    return rejected


def _persist(agent: Any, rejected: set[str]) -> None:
    if getattr(agent, "_persist_disabled", False):
        return
    patcher = getattr(getattr(agent, "_session_db", None), "patch_session_model_config", None)
    session_id = getattr(agent, "session_id", None)
    if not session_id or not callable(patcher):
        return
    try:
        patcher(session_id, {_MODEL_CONFIG_KEY: sorted(rejected)})
    except Exception:
        logger.debug("Anthropic thinking suppression persist failed", exc_info=True)


def _mirrored_readable_thinking(message: dict) -> str | None:
    from agent.anthropic_message_convert import assistant_replay_carrier

    _, carrier = assistant_replay_carrier(message)
    readable = [
        block["thinking"]
        for block in carrier
        if block.get("type") == "thinking" and isinstance(block.get("thinking"), str) and block["thinking"]
    ]
    return "\n\n".join(readable) if readable else None


def strip_rejected_thinking(message: Any, rejected: set[str] | None) -> int:
    """Drop rejected thinking blocks (every thinking block when ``rejected`` is None) from one request
    copy. Rebinds carrier keys on ``message`` only; the nested canonical lists are never mutated."""
    if not isinstance(message, dict):
        return 0

    mirror = _mirrored_readable_thinking(message) if message.get("role") == "assistant" else None
    removed = 0
    for key in _CARRIERS:
        blocks = message.get(key)
        if not isinstance(blocks, list):
            continue
        kept = []
        for block in blocks:
            fingerprint = _fingerprint(block)
            if (
                isinstance(block, dict)
                and block.get("type") in _THINKING_TYPES
                and (rejected is None or (fingerprint is not None and fingerprint in rejected))
            ):
                removed += 1
            else:
                kept.append(block)
        if kept:
            message[key] = kept
        else:
            message.pop(key, None)

    # Once a signed carrier is rejected, do not let its canonical readable mirror
    # re-enter native conversion as unsigned reasoning after context replacement.
    if removed and mirror is not None:
        for key in ("reasoning", "reasoning_content"):
            if message.get(key) == mirror:
                message.pop(key, None)
    return removed


def apply_rejected_thinking_suppression(agent: Any, messages: Any) -> int:
    """Filter rejected thinking from a request copy rebuilt from canonical history."""
    if not isinstance(messages, list) or not tracks_rejected_thinking(agent):
        return 0
    rejected = _rejected(agent)
    if not rejected:
        return 0
    return sum(strip_rejected_thinking(message, rejected) for message in messages)


def remember_rejected_thinking(agent: Any, api_messages: Any) -> int:
    """Fingerprint the rejected request and repair its retry copy without mutating history."""
    if not isinstance(api_messages, list):
        return 0

    current = {
        fingerprint
        for message in api_messages
        if isinstance(message, dict)
        for key in _CARRIERS
        for block in (message.get(key) if isinstance(message.get(key), list) else ())
        if (fingerprint := _fingerprint(block)) is not None
    }
    # The provider-visible history changed while the canonical content fingerprint did
    # not, so a previous prompt-token anchor is no longer valid for this request shape.
    from agent.usage_anchor import set_usage_anchor

    set_usage_anchor(agent, None)
    if not current:
        # Nothing to fingerprint: repair this retry copy only. A persisted strip-all would also
        # drop every valid signature the session produces later.
        return sum(strip_rejected_thinking(message, None) for message in api_messages)
    rejected = _rejected(agent)
    rejected.update(current)
    _persist(agent, rejected)
    return sum(strip_rejected_thinking(message, rejected) for message in api_messages)


def carry_rejected_thinking_to_session(agent: Any, old_session_id: str) -> None:
    """Compression publishes the child with the session's initial model_config while its retained
    tail still holds the rejected canonical rows; carry the fingerprints onto ``agent.session_id``."""
    cached = getattr(agent, "_anthropic_rejected_thinking", None)
    if isinstance(cached, tuple) and cached[0] == old_session_id:
        rejected = set(cached[1])
    else:
        rejected = rejected_thinking_fingerprints(getattr(agent, "_session_db", None), old_session_id)
    if not rejected:
        return
    rejected |= _rejected(agent)
    _bind(agent, getattr(agent, "session_id", None), rejected)
    _persist(agent, rejected)
