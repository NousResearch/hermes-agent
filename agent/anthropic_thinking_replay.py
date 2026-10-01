"""Durable request-local suppression for Anthropic thinking rejected by signature validation.

Canonical history stays untouched. We persist only fingerprints of rejected opaque
signature/data values, then filter those blocks from each rebuilt request copy. This is
needed beyond the immediate retry because context selection and process resume can rebuild
from canonical history later.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any


logger = logging.getLogger(__name__)

_MODEL_CONFIG_KEY = "_anthropic_rejected_thinking"
_THINKING_TYPES = frozenset({"thinking", "redacted_thinking"})
_CARRIERS = ("reasoning_details", "anthropic_content_blocks", "_anthropic_content_blocks")
_STRIP_ALL = "*"


def _fingerprint(block: Any) -> str | None:
    if not isinstance(block, dict) or block.get("type") not in _THINKING_TYPES:
        return None
    kind = block["type"]
    value = block.get("signature" if kind == "thinking" else "data")
    if value in (None, "", b""):
        return None
    payload = value if isinstance(value, str) else repr(value)
    return hashlib.sha256(f"{kind}\0{payload}".encode("utf-8", "replace")).hexdigest()


def _rejected(agent: Any) -> set[str]:
    if hasattr(agent, "_anthropic_rejected_thinking"):
        return set(getattr(agent, "_anthropic_rejected_thinking") or set())

    rejected: set[str] = set()
    getter = getattr(getattr(agent, "_session_db", None), "get_session_model_config_value", None)
    session_id = getattr(agent, "session_id", None)
    if session_id and callable(getter):
        try:
            raw = getter(session_id, _MODEL_CONFIG_KEY, [])
            if isinstance(raw, list):
                rejected.update(value for value in raw if isinstance(value, str) and value)
            elif isinstance(raw, dict):
                # Compatibility with the earlier review-fix head.
                values = raw.get("fingerprints")
                if isinstance(values, list):
                    rejected.update(value for value in values if isinstance(value, str) and value)
                if raw.get("strip_all"):
                    rejected.add(_STRIP_ALL)
        except Exception:
            logger.debug("Anthropic thinking suppression restore failed", exc_info=True)

    agent._anthropic_rejected_thinking = rejected
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


def _mirrored_readable_thinking(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    for key in ("reasoning_details", "anthropic_content_blocks"):
        blocks = message.get(key)
        if not isinstance(blocks, list):
            continue
        readable = [
            block.get("thinking")
            for block in blocks
            if (
                isinstance(block, dict)
                and block.get("type") == "thinking"
                and isinstance(block.get("thinking"), str)
                and block.get("thinking")
            )
        ]
        if readable:
            return "\n\n".join(readable)
    return None


def _strip_message(message: Any, rejected: set[str]) -> int:
    if not isinstance(message, dict):
        return 0

    mirror = _mirrored_readable_thinking(message)
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
                and (_STRIP_ALL in rejected or (fingerprint is not None and fingerprint in rejected))
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
    if getattr(agent, "api_mode", None) != "anthropic_messages" or not isinstance(messages, list):
        return 0
    rejected = _rejected(agent)
    if not rejected:
        return 0
    return sum(_strip_message(message, rejected) for message in messages)


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
    rejected = _rejected(agent)
    rejected.update(current or {_STRIP_ALL})
    agent._anthropic_rejected_thinking = rejected
    _persist(agent, rejected)

    # The provider-visible history changed while the canonical content fingerprint did
    # not, so a previous prompt-token anchor is no longer valid for this request shape.
    from agent.usage_anchor import set_usage_anchor

    set_usage_anchor(agent, None)
    return sum(_strip_message(message, rejected) for message in api_messages)
