"""Durable suppression of Anthropic thinking blocks rejected by signature validation.

A signature failure is request-specific evidence about opaque replay blocks, not a reason
to erase canonical reasoning history. Store only hashes of rejected signatures in the
session model config, filter every replay carrier on retry, and reapply the suppression
when a later request (or resumed process) rebuilds the wire copy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Iterable


logger = logging.getLogger(__name__)

_MODEL_CONFIG_KEY = "_anthropic_rejected_thinking"
_THINKING_TYPES = frozenset({"thinking", "redacted_thinking"})


def _opaque_value(block: Any) -> Any:
    if not isinstance(block, dict):
        return None
    if block.get("type") == "thinking":
        return block.get("signature")
    if block.get("type") == "redacted_thinking":
        return block.get("data")
    return None


def _fingerprint(block: Any) -> str | None:
    value = _opaque_value(block)
    if value in (None, "", b""):
        return None
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        payload = repr(value)
    kind = str(block.get("type") if isinstance(block, dict) else "")
    return hashlib.sha256(f"{kind}\0{payload}".encode("utf-8", "replace")).hexdigest()


def _carrier_lists(message: Any) -> Iterable[list]:
    if not isinstance(message, dict):
        return
    for key in ("reasoning_details", "anthropic_content_blocks", "_anthropic_content_blocks"):
        value = message.get(key)
        if isinstance(value, list):
            yield value


def _collect_fingerprints(messages: Any) -> set[str]:
    found: set[str] = set()
    if not isinstance(messages, list):
        return found
    for message in messages:
        for blocks in _carrier_lists(message):
            for block in blocks:
                if isinstance(block, dict) and block.get("type") in _THINKING_TYPES:
                    if fp := _fingerprint(block):
                        found.add(fp)
    return found


def _decode_state(raw: Any) -> tuple[set[str], bool]:
    if isinstance(raw, list):
        return ({v for v in raw if isinstance(v, str) and v}, False)
    if not isinstance(raw, dict):
        return set(), False
    values = raw.get("fingerprints")
    fingerprints = {
        value for value in (values if isinstance(values, list) else [])
        if isinstance(value, str) and value
    }
    return fingerprints, bool(raw.get("strip_all"))


def _load_state(agent: Any) -> tuple[set[str], bool]:
    if getattr(agent, "_anthropic_rejected_thinking_loaded", False):
        return (
            set(getattr(agent, "_anthropic_rejected_thinking_fingerprints", set()) or set()),
            bool(getattr(agent, "_anthropic_rejected_thinking_strip_all", False)),
        )

    fingerprints: set[str] = set()
    strip_all = False
    getter = getattr(getattr(agent, "_session_db", None), "get_session_model_config_value", None)
    session_id = getattr(agent, "session_id", None)
    if session_id and callable(getter):
        try:
            fingerprints, strip_all = _decode_state(getter(session_id, _MODEL_CONFIG_KEY, None))
        except Exception:
            logger.debug("Anthropic thinking suppression restore failed", exc_info=True)

    agent._anthropic_rejected_thinking_fingerprints = fingerprints
    agent._anthropic_rejected_thinking_strip_all = strip_all
    agent._anthropic_rejected_thinking_loaded = True
    return fingerprints, strip_all


def _persist_state(agent: Any, fingerprints: set[str], strip_all: bool) -> None:
    if getattr(agent, "_persist_disabled", False):
        return
    patcher = getattr(getattr(agent, "_session_db", None), "patch_session_model_config", None)
    session_id = getattr(agent, "session_id", None)
    if not session_id or not callable(patcher):
        return
    value = {
        "fingerprints": sorted(fingerprints),
        "strip_all": bool(strip_all),
    }
    try:
        patcher(session_id, {_MODEL_CONFIG_KEY: value})
    except Exception:
        logger.debug("Anthropic thinking suppression persist failed", exc_info=True)


def _should_remove(block: Any, fingerprints: set[str], strip_all: bool) -> bool:
    if not isinstance(block, dict) or block.get("type") not in _THINKING_TYPES:
        return False
    if strip_all:
        return True
    fp = _fingerprint(block)
    return fp is not None and fp in fingerprints


def _mirrored_readable_thinking(message: Any) -> str | None:
    """Readable thinking mirrored into canonical reasoning fields, if identifiable.

    Anthropic normalization stores each signed thinking block in reasoning_details
    and stores their readable text joined by a blank line in reasoning/reasoning_content.
    Ordered blocks duplicate the same data only for interleaved tool turns. Prefer
    reasoning_details so a rejected signed carrier cannot fall back through the generic
    mirror after suppression.
    """
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


def _filter_message(message: Any, fingerprints: set[str], strip_all: bool) -> int:
    if not isinstance(message, dict):
        return 0
    mirrored = _mirrored_readable_thinking(message)
    removed = 0
    for key in ("reasoning_details", "anthropic_content_blocks", "_anthropic_content_blocks"):
        blocks = message.get(key)
        if not isinstance(blocks, list):
            continue
        kept = []
        for block in blocks:
            if _should_remove(block, fingerprints, strip_all):
                removed += 1
            else:
                kept.append(block)
        if kept:
            message[key] = kept
        else:
            message.pop(key, None)

    # Canonical Anthropic storage also keeps the same readable thinking in generic
    # reasoning fields. If any signed block from that exact mirror was rejected,
    # remove the mirror from this request copy too; otherwise context selection can
    # resurrect the rejected thinking as an unsigned block after its carrier is gone.
    if removed and mirrored is not None:
        for key in ("reasoning", "reasoning_content"):
            if message.get(key) == mirrored:
                message.pop(key, None)
    return removed


def apply_rejected_thinking_suppression(agent: Any, messages: Any) -> int:
    """Filter previously rejected thinking from a freshly built request copy."""
    if getattr(agent, "api_mode", None) != "anthropic_messages" or not isinstance(messages, list):
        return 0
    fingerprints, strip_all = _load_state(agent)
    if not fingerprints and not strip_all:
        return 0
    return sum(_filter_message(message, fingerprints, strip_all) for message in messages)


def remember_rejected_thinking(agent: Any, api_messages: Any) -> int:
    """Remember signed blocks from a rejected request and remove them from every carrier.

    The provider error generally does not identify which signed block failed, so all signed
    thinking blocks in that rejected request become suppressed suspects. New blocks produced
    later have different signatures and continue to replay normally.
    """
    if not isinstance(api_messages, list):
        return 0

    current = _collect_fingerprints(api_messages)
    fingerprints, strip_all = _load_state(agent)
    if current:
        fingerprints.update(current)
    else:
        # Defensive fallback: a signature-classified error without an inspectable signature
        # must still change the retry request instead of burning the one-shot on identical bytes.
        strip_all = True

    agent._anthropic_rejected_thinking_fingerprints = fingerprints
    agent._anthropic_rejected_thinking_strip_all = strip_all
    agent._anthropic_rejected_thinking_loaded = True
    _persist_state(agent, fingerprints, strip_all)

    # Suppression changes provider-visible history without changing the canonical
    # content/tool fingerprint used by usage anchors. An old provider count would
    # therefore still appear valid and override the smaller repaired request.
    from agent.usage_anchor import set_usage_anchor

    set_usage_anchor(agent, None)
    return sum(_filter_message(message, fingerprints, strip_all) for message in api_messages)
