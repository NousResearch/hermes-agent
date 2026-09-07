"""Usage-anchored token accounting: the provider's real ``usage.prompt_tokens`` is the only
authoritative context size; the local ``bytes/4`` estimate covers ONLY messages appended since.

An anchor = provider usage at capture + a snapshot of the transcript position it priced:
``base_count`` (len(messages) at capture; the reply is not yet appended and is covered by
``completion_tokens``, so the delta walk skips an assistant row at that index), ``base_last_role``
and ``base_last_fp`` (content fingerprint of the last priced message; compaction, splices and
rewinds replace it → anchor fails closed → full estimation until the next real reading).

The fingerprint (not ``id()``) is the identity: the gateway re-reads the transcript from the DB
every turn and a resumed session runs in a fresh process, so object identity is never stable
across the surfaces where the estimate mattered most (#99421, #104462). The anchor also persists
on the session row (``model_config._usage_anchor``) so a restarted process can restore it; a
restored anchor is honored only while the durable transcript still matches its fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

USAGE_ANCHOR_MODEL_CONFIG_KEY = "_usage_anchor"

# Identity of a priced message = the provider-visible fields that round-trip the session DB
# byte-for-byte. Display/persistence metadata (timestamps, row ids, display kinds) is rewritten
# on reload and would only ever fail the match closed.
_FINGERPRINT_KEYS = ("role", "content", "api_content", "tool_call_id", "tool_calls")


def message_fingerprint(msg: Any) -> Optional[str]:
    """Stable digest of one transcript message over its provider-visible, persisted fields."""
    if not isinstance(msg, dict):
        return None
    payload = {k: msg.get(k) for k in _FINGERPRINT_KEYS if msg.get(k) is not None}
    try:
        raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        raw = repr(sorted(payload.items()))
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def capture_usage_anchor(prompt_tokens: Any, completion_tokens: Any, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a usage anchor from provider-reported usage, or None when usage is unusable."""
    try:
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
    except (TypeError, ValueError):
        return None
    if pt <= 0 or not isinstance(messages, list) or not messages:
        return None  # some endpoints omit usage — caller keeps its anchor
    last = messages[-1]
    return {
        "prompt_tokens": pt,
        "completion_tokens": max(0, ct),
        "base_count": len(messages),
        "base_last_role": last.get("role") if isinstance(last, dict) else None,
        "base_last_fp": message_fingerprint(last),
    }


def _anchor_matches(messages: List[Dict[str, Any]], anchor: Dict[str, Any]) -> bool:
    try:
        base_count = int(anchor.get("base_count") or 0)
    except (TypeError, ValueError):
        return False
    if base_count <= 0 or len(messages) < base_count:
        return False
    base_msg = messages[base_count - 1]
    if not isinstance(base_msg, dict) or base_msg.get("role") != anchor.get("base_last_role"):
        return False
    fp = anchor.get("base_last_fp")
    return isinstance(fp, str) and bool(fp) and message_fingerprint(base_msg) == fp


def anchored_context_tokens(messages: List[Dict[str, Any]], anchor: Optional[Dict[str, Any]], *, charge_stale_thinking: bool = True) -> Optional[int]:
    """Anchored prompt+completion tokens plus a rough estimate of ONLY the messages appended since;
    None when the anchor is missing or stale. The anchored response's own reply is skipped (already
    in completion_tokens). ``charge_stale_thinking`` is forwarded to the delta estimate."""
    if not isinstance(anchor, dict) or not isinstance(messages, list) or not _anchor_matches(messages, anchor):
        return None
    from agent.model_metadata import estimate_messages_tokens_rough

    total = int(anchor["prompt_tokens"]) + int(anchor.get("completion_tokens") or 0)
    delta = messages[int(anchor["base_count"]):]
    if delta and isinstance(delta[0], dict) and delta[0].get("role") == "assistant":
        delta = delta[1:]
    if delta:
        total += estimate_messages_tokens_rough(delta, charge_stale_thinking=charge_stale_thinking)
    return total


def _serialize(anchor: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(anchor, dict):
        return None
    try:
        pt, ct, base_count = (int(anchor.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "base_count"))
    except (TypeError, ValueError):
        return None
    fp, role = anchor.get("base_last_fp"), anchor.get("base_last_role")
    if pt <= 0 or base_count <= 0 or not isinstance(fp, str) or not fp:
        return None
    return {"prompt_tokens": pt, "completion_tokens": max(0, ct), "base_count": base_count,
            "base_last_role": role if isinstance(role, str) else None, "base_last_fp": fp}


def persist_usage_anchor(agent: Any, anchor: Optional[Dict[str, Any]]) -> None:
    """Write (or clear, ``None``) the session row's anchor blob. Best-effort: the row may not exist yet."""
    if getattr(agent, "_persist_disabled", False):
        return
    session_id = getattr(agent, "session_id", None)
    patcher = getattr(getattr(agent, "_session_db", None), "patch_session_model_config", None)
    if not session_id or not callable(patcher):
        return
    try:
        patcher(session_id, {USAGE_ANCHOR_MODEL_CONFIG_KEY: _serialize(anchor)})
    except Exception:
        logger.debug("usage anchor persist failed", exc_info=True)


def set_usage_anchor(agent: Any, anchor: Optional[Dict[str, Any]], *, turn_base: bool = False) -> None:
    """Install ``anchor`` on the agent (``None`` clears) and mirror it to the session row."""
    agent._usage_anchor = anchor
    if turn_base or anchor is None:
        agent._turn_base_usage_anchor = anchor
    persist_usage_anchor(agent, anchor)


def restore_usage_anchor(agent: Any, conversation_history: Optional[List[Dict[str, Any]]]) -> None:
    """On a resumed session, adopt the persisted anchor when ``conversation_history`` still carries
    the priced prefix; otherwise clear the stale blob so it can never suppress compression."""
    if getattr(agent, "_usage_anchor", None) is not None or getattr(agent, "_persist_disabled", False):
        return
    session_id = getattr(agent, "session_id", None)
    getter = getattr(getattr(agent, "_session_db", None), "get_session_model_config_value", None)
    if not session_id or not callable(getter) or not isinstance(conversation_history, list):
        return
    try:
        anchor = _serialize(getter(session_id, USAGE_ANCHOR_MODEL_CONFIG_KEY, None))
    except Exception:
        logger.debug("usage anchor load failed", exc_info=True)
        return
    if anchor is None:
        return
    if _anchor_matches(conversation_history, anchor):
        agent._usage_anchor = anchor
    else:
        persist_usage_anchor(agent, None)


def persisted_anchor_tokens(session_db: Any, session_id: Any, messages: Any) -> Optional[int]:
    """Anchored token figure from the session row's persisted anchor, for callers without a live
    agent (gateway hygiene); None when absent, unreadable, or stale against ``messages``."""
    getter = getattr(session_db, "get_session_model_config_value", None)
    if not session_id or not callable(getter) or not isinstance(messages, list):
        return None
    try:
        anchor = _serialize(getter(session_id, USAGE_ANCHOR_MODEL_CONFIG_KEY, None))
    except Exception:
        logger.debug("usage anchor load failed", exc_info=True)
        return None
    return anchored_context_tokens(messages, anchor) if anchor else None
