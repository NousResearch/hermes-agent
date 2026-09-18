"""Mid-conversation reasoning-effort markers that keep the provider prompt cache warm.

Switching effort between turns (``/reasoning high`` → ``/reasoning low``) changes the top-level
``output_config.effort``, which Anthropic renders into the prompt: the whole prefix cache (tools,
system prompt, every earlier turn) misses on the next request. Opus 5 / Fable 5.1 / Mythos 5.1
accept an empty ``role: system`` message *inside* ``messages`` carrying ``output_config.effort``
(beta ``mid-conversation-output-config-2026-07-01``): the top-level value stays frozen at the
session's baseline and every switch is lowered in place, so the cached prefix survives.
Port of anomalyco/opencode#48513.

A marker is a durable transcript row — ``role: system``, ``display_kind: hidden`` (never painted by
any surface; the compressor treats it as scaffolding) — with the payload in ``display_metadata``.
Routes without a native per-message update strip markers and send the current effort top-level,
which is exactly today's behaviour.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EFFORT_UPDATE_KEY = "effort_update"
ANTHROPIC_MID_CONVERSATION_EFFORT_BETA = "mid-conversation-output-config-2026-07-01"


def effort_update(msg: Any) -> Optional[Dict[str, Any]]:
    """The marker payload (``{"effort", "previous"}``) when ``msg`` is an effort marker, else None.
    Read from the top-level key (per-call API copies) or ``display_metadata`` (durable rows)."""
    if not isinstance(msg, dict) or msg.get("role") != "system":
        return None
    payload = msg.get(EFFORT_UPDATE_KEY)
    if payload is None:
        metadata = msg.get("display_metadata")
        payload = metadata.get(EFFORT_UPDATE_KEY) if isinstance(metadata, dict) else None
    return payload if isinstance(payload, dict) and isinstance(payload.get("effort"), str) else None


def make_effort_update_message(effort: str, previous: Optional[str]) -> Dict[str, Any]:
    return {
        "role": "system",
        "content": "",
        "display_kind": "hidden",
        "display_metadata": {EFFORT_UPDATE_KEY: {"effort": effort, "previous": previous}},
    }


def strip_effort_updates(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``messages`` without markers (the same list object when there are none)."""
    stripped = [m for m in messages if effort_update(m) is None]
    return messages if len(stripped) == len(messages) else stripped


def resolve_effort_updates(
    messages: List[Dict[str, Any]], current: Optional[str]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """``(messages, top_level_effort)`` for a route that lowers markers natively. The top-level
    effort is the FIRST marker's ``previous`` (the baseline every cached request was built with).
    Reverted or forked history can leave the last marker disagreeing with the requested effort;
    then the markers are stripped and the request falls back to a plain top-level change."""
    updates = [u for m in messages if (u := effort_update(m)) is not None]
    if not updates:
        return messages, current
    if updates[-1].get("effort") != current:
        return strip_effort_updates(messages), current
    return messages, updates[0].get("previous")


def requested_effort(reasoning_config: Any) -> Optional[str]:
    """The effort a ``reasoning_config`` asks for; None when reasoning is off/unset (a thinking
    enable/disable flip changes the ``thinking`` parameter itself, which no marker can carry)."""
    if not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
        return None
    effort = reasoning_config.get("effort")
    return str(effort).lower() if isinstance(effort, str) and effort else None


def _baseline_effort(agent: Any) -> Optional[str]:
    """Effort the session's cached prefix was built with: the session row's ``model_config`` keeps
    the FIRST ``reasoning_config`` (the upsert never overwrites it), so a re-created gateway agent
    or a resumed CLI sees the same baseline as the process that started the session."""
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    stored = None
    if db is not None and session_id:
        try:
            stored = db.get_session_model_config_value(session_id, "reasoning_config")
        except Exception:
            logger.debug("effort baseline: session model_config unreadable", exc_info=True)
    if stored is None:
        stored = (getattr(agent, "_session_init_model_config", None) or {}).get("reasoning_config")
    return requested_effort(stored)


def record_effort_switch(agent: Any, messages: List[Dict[str, Any]]) -> bool:
    """Append a marker to ``messages`` when this turn's effort differs from the effort the history
    was sent with. Called right before the new user message lands so the marker sits at a fixed
    position in the durable transcript (a marker that moved each turn would invalidate the tail it
    was meant to protect). Only after a completed assistant turn: a marker between a tool_use and
    its tool_result would break Anthropic's adjacency rule, and an empty history has no cache to
    keep. Returns True when a marker was appended."""
    requested = requested_effort(getattr(agent, "reasoning_config", None))
    if requested is None or not messages:
        return False
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant" or last.get("tool_calls"):
        return False
    previous: Optional[str] = None
    for msg in reversed(messages):
        update = effort_update(msg)
        if update is not None:
            previous = update["effort"]
            break
    else:
        previous = _baseline_effort(agent)
    if previous is None or previous == requested:
        return False
    messages.append(make_effort_update_message(requested, previous))
    logger.info("reasoning effort switched %s -> %s; recorded mid-conversation marker", previous, requested)
    return True
