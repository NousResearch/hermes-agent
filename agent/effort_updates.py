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
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EFFORT_UPDATE_KEY = "effort_update"
EFFORT_BASELINE_KEY = "_reasoning_effort_baseline"
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


def make_effort_update_message(
    effort: str, previous: str, *, route: str = "", lineage: str = "", reset: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"effort": effort, "previous": previous}
    if route:
        payload["route"] = route
    if lineage:
        payload["lineage"] = lineage
    if reset:
        payload["reset"] = True
    return {
        "role": "system",
        "content": "",
        "display_kind": "hidden",
        "display_metadata": {EFFORT_UPDATE_KEY: payload},
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
    lineage = updates[-1].get("lineage")
    active = [update for update in updates if not lineage or update.get("lineage") == lineage]
    if not active or active[-1].get("effort") != current:
        return strip_effort_updates(messages), current
    selected = [
        message for message in messages
        if (update := effort_update(message)) is None or update in active
    ]
    return (messages if len(selected) == len(messages) else selected), active[0].get("previous")


def requested_effort(reasoning_config: Any) -> Optional[str]:
    """The effort a ``reasoning_config`` asks for; None when reasoning is off/unset (a thinking
    enable/disable flip changes the ``thinking`` parameter itself, which no marker can carry)."""
    if not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
        return None
    effort = reasoning_config.get("effort")
    return str(effort).lower() if isinstance(effort, str) and effort else None


def _effort_route(agent: Any) -> str:
    """Model/provider identity for one cache lineage."""
    return "|".join((
        str(getattr(agent, "provider", "") or "").strip().lower(),
        str(getattr(agent, "model", "") or "").strip().lower(),
        str(getattr(agent, "base_url", "") or "").strip().rstrip("/").lower(),
    ))


def _baseline_record(agent: Any) -> Optional[Dict[str, Any]]:
    """Return the durable effort/route/lineage record for this session."""
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    stored = None
    if db is not None and session_id:
        try:
            stored = db.get_session_model_config_value(session_id, EFFORT_BASELINE_KEY)
        except Exception:
            logger.debug("effort baseline: session model_config unreadable", exc_info=True)
    if stored is None:
        initial = getattr(agent, "_session_init_model_config", None)
        stored = initial.get(EFFORT_BASELINE_KEY) if isinstance(initial, dict) else None
    if isinstance(stored, dict):
        return dict(stored)

    # Upgrade path for sessions created before the durable route record existed: #114534's
    # original implementation stored the first reasoning_config in the session row.
    legacy = None
    if db is not None and session_id:
        try:
            legacy = db.get_session_model_config_value(session_id, "reasoning_config")
        except Exception:
            logger.debug("effort baseline: legacy reasoning_config unreadable", exc_info=True)
    if legacy is None:
        initial = getattr(agent, "_session_init_model_config", None)
        legacy = initial.get("reasoning_config") if isinstance(initial, dict) else None
    effort = requested_effort(legacy)
    if effort is None:
        return None
    return {
        "route": _effort_route(agent), "effort": effort, "lineage": str(uuid.uuid4()), "legacy": True,
    }


def _write_baseline_record(agent: Any, record: Dict[str, Any]) -> None:
    initial = getattr(agent, "_session_init_model_config", None)
    if isinstance(initial, dict):
        initial[EFFORT_BASELINE_KEY] = dict(record)
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is not None and session_id:
        try:
            db.patch_session_model_config(session_id, {EFFORT_BASELINE_KEY: record})
        except Exception:
            logger.debug("effort baseline: session model_config write failed", exc_info=True)


def set_initial_effort_baseline(agent: Any) -> None:
    """Make a policy-selected first-turn effort the session's durable baseline.

    There is no cached conversation prefix before the first user turn, so selecting a
    different effort does not need a marker.  Persist both the legacy
    ``reasoning_config`` and the route-aware baseline record: a session row may have
    been created eagerly with the configured default, and leaving that value behind
    would make the next turn's marker name the wrong previous effort.
    """
    config = dict(getattr(agent, "reasoning_config", None) or {})
    effort = requested_effort(config)
    if effort is None:
        return
    record = {
        "route": _effort_route(agent),
        "effort": effort,
        "lineage": str(uuid.uuid4()),
    }
    initial = getattr(agent, "_session_init_model_config", None)
    if isinstance(initial, dict):
        initial["reasoning_config"] = dict(config)
        initial[EFFORT_BASELINE_KEY] = dict(record)
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is not None and session_id:
        try:
            db.patch_session_model_config(session_id, {
                "reasoning_config": config,
                EFFORT_BASELINE_KEY: record,
            })
        except Exception:
            # A not-yet-created first-turn row is expected; _session_init_model_config
            # carries the same values into its later creation.
            logger.debug("initial effort baseline: session model_config write skipped", exc_info=True)


def record_effort_switch(agent: Any, messages: List[Dict[str, Any]]) -> bool:
    """Append a marker to ``messages`` when this turn's effort differs from the effort the history
    was sent with. Called right before the new user message lands so the marker sits at a fixed
    position in the durable transcript (a marker that moved each turn would invalidate the tail it
    was meant to protect). Only after a completed assistant turn: a marker between a tool_use and
    its tool_result would break Anthropic's adjacency rule, and an empty history has no cache to
    keep. Returns True when a marker was appended."""
    requested = requested_effort(getattr(agent, "reasoning_config", None))
    route = _effort_route(agent)
    baseline = _baseline_record(agent)
    if baseline and baseline.pop("legacy", False):
        _write_baseline_record(agent, baseline)
    if requested is None:
        if not baseline or baseline.get("route") != route or baseline.get("effort") is not None:
            _write_baseline_record(agent, {
                "route": route, "effort": None, "lineage": str(uuid.uuid4()),
            })
        return False

    last = messages[-1] if messages else None
    can_mark = isinstance(last, dict) and last.get("role") == "assistant" and not last.get("tool_calls")
    if not baseline or baseline.get("route") != route or baseline.get("effort") is None:
        lineage = str(uuid.uuid4())
        _write_baseline_record(agent, {"route": route, "effort": requested, "lineage": lineage})
        if can_mark:
            messages.append(make_effort_update_message(
                requested, requested, route=route, lineage=lineage, reset=True,
            ))
            return True
        return False

    lineage = str(baseline.get("lineage") or uuid.uuid4())
    previous = baseline.get("effort")
    for message in reversed(messages):
        update = effort_update(message)
        if update is not None and update.get("lineage") == lineage:
            previous = update["effort"]
            break
    if not can_mark or not isinstance(previous, str) or previous == requested:
        return False
    messages.append(make_effort_update_message(
        requested, previous, route=route, lineage=lineage,
    ))
    logger.info("reasoning effort switched %s -> %s; recorded mid-conversation marker", previous, requested)
    return True
