#!/usr/bin/env python3
"""Deterministic Telegram/OpenClaw transport-topology routing helper.

Wave-5 slice-2 scope:
- codify One Topic = One Lane = One Agent mapping rules
- preserve existing non-topic session key semantics
- normalize Telegram General-topic quirks for deterministic routing

Note: this is the transport/topic router. Model-route selection uses
`scripts/session_topology_router.py`.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

SCHEMA_VERSION = "session.topology.routing_decision.v1"


class SessionTopologyRouteError(ValueError):
    """Structured fail-closed routing error for transport topology resolution."""

    def __init__(self, code: str, *, gate: str, details: Optional[Mapping[str, Any]] = None, hint: Optional[str] = None) -> None:
        self.code = str(code or "transport_route_error").strip() or "transport_route_error"
        self.gate = str(gate or "transport_request").strip() or "transport_request"
        self.details = dict(details) if isinstance(details, Mapping) else {}
        self.hint = str(hint or "").strip() or None

        message_parts = [self.code]
        if self.details:
            message_parts.append(json.dumps(self.details, ensure_ascii=False, sort_keys=True))
        if self.hint:
            message_parts.append(f"hint={self.hint}")
        super().__init__("; ".join(message_parts))

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "gate": self.gate,
            "details": self.details,
        }
        if self.hint:
            payload["hint"] = self.hint
        return payload


def _routing_error(code: str, *, gate: str, details: Optional[Mapping[str, Any]] = None, hint: Optional[str] = None) -> SessionTopologyRouteError:
    return SessionTopologyRouteError(code, gate=gate, details=details, hint=hint)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return False


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = int(text)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _as_non_empty_str(value: Any) -> str:
    return str(value or "").strip()


def _chat_map(topology: Mapping[str, Any], scope: str, chat_id: str) -> Mapping[str, Any]:
    telegram = topology.get("telegram") if isinstance(topology.get("telegram"), Mapping) else {}
    if scope == "group":
        lane_map = telegram.get("groups") if isinstance(telegram.get("groups"), Mapping) else {}
    else:
        lane_map = telegram.get("direct") if isinstance(telegram.get("direct"), Mapping) else {}
    row = lane_map.get(chat_id)
    return row if isinstance(row, Mapping) else {}


def _topic_map(chat_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    topics = chat_cfg.get("topics")
    return topics if isinstance(topics, Mapping) else {}


def build_session_key(*, agent_id: str, scope: str, chat_id: str, thread_id: int | None) -> str:
    """Build canonical session key while preserving current non-topic semantics."""

    parts = ["agent", agent_id, "telegram", scope, chat_id]
    if thread_id is not None:
        parts.extend(["topic", str(thread_id)])
    return ":".join(parts)


def _request_route_lock(request: Mapping[str, Any]) -> dict[str, Any]:
    lock_raw = request.get("route_lock") if isinstance(request.get("route_lock"), Mapping) else {}

    def pick_str(*keys: str) -> str:
        for key in keys:
            value = None
            if isinstance(lock_raw, Mapping) and key in lock_raw:
                value = lock_raw.get(key)
            elif key in request:
                value = request.get(key)
            text = _as_non_empty_str(value)
            if text:
                return text
        return ""

    out: dict[str, Any] = {}
    agent_id = pick_str("requested_agent_id", "agent_id")
    lane_name = pick_str("requested_lane_name", "lane_name")
    session_key = pick_str("requested_session_key", "session_key")

    if agent_id:
        out["agent_id"] = agent_id
    if lane_name:
        out["lane_name"] = lane_name
    if session_key:
        out["session_key"] = session_key

    thread_raw = None
    if isinstance(lock_raw, Mapping) and "requested_message_thread_id" in lock_raw:
        thread_raw = lock_raw.get("requested_message_thread_id")
    elif isinstance(lock_raw, Mapping) and "message_thread_id" in lock_raw:
        thread_raw = lock_raw.get("message_thread_id")
    elif "requested_message_thread_id" in request:
        thread_raw = request.get("requested_message_thread_id")
    elif "message_thread_id" in request:
        thread_raw = request.get("message_thread_id")

    if thread_raw is not None:
        if isinstance(thread_raw, str) and thread_raw.strip().lower() in {"main", "none", "null"}:
            out["message_thread_id"] = None
        else:
            parsed = _as_positive_int(thread_raw)
            if parsed is None:
                raise _routing_error(
                    "route_lock_invalid_message_thread_id",
                    gate="route_lock_alignment",
                    details={"message_thread_id": thread_raw},
                    hint="Provide a positive integer message_thread_id or use null/main for non-topic routing.",
                )
            out["message_thread_id"] = parsed

    return out


def _assert_route_lock(
    *,
    route_lock: Mapping[str, Any],
    lane_name: str,
    agent_id: str,
    session_key: str,
    resolved_thread_id: int | None,
) -> None:
    if not route_lock:
        return

    mismatches: list[dict[str, Any]] = []

    expected_agent = _as_non_empty_str(route_lock.get("agent_id"))
    if expected_agent and expected_agent != agent_id:
        mismatches.append({"field": "agent_id", "expected": expected_agent, "actual": agent_id})

    expected_lane = _as_non_empty_str(route_lock.get("lane_name"))
    if expected_lane and expected_lane != lane_name:
        mismatches.append({"field": "lane_name", "expected": expected_lane, "actual": lane_name})

    if "message_thread_id" in route_lock:
        expected_thread = route_lock.get("message_thread_id")
        if expected_thread != resolved_thread_id:
            mismatches.append(
                {
                    "field": "message_thread_id",
                    "expected": expected_thread,
                    "actual": resolved_thread_id,
                }
            )

    expected_session_key = _as_non_empty_str(route_lock.get("session_key"))
    if expected_session_key and expected_session_key != session_key:
        mismatches.append({"field": "session_key", "expected": expected_session_key, "actual": session_key})

    if mismatches:
        raise _routing_error(
            "route_lock_mismatch",
            gate="route_lock_alignment",
            details={
                "requested": dict(route_lock),
                "resolved": {
                    "agent_id": agent_id,
                    "lane_name": lane_name,
                    "message_thread_id": resolved_thread_id,
                    "session_key": session_key,
                },
                "mismatches": mismatches,
            },
            hint="Regenerate request.route_lock from the resolved transport decision before mutating lane/session state.",
        )


def _resolve_thread_id(
    *,
    topic_mode: bool,
    raw_thread_id: int | None,
    event_kind: str,
    is_topic_message: bool,
    general_topic_thread_id: int,
    reaction_fallback_to_general: bool,
) -> tuple[int | None, str]:
    if not topic_mode:
        return None, "topic_mode_disabled"
    if raw_thread_id is not None:
        return raw_thread_id, "source_message_thread_id"

    if event_kind in {"reaction", "reaction_count"} and reaction_fallback_to_general:
        return general_topic_thread_id, "reaction_without_thread_fallback_general"

    if is_topic_message:
        return general_topic_thread_id, "topic_message_without_thread_fallback_general"

    return general_topic_thread_id, "topic_mode_default_general"


def resolve_session_topology_route(request: Mapping[str, Any], topology: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve deterministic lane/session routing from Telegram transport identity."""

    channel = _as_non_empty_str(request.get("channel") or "telegram")
    if channel != "telegram":
        raise _routing_error(
            "only_telegram_channel_supported",
            gate="transport_request",
            details={"channel": channel or None},
            hint="Use the transport router only for Telegram request tuples.",
        )

    chat = request.get("chat") if isinstance(request.get("chat"), Mapping) else {}
    event = request.get("event") if isinstance(request.get("event"), Mapping) else {}
    route_lock = _request_route_lock(request)

    scope = _as_non_empty_str(chat.get("scope") or "direct").lower()
    if scope not in {"direct", "group"}:
        raise _routing_error(
            "chat_scope_invalid",
            gate="transport_request",
            details={"chat.scope": scope or None},
            hint="Set request.chat.scope to direct or group.",
        )

    chat_id = _as_non_empty_str(chat.get("id"))
    if not chat_id:
        raise _routing_error(
            "chat_id_missing",
            gate="transport_request",
            details={"chat": dict(chat) if isinstance(chat, Mapping) else {}},
            hint="Set request.chat.id to a Telegram chat identifier before routing.",
        )

    event_kind = _as_non_empty_str(event.get("kind") or "message").lower()
    raw_thread_id = _as_positive_int(event.get("message_thread_id"))
    is_topic_message = _as_bool(event.get("is_topic_message"))

    chat_cfg = _chat_map(topology, scope, chat_id)

    global_default_agent = _as_non_empty_str(topology.get("default_agent_id")) or "codex-orchestrator-pro"
    chat_default_agent = _as_non_empty_str(chat_cfg.get("default_agent_id")) or global_default_agent
    chat_default_lane = _as_non_empty_str(chat_cfg.get("default_lane_name")) or "inbox"

    if scope == "group":
        topic_mode = _as_bool(chat.get("is_forum"))
    else:
        topic_mode = _as_bool(chat.get("has_topics_enabled"))

    if "topics_enabled" in chat_cfg:
        topic_mode = _as_bool(chat_cfg.get("topics_enabled"))

    general_topic_thread_id = _as_positive_int(chat_cfg.get("general_topic_thread_id")) or 1
    reaction_fallback_to_general = True
    if "reaction_fallback_to_general" in chat_cfg:
        reaction_fallback_to_general = _as_bool(chat_cfg.get("reaction_fallback_to_general"))

    resolved_thread_id, thread_resolution = _resolve_thread_id(
        topic_mode=topic_mode,
        raw_thread_id=raw_thread_id,
        event_kind=event_kind,
        is_topic_message=is_topic_message,
        general_topic_thread_id=general_topic_thread_id,
        reaction_fallback_to_general=reaction_fallback_to_general,
    )

    topic_cfg = None
    topic_cfg_source = "none"
    if resolved_thread_id is not None:
        topics = _topic_map(chat_cfg)
        maybe_cfg = topics.get(str(resolved_thread_id))
        if isinstance(maybe_cfg, Mapping):
            topic_cfg = maybe_cfg
            topic_cfg_source = "mapped_topic"

    if topic_cfg is not None:
        lane_name = _as_non_empty_str(topic_cfg.get("lane_name")) or f"topic_{resolved_thread_id}"
        agent_id = _as_non_empty_str(topic_cfg.get("agent_id")) or chat_default_agent
        lane_source = "topic_map"
    elif resolved_thread_id is not None:
        lane_name = "general" if resolved_thread_id == general_topic_thread_id else f"topic_{resolved_thread_id}"
        agent_id = chat_default_agent
        lane_source = "topic_default"
    else:
        lane_name = chat_default_lane
        agent_id = chat_default_agent
        lane_source = "chat_default"

    session_key = build_session_key(
        agent_id=agent_id,
        scope=scope,
        chat_id=chat_id,
        thread_id=resolved_thread_id,
    )

    send_policy = chat_cfg.get("send_policy") if isinstance(chat_cfg.get("send_policy"), Mapping) else {}
    omit_general_thread_id = True
    if "omit_general_topic_thread_id_1" in send_policy:
        omit_general_thread_id = _as_bool(send_policy.get("omit_general_topic_thread_id_1"))

    include_thread = resolved_thread_id is not None
    omit_reason = None
    if include_thread and omit_general_thread_id and resolved_thread_id == general_topic_thread_id:
        include_thread = False
        omit_reason = "omit_general_topic_thread_id_1"

    transport_key = f"telegram|{scope}|{chat_id}|{resolved_thread_id if resolved_thread_id is not None else 'main'}"

    _assert_route_lock(
        route_lock=route_lock,
        lane_name=lane_name,
        agent_id=agent_id,
        session_key=session_key,
        resolved_thread_id=resolved_thread_id,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "router_layer": "transport_topology",
        "router_module": "ops/openclaw/continuity/session_topology_router.py",
        "decision": "PASS",
        "final_state": "ROUTED",
        "block_gate": None,
        "block_reason": None,
        "routing_basis": {
            "channel": "telegram",
            "scope": scope,
            "chat_id": chat_id,
            "event_kind": event_kind,
            "topic_mode": topic_mode,
            "thread_resolution": thread_resolution,
            "transport_key": transport_key,
            "route_lock": route_lock,
        },
        "lane": {
            "name": lane_name,
            "agent_id": agent_id,
            "source": lane_source,
            "topic_map_source": topic_cfg_source,
        },
        "transport": {
            "message_thread_id": resolved_thread_id,
            "general_topic_thread_id": general_topic_thread_id,
        },
        "session": {
            "session_key": session_key,
            "is_topic_isolated": resolved_thread_id is not None,
        },
        "outbound": {
            "chat_id": chat_id,
            "include_message_thread_id": include_thread,
            "message_thread_id": resolved_thread_id if include_thread else None,
            "omit_reason": omit_reason,
        },
        "invariants": {
            "transport_only_routing": True,
            "content_independent_routing": True,
            "one_topic_one_lane_one_agent": topic_mode,
        },
    }


def evaluate_session_topology_route(request: Mapping[str, Any], topology: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one transport routing request with structured fail-closed output."""

    try:
        decision = resolve_session_topology_route(request, topology)
    except SessionTopologyRouteError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "router_layer": "transport_topology",
            "router_module": "ops/openclaw/continuity/session_topology_router.py",
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": exc.gate,
            "block_reason": exc.code,
            "error": exc.as_dict(),
            "actionable_failure": {
                "gate": exc.gate,
                "reason": exc.code,
                "hint": exc.hint,
            },
        }
    except Exception as exc:  # pragma: no cover
        return {
            "schema_version": SCHEMA_VERSION,
            "router_layer": "transport_topology",
            "router_module": "ops/openclaw/continuity/session_topology_router.py",
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "transport_request",
            "block_reason": "transport_route_evaluation_failed",
            "error": {
                "error": "transport_route_evaluation_failed",
                "detail": str(exc),
            },
            "actionable_failure": {
                "gate": "transport_request",
                "reason": "transport_route_evaluation_failed",
                "hint": "Inspect request/topology payloads and rerun the transport routing probe.",
            },
        }

    decision["actionable_failure"] = None
    return decision
