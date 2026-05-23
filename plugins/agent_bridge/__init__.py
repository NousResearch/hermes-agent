"""Local bridge for Hermes agents sharing a visible group-chat room.

The plugin keeps bot-to-bot turns inside Hermes when external platforms do not
forward bot-originated mentions to other bots. Real human messages still flow
through the platform adapter; synthetic bridge events are marked internal and
fed back through the normal adapter processing path so transcripts and delivery
stay aligned with regular gateway behavior.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from urllib import error, parse, request

logger = logging.getLogger(__name__)

_DEFAULT_SERVER_URL = "http://127.0.0.1:8791"
_DEFAULT_TOKEN_ENV = "HERMES_AGENT_BRIDGE_TOKEN"
_TOOLSET_NAME = "agent_bridge"
_FORMAT_WAKE_TOOL_NAME = "agent_bridge_format_wake_message"


@dataclass
class RoomConfig:
    room_id: str
    external_targets: list[dict[str, Any]]
    participants: list[dict[str, Any]]
    max_bot_messages: int = 16
    idle_timeout_seconds: int = 1800


@dataclass
class BridgeConfig:
    enabled: bool
    agent_id: str
    display_name: str
    server_url: str
    token: str
    rooms: dict[str, RoomConfig]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_bridge_config() -> BridgeConfig:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logger.debug("agent_bridge config load failed: %s", exc)
        cfg = {}

    raw = cfg.get("agent_bridge") if isinstance(cfg, dict) else {}
    if not isinstance(raw, dict):
        raw = {}

    token_env = str(raw.get("token_env") or _DEFAULT_TOKEN_ENV)
    token = str(os.environ.get(token_env) or raw.get("token") or "")
    agent_id = str(raw.get("agent_id") or os.environ.get("USER") or "agent").strip()
    display_name = str(raw.get("display_name") or agent_id).strip()
    rooms: dict[str, RoomConfig] = {}
    for room_id, room_raw in (raw.get("rooms") or {}).items():
        if not isinstance(room_raw, dict):
            continue
        targets = room_raw.get("external_targets") or []
        if not isinstance(targets, list):
            targets = []
        participants = room_raw.get("participants") or []
        if not isinstance(participants, list):
            participants = []
        rooms[str(room_id)] = RoomConfig(
            room_id=str(room_id),
            external_targets=[t for t in targets if isinstance(t, dict)],
            participants=[p for p in participants if isinstance(p, dict)],
            max_bot_messages=max(1, _as_int(room_raw.get("max_bot_messages"), 16)),
            idle_timeout_seconds=max(30, _as_int(room_raw.get("idle_timeout_seconds"), 1800)),
        )

    return BridgeConfig(
        enabled=_as_bool(raw.get("enabled")),
        agent_id=agent_id,
        display_name=display_name,
        server_url=str(raw.get("server_url") or _DEFAULT_SERVER_URL).rstrip("/"),
        token=token,
        rooms=rooms,
    )


def _platform_value(platform: Any) -> str:
    return str(getattr(platform, "value", platform) or "").lower()


def _target_matches_source(target: dict[str, Any], source: Any) -> bool:
    target_platform = str(target.get("platform") or "").lower()
    if target_platform and target_platform != _platform_value(getattr(source, "platform", "")):
        return False
    target_chat = target.get("chat_id")
    if target_chat is not None and str(target_chat) != str(getattr(source, "chat_id", "")):
        return False
    target_thread = target.get("thread_id")
    if target_thread is not None and str(target_thread) != str(getattr(source, "thread_id", "")):
        return False
    return True


def _room_for_source(cfg: BridgeConfig, source: Any) -> Optional[RoomConfig]:
    for room in cfg.rooms.values():
        if any(_target_matches_source(target, source) for target in room.external_targets):
            return room
    return None


def _room_payload(room: RoomConfig) -> dict[str, Any]:
    return {
        "room_id": room.room_id,
        "external_targets": room.external_targets,
        "participants": room.participants,
        "max_bot_messages": room.max_bot_messages,
        "idle_timeout_seconds": room.idle_timeout_seconds,
    }


def _participant_is_self(participant: dict[str, Any], cfg: BridgeConfig) -> bool:
    return str(participant.get("agent_id") or "") == cfg.agent_id


def _known_agent_ids(room: RoomConfig) -> set[str]:
    return {str(p.get("agent_id")) for p in room.participants if p.get("agent_id")}


def _participant_agent_id(participant: dict[str, Any]) -> str:
    return str(participant.get("agent_id") or "").strip()


def _participant_display_name(participant: dict[str, Any]) -> str:
    return str(participant.get("display_name") or _participant_agent_id(participant)).strip()


def _wake_names_for_participant(participant: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    explicit_wake_names = participant.get("wake_names") or []
    for value in explicit_wake_names:
        value = str(value or "").strip()
        if value:
            names.add(value)
    if not explicit_wake_names:
        for value in participant.get("mention_names") or []:
            value = str(value or "").strip()
            if value.startswith("@"):
                names.add(value)
        for value in (_participant_display_name(participant), _participant_agent_id(participant)):
            if value:
                names.add(f"@{value}")
    return sorted(names, key=len, reverse=True)


def _preferred_wake_name_for_participant(participant: dict[str, Any]) -> str:
    for value in participant.get("wake_names") or []:
        value = str(value or "").strip()
        if value:
            return value
    for value in participant.get("mention_names") or []:
        value = str(value or "").strip()
        if value.startswith("@"):
            return value
    display_name = _participant_display_name(participant)
    if display_name:
        return f"@{display_name}"
    agent_id = _participant_agent_id(participant)
    if agent_id:
        return f"@{agent_id}"
    return ""


def _wake_names_for_self(room: RoomConfig, cfg: BridgeConfig) -> list[str]:
    names: set[str] = set()
    matched_participant = False
    for participant in room.participants:
        if not _participant_is_self(participant, cfg):
            continue
        matched_participant = True
        names.update(_wake_names_for_participant(participant))
    if not matched_participant:
        for value in (cfg.display_name, cfg.agent_id):
            value = str(value or "").strip()
            if value:
                names.add(f"@{value}")
    return sorted(names, key=len, reverse=True)


def _wakes_self(text: str, room: RoomConfig, cfg: BridgeConfig) -> bool:
    folded = str(text or "").casefold()
    return any(name.casefold() in folded for name in _wake_names_for_self(room, cfg) if name)


def _mentions_self(text: str, room: RoomConfig, cfg: BridgeConfig) -> bool:
    return _wakes_self(text, room, cfg)


def _escape_gateway_command(text: str) -> str:
    if str(text or "").startswith("/"):
        return "\u200b" + text
    return text


def _is_gateway_slash_command(text: str, room: RoomConfig, cfg: BridgeConfig) -> bool:
    stripped = str(text or "").lstrip()
    if stripped.startswith("/"):
        return True
    folded = stripped.casefold()
    for name in _wake_names_for_self(room, cfg):
        if not name:
            continue
        if folded.startswith(name.casefold()):
            return stripped[len(name):].lstrip().startswith("/")
    return False


def _event_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("\x1f".join(str(p or "") for p in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _bridge_request(cfg: BridgeConfig, method: str, path: str, payload: Optional[dict[str, Any]] = None, timeout: float = 2.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if cfg.token:
        headers["Authorization"] = f"Bearer {cfg.token}"
    req = request.Request(f"{cfg.server_url}{path}", data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    if not body:
        return {}
    return json.loads(body)


def _room_for_tool(cfg: BridgeConfig, room_id: str = "") -> tuple[Optional[RoomConfig], str]:
    room_id = str(room_id or "").strip()
    if room_id:
        room = cfg.rooms.get(room_id)
        if room is not None:
            return room, ""
        return None, f"Unknown agent_bridge room_id: {room_id}"
    rooms = list(cfg.rooms.values())
    if not rooms:
        return None, "agent_bridge has no configured rooms"
    if len(rooms) > 1:
        return None, "room_id is required because agent_bridge has multiple rooms"
    return rooms[0], ""


def _available_peer_agents(room: RoomConfig, cfg: BridgeConfig) -> list[dict[str, str]]:
    peers: list[dict[str, str]] = []
    for participant in room.participants:
        agent_id = _participant_agent_id(participant)
        if not agent_id or agent_id == cfg.agent_id:
            continue
        peers.append({
            "agent_id": agent_id,
            "display_name": _participant_display_name(participant),
            "wake_name": _preferred_wake_name_for_participant(participant),
            "room_id": room.room_id,
        })
    return peers


def _format_wake_message_payload(args: dict[str, Any], cfg: BridgeConfig) -> dict[str, Any]:
    if not cfg.enabled:
        return {"success": False, "error": "agent_bridge is disabled"}

    target_agent_id = str(args.get("target_agent_id") or "").strip()
    message = str(args.get("message") or "").strip()
    if not target_agent_id:
        return {"success": False, "error": "target_agent_id is required"}
    if not message:
        return {"success": False, "error": "message is required"}

    room, error_message = _room_for_tool(cfg, str(args.get("room_id") or ""))
    if room is None:
        return {"success": False, "error": error_message}
    if target_agent_id == cfg.agent_id:
        return {"success": False, "error": "target_agent_id must be a different agent"}

    target = None
    for participant in room.participants:
        if _participant_agent_id(participant) == target_agent_id:
            target = participant
            break
    if target is None:
        return {
            "success": False,
            "error": f"Unknown target_agent_id: {target_agent_id}",
            "available_agents": _available_peer_agents(room, cfg),
        }

    wake_names = _wake_names_for_participant(target)
    preferred_wake_name = _preferred_wake_name_for_participant(target)
    if not preferred_wake_name:
        return {"success": False, "error": f"target_agent_id has no wake name: {target_agent_id}"}

    folded_message = message.casefold()
    already_wakes_target = any(name.casefold() in folded_message for name in wake_names if name)
    wake_text = message if already_wakes_target else f"{preferred_wake_name} {message}"

    return {
        "success": True,
        "wake_text": wake_text,
        "target_agent_id": target_agent_id,
        "target_display_name": _participant_display_name(target),
        "target_wake_name": preferred_wake_name,
        "room_id": room.room_id,
        "instruction": "Include wake_text verbatim in your final visible reply if you want this agent to respond.",
    }


def _agent_bridge_format_wake_message(args: dict[str, Any], **_: Any) -> str:
    cfg = _load_bridge_config()
    payload = _format_wake_message_payload(args, cfg)
    return json.dumps(payload, ensure_ascii=False)


def _agent_bridge_llm_context(cfg: BridgeConfig) -> str:
    if not cfg.enabled:
        return ""
    peers: list[dict[str, str]] = []
    for room in cfg.rooms.values():
        peers.extend(_available_peer_agents(room, cfg))
    if not peers:
        return ""

    seen: set[tuple[str, str]] = set()
    peer_lines: list[str] = []
    for peer in peers:
        key = (peer["room_id"], peer["agent_id"])
        if key in seen:
            continue
        seen.add(key)
        peer_lines.append(
            f"- {peer['display_name']} (agent_id: {peer['agent_id']}, wake: {peer['wake_name']}, room_id: {peer['room_id']})"
        )

    return "\n".join([
        "Agent bridge group-chat rule:",
        f"- You are {cfg.display_name} (agent_id: {cfg.agent_id}).",
        "- If you want another bridge agent to reply, your final visible group message must explicitly include that agent's wake @mention.",
        "- Bare names are only observed and will not wake another agent.",
        f"- When handing off, asking, or inviting another agent to answer, call `{_FORMAT_WAKE_TOOL_NAME}` and include the returned wake_text verbatim in your final reply.",
        "- Available peer agents:",
        *peer_lines,
    ])


def _on_pre_llm_call(**_: Any) -> Optional[dict[str, str]]:
    context = _agent_bridge_llm_context(_load_bridge_config())
    if not context:
        return None
    return {"context": context}


def _check_agent_bridge_available() -> bool:
    cfg = _load_bridge_config()
    return bool(cfg.enabled and cfg.rooms)


_FORMAT_WAKE_TOOL_SCHEMA = {
    "name": _FORMAT_WAKE_TOOL_NAME,
    "description": (
        "Format a visible group-chat wake message for another Hermes bridge agent. "
        "Use this when you want that agent to reply. Include the returned wake_text "
        "verbatim in your final response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_agent_id": {
                "type": "string",
                "description": "The agent_id of the bridge agent that should reply.",
            },
            "message": {
                "type": "string",
                "description": "The message to send to the target agent, without needing to manually add @.",
            },
            "room_id": {
                "type": "string",
                "description": "Optional bridge room_id. Required only when multiple bridge rooms are configured.",
            },
        },
        "required": ["target_agent_id", "message"],
    },
}


class _Runtime:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.cfg = _load_bridge_config()
        self.gateway = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.poller: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.registered = False
        self.last_error_log = 0.0
        self.seen_event_ids: set[str] = set()
        self.last_thread_by_session: dict[str, str] = {}

    def refresh(self) -> BridgeConfig:
        cfg = _load_bridge_config()
        with self.lock:
            self.cfg = cfg
        return cfg

    def maybe_start(self, gateway: Any = None) -> Optional[BridgeConfig]:
        cfg = self.refresh()
        if not cfg.enabled:
            return None
        if not cfg.agent_id:
            logger.warning("agent_bridge is enabled but agent_id is empty")
            return None
        if gateway is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            with self.lock:
                self.gateway = gateway
                if loop is not None:
                    self.loop = loop
                if self.poller is None or not self.poller.is_alive():
                    self.stop_event.clear()
                    self.poller = threading.Thread(
                        target=self._poll_loop,
                        name="hermes-agent-bridge",
                        daemon=True,
                    )
                    self.poller.start()
            self._register_best_effort(cfg)
        return cfg

    def _log_error(self, message: str, exc: Exception) -> None:
        now = time.time()
        if now - self.last_error_log > 30:
            logger.warning("%s: %s", message, exc)
            self.last_error_log = now
        else:
            logger.debug("%s: %s", message, exc)

    def _register_best_effort(self, cfg: BridgeConfig) -> None:
        payload = {
            "agent_id": cfg.agent_id,
            "display_name": cfg.display_name,
            "rooms": {rid: _room_payload(room) for rid, room in cfg.rooms.items()},
        }
        try:
            _bridge_request(cfg, "POST", "/v1/register", payload, timeout=1.5)
            self.registered = True
        except Exception as exc:
            self._log_error("agent_bridge register failed", exc)

    def _poll_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                cfg = self.cfg
                loop = self.loop
            if not cfg.enabled or loop is None:
                self.stop_event.wait(2.0)
                continue
            query = parse.urlencode({"agent_id": cfg.agent_id, "timeout": 25, "limit": 20})
            try:
                data = _bridge_request(cfg, "GET", f"/v1/events?{query}", timeout=30.0)
                for event in data.get("events") or []:
                    future = asyncio.run_coroutine_threadsafe(_handle_bridge_event(event), loop)
                    future.add_done_callback(_log_future_error)
            except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                self._log_error("agent_bridge poll failed", exc)
                self.stop_event.wait(5.0)
            except Exception as exc:
                self._log_error("agent_bridge poll failed", exc)
                self.stop_event.wait(5.0)


_RUNTIME = _Runtime()


def _log_future_error(future: Any) -> None:
    try:
        future.result()
    except Exception as exc:
        logger.warning("agent_bridge event handling failed: %s", exc)


def _current_runtime(gateway: Any = None) -> tuple[Optional[_Runtime], Optional[BridgeConfig]]:
    cfg = _RUNTIME.maybe_start(gateway)
    if cfg is None:
        return None, None
    return _RUNTIME, cfg


def _publish_event(
    *,
    cfg: BridgeConfig,
    room: RoomConfig,
    source: Any,
    author_id: str,
    author_name: str,
    author_type: str,
    text: str,
    origin_agent_id: str,
    message_id: str,
    thread_id: str = "",
) -> dict[str, Any]:
    payload = {
        "id": _event_id("bridge", room.room_id, author_id, message_id, text),
        "room_id": room.room_id,
        "room": _room_payload(room),
        "origin_agent_id": origin_agent_id,
        "author_id": author_id,
        "author_name": author_name,
        "author_type": author_type,
        "text": text,
        "platform": _platform_value(getattr(source, "platform", "")),
        "chat_id": str(getattr(source, "chat_id", "")),
        "chat_name": getattr(source, "chat_name", None),
        "chat_type": getattr(source, "chat_type", None) or "group",
        "thread_id": thread_id,
        "external_message_id": message_id,
        "timestamp": datetime.now().isoformat(),
    }
    return _bridge_request(cfg, "POST", "/v1/events", payload, timeout=1.5)


def _on_gateway_startup(gateway: Any = None, platforms: Optional[list[str]] = None, **_: Any) -> None:
    del platforms
    _current_runtime(gateway)


def _on_pre_gateway_dispatch(event: Any, gateway: Any = None, session_store: Any = None, **_: Any) -> None:
    del session_store
    runtime, cfg = _current_runtime(gateway)
    if runtime is None or cfg is None or event is None:
        return None
    source = getattr(event, "source", None)
    if source is None:
        return None
    room = _room_for_source(cfg, source)
    if room is None:
        return None

    # Real callbacks from known bridge agents are allowed to continue through
    # Hermes, but are not re-published into the bridge to avoid duplicates on
    # platforms that *do* forward bot messages.
    known_agent_ids = _known_agent_ids(room)
    source_user_id = str(getattr(source, "user_id", "") or "")
    if bool(getattr(source, "is_bot", False)) or source_user_id in known_agent_ids:
        return None

    text = str(getattr(event, "text", "") or "")
    if _is_gateway_slash_command(text, room, cfg):
        return None
    message_id = str(getattr(event, "message_id", "") or getattr(source, "message_id", "") or _event_id("human", source_user_id, text, time.time()))
    try:
        result = _publish_event(
            cfg=cfg,
            room=room,
            source=source,
            author_id=source_user_id or str(getattr(source, "user_name", "") or "human"),
            author_name=str(getattr(source, "user_name", "") or source_user_id or "human"),
            author_type="human",
            text=text,
            origin_agent_id=cfg.agent_id,
            message_id=message_id,
        )
        thread_id = str(result.get("thread_id") or "")
        if thread_id and gateway is not None:
            try:
                session_key = gateway._session_key_for_source(source)
                runtime.last_thread_by_session[session_key] = thread_id
            except Exception:
                pass
    except Exception as exc:
        runtime._log_error("agent_bridge publish inbound failed", exc)
    return None


def _on_post_gateway_response(
    *,
    event: Any = None,
    source: Any = None,
    session_key: str = "",
    session_id: str = "",
    response: str = "",
    gateway: Any = None,
    already_sent: bool = False,
    **_: Any,
) -> None:
    del session_id, already_sent
    runtime, cfg = _current_runtime(gateway)
    if runtime is None or cfg is None or source is None or not str(response or "").strip():
        return None
    room = _room_for_source(cfg, source)
    if room is None:
        return None

    raw = getattr(event, "raw_message", None) or {}
    bridge_meta = raw.get("agent_bridge") if isinstance(raw, dict) else None
    thread_id = ""
    if isinstance(bridge_meta, dict):
        thread_id = str(bridge_meta.get("thread_id") or "")
    if not thread_id and session_key:
        thread_id = runtime.last_thread_by_session.get(session_key, "")

    source_message_id = str(getattr(event, "message_id", "") or getattr(source, "message_id", "") or "")
    message_id = _event_id("agent", cfg.agent_id, source_message_id, response)
    try:
        result = _publish_event(
            cfg=cfg,
            room=room,
            source=source,
            author_id=cfg.agent_id,
            author_name=cfg.display_name,
            author_type="agent",
            text=str(response),
            origin_agent_id=cfg.agent_id,
            message_id=message_id,
            thread_id=thread_id,
        )
        returned_thread = str(result.get("thread_id") or "")
        if returned_thread and session_key:
            runtime.last_thread_by_session[session_key] = returned_thread
    except Exception as exc:
        runtime._log_error("agent_bridge publish response failed", exc)
    return None


def _source_from_payload(payload: dict[str, Any]):
    from gateway.config import Platform
    from gateway.session import SessionSource

    return SessionSource(
        platform=Platform(str(payload.get("platform") or "wecom")),
        chat_id=str(payload.get("chat_id") or ""),
        chat_name=payload.get("chat_name"),
        chat_type=str(payload.get("chat_type") or "group"),
        user_id=str(payload.get("author_id") or ""),
        user_name=str(payload.get("author_name") or payload.get("author_id") or ""),
        message_id=f"agent_bridge:{payload.get('id')}",
        is_bot=str(payload.get("author_type") or "") == "agent",
    )


def _content_for_transcript(payload: dict[str, Any], source: Any, gateway: Any) -> str:
    text = str(payload.get("text") or "")
    try:
        from gateway.session import is_shared_multi_user_session

        shared = is_shared_multi_user_session(
            source,
            group_sessions_per_user=getattr(gateway.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(gateway.config, "thread_sessions_per_user", False),
        )
    except Exception:
        shared = True
    if shared and getattr(source, "user_name", None):
        return f"[{source.user_name}] {text}"
    return text


def _append_observed_event(payload: dict[str, Any], gateway: Any) -> None:
    source = _source_from_payload(payload)
    entry = gateway.session_store.get_or_create_session(source)
    gateway.session_store.append_to_transcript(
        entry.session_id,
        {
            "role": "user",
            "content": _content_for_transcript(payload, source, gateway),
            "timestamp": datetime.now().isoformat(),
            "agent_bridge_event_id": payload.get("id"),
            "agent_bridge_observed": True,
        },
    )


async def _inject_addressed_event(payload: dict[str, Any], gateway: Any) -> None:
    from gateway.platforms.base import MessageEvent, MessageType

    source = _source_from_payload(payload)
    event = MessageEvent(
        text=_escape_gateway_command(str(payload.get("text") or "")),
        message_type=MessageType.TEXT,
        source=source,
        raw_message={"agent_bridge": payload},
        message_id=f"agent_bridge:{payload.get('id')}",
        internal=True,
    )
    adapter = gateway.adapters.get(source.platform)
    if adapter is not None and hasattr(adapter, "handle_message"):
        await adapter.handle_message(event)
        return
    response = await gateway._handle_message(event)
    if response:
        logger.warning(
            "agent_bridge produced a response for %s but no %s adapter is available to deliver it",
            source.chat_id,
            _platform_value(source.platform),
        )


async def _handle_bridge_event(payload: dict[str, Any]) -> None:
    runtime, cfg = _current_runtime()
    if runtime is None or cfg is None:
        return
    event_id = str(payload.get("id") or "")
    if not event_id:
        return
    with runtime.lock:
        if event_id in runtime.seen_event_ids:
            return
        runtime.seen_event_ids.add(event_id)
        gateway = runtime.gateway
    if gateway is None:
        return
    if str(payload.get("author_id") or "") == cfg.agent_id:
        return
    room = cfg.rooms.get(str(payload.get("room_id") or ""))
    if room is None:
        room_payload = payload.get("room") or {}
        if isinstance(room_payload, dict):
            room = RoomConfig(
                room_id=str(payload.get("room_id") or room_payload.get("room_id") or ""),
                external_targets=room_payload.get("external_targets") or [],
                participants=room_payload.get("participants") or [],
                max_bot_messages=max(1, _as_int(room_payload.get("max_bot_messages"), 16)),
                idle_timeout_seconds=max(30, _as_int(room_payload.get("idle_timeout_seconds"), 1800)),
            )
    if room is None:
        return
    should_reply = (
        str(payload.get("author_type") or "") == "agent"
        and bool(payload.get("allow_auto_reply", True))
        and _wakes_self(str(payload.get("text") or ""), room, cfg)
    )
    if should_reply:
        await _inject_addressed_event(payload, gateway)
    else:
        await asyncio.to_thread(_append_observed_event, payload, gateway)


def register(ctx) -> None:
    ctx.register_hook("gateway_startup", _on_gateway_startup)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    ctx.register_hook("post_gateway_response", _on_post_gateway_response)
    ctx.register_tool(
        name=_FORMAT_WAKE_TOOL_NAME,
        toolset=_TOOLSET_NAME,
        schema=_FORMAT_WAKE_TOOL_SCHEMA,
        handler=_agent_bridge_format_wake_message,
        check_fn=_check_agent_bridge_available,
        description="Format visible @mention messages that wake peer bridge agents.",
        emoji="@",
    )
