"""Raft channel platform adapter.

Starts a local wake endpoint, spawns ``raft agent bridge`` as a child process,
and injects content-free wake hints into Hermes' normal gateway session pipeline.
Token and port are auto-generated when not provided via env/config.
The bridge remains responsible for Raft message cursors and body materialization;
the agent uses the Raft CLI according to the Raft manual.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
import weakref
from typing import Any, Deque, Dict, List, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    merge_pending_message_event,
)
from gateway.session import build_session_key

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 0
DEFAULT_PATH = "/wake"
DEFAULT_RUNTIME_SESSION = "default"
DEFAULT_MAX_BODY_BYTES = 16_384
DEFAULT_ACTIVITY_QUEUE_CAP = 500
ACTIVITY_CONTENT_CAP = 4096
ACTIVITY_EVENT_SCHEMA = "raft-activity.v1"
ACTIVITY_DRAIN_SCHEMA = "raft-activity-drain.v1"
BRIDGE_TOKEN_HEADER = "x-raft-bridge-token"

_CONTENT_FIELD_NAMES = {
    "body",
    "content",
    "message",
    "messages",
    "preview",
    "snippet",
    "text",
}

_SAFE_SCALAR_RE = re.compile(r"^[a-zA-Z0-9._:@/ -]+$")
_MAX_SCALAR_LENGTH = 120
_ACTIVITY_ALLOWED_FIELDS = {
    "schema",
    "eventId",
    "sessionId",
    "hookEventName",
    "status",
    "occurredAt",
    "toolName",
    "toolInput",
    "toolOutput",
    "toolInputTruncated",
    "toolOutputTruncated",
    "truncated",
    "errorClass",
    "durationMs",
}
_ACTIVE_ADAPTERS: "weakref.WeakSet[RaftAdapter]" = weakref.WeakSet()
_ACTIVE_ADAPTERS_LOCK = threading.Lock()
_RAFT_CONTEXT_LOCK = threading.Lock()
_RAFT_SESSION_IDS: set[str] = set()
_RAFT_TURN_IDS: set[str] = set()
_RAFT_PROMPT_TURN_IDS: set[str] = set()


def check_raft_requirements() -> bool:
    """Check if Raft channel dependencies are available.

    Intentionally silent on failure — this is a passive probe registered as
    the platform's ``check_fn``. It is called on every
    ``load_gateway_config()`` (message handling, display lookups, agent
    turns), so logging here floods the logs for every user without the
    ``raft`` CLI installed. The caller (``gateway/platform_registry.py``
    ``create_adapter()``) emits its own warning when requirements are not met
    and an adapter is actually requested. This matches the convention used by
    other platform adapters (e.g. ``teams/adapter.py``).
    """
    if not AIOHTTP_AVAILABLE:
        return False
    if not shutil.which("raft"):
        return False
    return True


def _path_value(value: Any) -> str:
    path = str(value or DEFAULT_PATH).strip() or DEFAULT_PATH
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _has_content_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().lower() in _CONTENT_FIELD_NAMES:
                return True
            if _has_content_field(nested):
                return True
    elif isinstance(value, list):
        return any(_has_content_field(item) for item in value)
    return False


def _platform_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _safe_scalar(value: Any, default: Optional[str] = None) -> Optional[str]:
    if not isinstance(value, str):
        return default
    if not value or len(value) > _MAX_SCALAR_LENGTH:
        return default
    if not _SAFE_SCALAR_RE.match(value):
        return default
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _content_string(value: Any) -> Optional[tuple[str, bool]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return None
    if not text:
        return None
    if len(text) > ACTIVITY_CONTENT_CAP:
        return text[:ACTIVITY_CONTENT_CAP], True
    return text, False


def _duration_ms(value: Any) -> Optional[int]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    duration = int(value)
    if duration < 0:
        return None
    return duration


def _make_activity_event(
    *,
    hook_event_name: str,
    session_id: Any,
    status: str = "ok",
    tool_name: Any = None,
    tool_input: Any = None,
    tool_output: Any = None,
    error_class: Any = None,
    duration_ms: Any = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "schema": ACTIVITY_EVENT_SCHEMA,
        "eventId": f"hermes-{uuid.uuid4()}",
        "sessionId": _safe_scalar(session_id, "unknown") or "unknown",
        "hookEventName": hook_event_name,
        "status": "error" if status == "error" else "ok",
        "occurredAt": _now_iso(),
    }
    safe_tool_name = _safe_scalar(tool_name)
    if safe_tool_name:
        event["toolName"] = safe_tool_name
    safe_error_class = _safe_scalar(error_class)
    if safe_error_class:
        event["errorClass"] = safe_error_class
    safe_duration_ms = _duration_ms(duration_ms)
    if safe_duration_ms is not None:
        event["durationMs"] = safe_duration_ms

    truncated = False
    input_value = _content_string(tool_input)
    if input_value:
        event["toolInput"], input_truncated = input_value
        if input_truncated:
            event["toolInputTruncated"] = True
            truncated = True
    output_value = _content_string(tool_output)
    if output_value:
        event["toolOutput"], output_truncated = output_value
        if output_truncated:
            event["toolOutputTruncated"] = True
            truncated = True
    if truncated:
        event["truncated"] = True
    return event


def _validate_activity_event(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("activity event must be an object")
    if value.get("schema") != ACTIVITY_EVENT_SCHEMA:
        raise ValueError("unsupported activity event schema")
    unknown = set(value) - _ACTIVITY_ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"activity event field {sorted(unknown)[0]} is not allowed")
    for key in ("eventId", "sessionId", "hookEventName", "occurredAt"):
        if not _safe_scalar(value.get(key)):
            raise ValueError(f"activity event {key} must be a safe non-empty string")
    if value.get("status") not in {"ok", "error"}:
        raise ValueError("activity event status must be ok|error")
    if value.get("toolName") is not None and not _safe_scalar(value.get("toolName")):
        raise ValueError("activity event toolName must be a safe string")
    if value.get("errorClass") is not None and not _safe_scalar(value.get("errorClass")):
        raise ValueError("activity event errorClass must be a safe string")
    if value.get("durationMs") is not None and _duration_ms(value.get("durationMs")) is None:
        raise ValueError("activity event durationMs must be a non-negative number")
    for key in ("truncated", "toolInputTruncated", "toolOutputTruncated"):
        if value.get(key) is not None and not isinstance(value.get(key), bool):
            raise ValueError(f"activity event {key} must be a boolean")

    event = dict(value)
    if event.get("durationMs") is not None:
        event["durationMs"] = _duration_ms(event["durationMs"])
    for key in ("toolInput", "toolOutput"):
        content = event.get(key)
        if content is None:
            continue
        if not isinstance(content, str):
            raise ValueError(f"activity event {key} must be a string")
        if len(content) > ACTIVITY_CONTENT_CAP:
            event[key] = content[:ACTIVITY_CONTENT_CAP]
            event["truncated"] = True
            event[f"{key}Truncated"] = True
    return event


class ActivityQueue:
    """Bounded at-most-once queue for Raft external activity telemetry."""

    def __init__(self, cap: int = DEFAULT_ACTIVITY_QUEUE_CAP):
        self._cap = max(1, int(cap or DEFAULT_ACTIVITY_QUEUE_CAP))
        self._events: Deque[Dict[str, Any]] = deque()
        self._dropped_since_drain = 0
        self._lock = threading.Lock()

    def push(self, event: Dict[str, Any]) -> None:
        validated = _validate_activity_event(event)
        with self._lock:
            self._events.append(validated)
            while len(self._events) > self._cap:
                self._events.popleft()
                self._dropped_since_drain += 1

    def drain(self, max_events: int = 200) -> Dict[str, Any]:
        limit = max(1, int(max_events or 200))
        with self._lock:
            events: List[Dict[str, Any]] = []
            while self._events and len(events) < limit:
                events.append(self._events.popleft())
            dropped = self._dropped_since_drain
            self._dropped_since_drain = 0
        return {"schema": ACTIVITY_DRAIN_SCHEMA, "events": events, "dropped": dropped}

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)


def _remember_raft_context(session_id: Any, turn_id: Any = None) -> None:
    safe_session_id = _safe_scalar(session_id)
    safe_turn_id = _safe_scalar(turn_id)
    with _RAFT_CONTEXT_LOCK:
        if safe_session_id:
            _RAFT_SESSION_IDS.add(safe_session_id)
        if safe_turn_id:
            _RAFT_TURN_IDS.add(safe_turn_id)


def _forget_raft_context(session_id: Any, turn_id: Any = None, *, forget_session: bool = False) -> None:
    safe_session_id = _safe_scalar(session_id)
    safe_turn_id = _safe_scalar(turn_id)
    with _RAFT_CONTEXT_LOCK:
        if safe_turn_id:
            _RAFT_TURN_IDS.discard(safe_turn_id)
            _RAFT_PROMPT_TURN_IDS.discard(safe_turn_id)
        if forget_session and safe_session_id:
            _RAFT_SESSION_IDS.discard(safe_session_id)


def _is_raft_context(**kwargs: Any) -> bool:
    if _platform_value(kwargs.get("platform")) == "raft":
        _remember_raft_context(kwargs.get("session_id"), kwargs.get("turn_id"))
        return True
    safe_session_id = _safe_scalar(kwargs.get("session_id"))
    safe_turn_id = _safe_scalar(kwargs.get("turn_id"))
    with _RAFT_CONTEXT_LOCK:
        return bool(
            (safe_turn_id and safe_turn_id in _RAFT_TURN_IDS)
            or (safe_session_id and safe_session_id in _RAFT_SESSION_IDS)
        )


def _report_activity(event: Dict[str, Any]) -> None:
    with _ACTIVE_ADAPTERS_LOCK:
        adapters = list(_ACTIVE_ADAPTERS)
    for adapter in adapters:
        adapter.report_activity(event)


def _on_session_start(**kwargs: Any) -> None:
    if not _is_raft_context(**kwargs):
        return
    try:
        from tools.env_passthrough import register_env_passthrough

        register_env_passthrough(["RAFT_PROFILE"])
    except Exception:
        logger.debug("[raft] failed to register RAFT_PROFILE env passthrough", exc_info=True)
    _report_activity(
        _make_activity_event(
            hook_event_name="SessionStart",
            session_id=kwargs.get("session_id"),
        )
    )


def _on_pre_llm_call(**kwargs: Any) -> None:
    if not _is_raft_context(**kwargs):
        return
    safe_turn_id = _safe_scalar(kwargs.get("turn_id"))
    if safe_turn_id:
        with _RAFT_CONTEXT_LOCK:
            if safe_turn_id in _RAFT_PROMPT_TURN_IDS:
                return
            _RAFT_PROMPT_TURN_IDS.add(safe_turn_id)
    _report_activity(
        _make_activity_event(
            hook_event_name="UserPromptSubmit",
            session_id=kwargs.get("session_id"),
        )
    )


def _on_pre_tool_call(**kwargs: Any) -> None:
    if not _is_raft_context(**kwargs):
        return
    _report_activity(
        _make_activity_event(
            hook_event_name="PreToolUse",
            session_id=kwargs.get("session_id"),
            tool_name=kwargs.get("tool_name"),
            tool_input=kwargs.get("args"),
        )
    )


def _on_post_tool_call(**kwargs: Any) -> None:
    if not _is_raft_context(**kwargs):
        return
    status = "error" if kwargs.get("status") in {"error", "blocked"} or kwargs.get("error_type") else "ok"
    hook_name = "PostToolUseFailure" if status == "error" else "PostToolUse"
    _report_activity(
        _make_activity_event(
            hook_event_name=hook_name,
            session_id=kwargs.get("session_id"),
            status=status,
            tool_name=kwargs.get("tool_name"),
            tool_input=kwargs.get("args"),
            tool_output=kwargs.get("error_message") or kwargs.get("result"),
            error_class=kwargs.get("error_type") or ("tool_failure" if status == "error" else None),
            duration_ms=kwargs.get("duration_ms"),
        )
    )


def _on_post_llm_call(**kwargs: Any) -> None:
    if not _is_raft_context(**kwargs):
        return
    _report_activity(
        _make_activity_event(
            hook_event_name="Stop",
            session_id=kwargs.get("session_id"),
        )
    )


def _on_session_end(**kwargs: Any) -> None:
    if not _is_raft_context(**kwargs):
        return
    if kwargs.get("interrupted") or kwargs.get("completed") is False:
        _report_activity(
            _make_activity_event(
                hook_event_name="Stop",
                session_id=kwargs.get("session_id"),
                status="error",
                error_class="interrupted" if kwargs.get("interrupted") else "incomplete",
            )
        )
    _forget_raft_context(kwargs.get("session_id"), kwargs.get("turn_id"))


def _on_session_finalize(**kwargs: Any) -> None:
    if not _is_raft_context(**kwargs):
        return
    _report_activity(
        _make_activity_event(
            hook_event_name="SessionEnd",
            session_id=kwargs.get("session_id"),
        )
    )
    _forget_raft_context(kwargs.get("session_id"), kwargs.get("turn_id"), forget_session=True)


class RaftAdapter(BasePlatformAdapter):
    """Local HTTP endpoint for Raft channel bridge delivery."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("raft"))
        extra = config.extra or {}
        self._host: str = str(extra.get("host", DEFAULT_HOST))
        self._port: int = int(extra.get("port", DEFAULT_PORT))
        self._path: str = _path_value(extra.get("path", DEFAULT_PATH))
        self._bridge_token: str = str(extra.get("bridge_token", ""))
        self._runtime_session: str = str(
            extra.get("runtime_session", DEFAULT_RUNTIME_SESSION)
            or DEFAULT_RUNTIME_SESSION
        )
        self._max_body_bytes: int = int(
            extra.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES)
        )
        self._runner = None
        self._bridge_process: Optional[subprocess.Popen] = None
        self._activity_queue = ActivityQueue()

    @property
    def runtime_session(self) -> str:
        return self._runtime_session

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self._bridge_token:
            self._bridge_token = secrets.token_hex(32)
            logger.info("[raft] Auto-generated bridge token")

        # client_max_size makes aiohttp enforce the cap on every read path,
        # including Transfer-Encoding: chunked bodies that carry no
        # Content-Length and would otherwise bypass the header checks below
        # (mirrors gateway/platforms/webhook.py's connect()).
        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._path, self._handle_wake)
        app.router.add_post("/activity", self._handle_activity)
        app.router.add_get("/activity/drain", self._handle_activity_drain)

        if self._port != 0:
            import socket as _socket

            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    sock.connect(("127.0.0.1", self._port))
                logger.error(
                    "[raft] Port %d already in use. Set platforms.raft.extra.port in config",
                    self._port,
                )
                return False
            except (ConnectionRefusedError, OSError):
                logger.debug("Suppressed exception", exc_info=True)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

        bound_port = self._port
        if bound_port == 0 and site._server and site._server.sockets:
            bound_port = site._server.sockets[0].getsockname()[1]

        self._mark_connected()
        with _ACTIVE_ADAPTERS_LOCK:
            _ACTIVE_ADAPTERS.add(self)
        logger.info("[raft] Raft channel listening on %s:%d%s", self._host, bound_port, self._path)

        self._spawn_bridge(bound_port)
        return True

    async def disconnect(self) -> None:
        self._stop_bridge()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        with _ACTIVE_ADAPTERS_LOCK:
            _ACTIVE_ADAPTERS.discard(self)
        self._mark_disconnected()
        logger.info("[raft] Disconnected")

    def _spawn_bridge(self, port: int) -> None:
        raft_bin = shutil.which("raft")
        if not raft_bin:
            logger.warning("[raft] raft CLI not found in PATH; bridge not spawned — wake-only polling mode")
            return

        profile = os.environ.get("RAFT_PROFILE", "")
        if not profile:
            logger.warning("[raft] RAFT_PROFILE not set; bridge not spawned")
            return

        endpoint = f"http://{self._host}:{port}{self._path}"
        cmd: List[str] = [
            raft_bin, "--profile", profile,
            "agent", "bridge",
            "--wake-adapter", "wake-channel",
            "--wake-channel-endpoint", endpoint,
        ]
        env = {**os.environ, "RAFT_CHANNEL_TOKEN": self._bridge_token}
        try:
            self._bridge_process = subprocess.Popen(
                cmd, env=env, stdin=subprocess.DEVNULL
            )
            logger.info("[raft] Spawned bridge pid=%d profile=%s endpoint=%s", self._bridge_process.pid, profile, endpoint)
        except Exception:
            logger.exception("[raft] Failed to spawn bridge")

    def _stop_bridge(self) -> None:
        proc = self._bridge_process
        if proc is None:
            return
        self._bridge_process = None
        try:
            proc.terminate()
            proc.wait(timeout=5)
            logger.info("[raft] Bridge process terminated (pid=%d)", proc.pid)
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.warning("[raft] Bridge process killed after timeout (pid=%d)", proc.pid)
        except Exception:
            logger.exception("[raft] Error stopping bridge")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        logger.debug("[raft] adapter send is a no-op; agent delivers via raft CLI")
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"raft/{chat_id}", "type": "raft"}

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "platform": "raft",
                "runtimeSession": self._runtime_session,
                "activity": {
                    "queueSize": self._activity_queue.size,
                    "endpoint": "/activity",
                    "drainEndpoint": "/activity/drain",
                },
            }
        )

    async def _handle_wake(self, request: "web.Request") -> "web.Response":
        if not self._validate_bridge_token(request.headers.get(BRIDGE_TOKEN_HEADER, "")):
            return web.json_response({"ok": False, "error": "unauthorized"}, status=401)

        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"ok": False, "error": "payload_too_large"}, status=413)

        try:
            raw_body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            # aiohttp's client_max_size tripped — chunked or lying
            # Content-Length. Same 413 as the header check above.
            return web.json_response({"ok": False, "error": "payload_too_large"}, status=413)
        except Exception:
            return web.json_response({"ok": False, "error": "bad_request"}, status=400)
        if len(raw_body) > self._max_body_bytes:
            # Defense in depth: enforce the cap on the actual bytes read even
            # if the server-level limit was bypassed or misconfigured.
            return web.json_response({"ok": False, "error": "payload_too_large"}, status=413)

        payload: Dict[str, Any] = {}
        if raw_body.strip():
            try:
                parsed = json.loads(raw_body)
            except json.JSONDecodeError:
                return web.json_response({"ok": False, "error": "invalid_json"}, status=400)
            if not isinstance(parsed, dict):
                return web.json_response({"ok": False, "error": "invalid_payload"}, status=400)
            payload = parsed

        # Do not gate on payload["schema"]: the bridge owns schema evolution;
        # Hermes only verifies that wake hints are content-free.
        if _has_content_field(payload):
            return web.json_response({"ok": False, "error": "content_not_allowed"}, status=400)

        accepted = await self._accept_wake(payload)
        if not accepted:
            return web.json_response(
                {
                    "ok": False,
                    "error": "not_ready",
                    "runtimeSession": self._runtime_session,
                },
                status=503,
            )

        return web.json_response(
            {
                "ok": True,
                "runtimeSession": self._runtime_session,
            },
            status=202,
        )

    async def _handle_activity(self, request: "web.Request") -> "web.Response":
        if not self._validate_bridge_token(request.headers.get(BRIDGE_TOKEN_HEADER, "")):
            return web.json_response({"ok": False, "error": "unauthorized"}, status=401)

        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"ok": False, "error": "payload_too_large"}, status=413)

        try:
            raw_text = await request.text()
        except web.HTTPRequestEntityTooLarge:
            # aiohttp's client_max_size tripped — chunked or lying
            # Content-Length. Same 413 as the header check above.
            return web.json_response({"ok": False, "error": "payload_too_large"}, status=413)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        if len(raw_text.encode("utf-8")) > self._max_body_bytes:
            # Defense in depth: enforce the cap on the actual bytes read even
            # if the server-level limit was bypassed or misconfigured.
            return web.json_response({"ok": False, "error": "payload_too_large"}, status=413)

        try:
            payload = json.loads(raw_text)
            self._activity_queue.push(payload)
        except json.JSONDecodeError:
            return web.json_response({"ok": False, "error": "invalid_json"}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

        return web.json_response({"ok": True}, status=202)

    async def _handle_activity_drain(self, request: "web.Request") -> "web.Response":
        if not self._validate_bridge_token(request.headers.get(BRIDGE_TOKEN_HEADER, "")):
            return web.json_response({"ok": False, "error": "unauthorized"}, status=401)
        try:
            max_events = int(request.query.get("max", "200"))
        except ValueError:
            max_events = 200
        return web.json_response(self._activity_queue.drain(max_events))

    def _validate_bridge_token(self, token: str) -> bool:
        if not self._bridge_token or not token:
            return False
        return hmac.compare_digest(token, self._bridge_token)

    async def _accept_wake(self, payload: Dict[str, Any]) -> bool:
        if not self._message_handler:
            logger.warning("[raft] Wake received before gateway message handler was attached")
            return False

        delivery_id = str(
            payload.get("eventId")
            or payload.get("attemptId")
            or payload.get("messageId")
            or payload.get("delivery_id")
            or payload.get("wake_id")
            or payload.get("id")
            or f"raft-wake-{int(time.time() * 1000)}"
        )
        source = self.build_source(
            chat_id=self._runtime_session,
            chat_name="Raft channel",
            chat_type="dm",
            user_id="raft-bridge",
            user_name="Raft Bridge",
        )
        event = MessageEvent(
            text=self._wake_prompt(),
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=delivery_id,
            internal=True,
        )
        try:
            await self.handle_message(event)
        except Exception:
            logger.exception("[raft] Failed to inject wake event")
            return False
        return True

    async def handle_message(self, event: MessageEvent) -> None:
        """Accept Raft wake hints without interrupting an active Hermes turn."""
        if not self._message_handler:
            return

        session_key = build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

        if session_key in self._active_sessions:
            logger.debug("[raft] Wake queued for busy session %s", session_key)
            merge_pending_message_event(self._pending_messages, session_key, event)
            return

        await super().handle_message(event)

    @staticmethod
    def _wake_prompt() -> str:
        return (
            "Raft wake hint received. New Raft messages may be pending. "
            "If you have not read the Raft manual in this session, run "
            "`raft manual get raft-cli-overview` before using Raft commands."
        )

    def report_activity(self, event: Dict[str, Any]) -> None:
        try:
            self._activity_queue.push(event)
        except Exception:
            logger.debug("[raft] activity event dropped during validation", exc_info=True)


def _is_connected(config: PlatformConfig) -> bool:
    extra = config.extra or {}
    return bool(extra.get("enabled") or extra.get("bridge_token"))


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env vars during gateway config load.

    Auto-enables when RAFT_PROFILE is set (the adapter needs it anyway).
    """
    if not os.getenv("RAFT_PROFILE"):
        return None

    return {"enabled": True}


def interactive_setup() -> None:
    """Interactive ``hermes gateway setup`` flow for the Raft platform.

    Lazy-imports CLI helpers so the plugin stays importable in gateway runtime
    and test contexts. The flow persists ``RAFT_PROFILE`` to the Hermes env
    file so the Raft adapter auto-enables after a gateway restart.
    """
    from hermes_cli.cli_output import (
        print_header,
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_yes_no,
    )
    from hermes_cli.config import get_env_value, save_env_value

    print_header("Raft")
    existing_profile = get_env_value("RAFT_PROFILE")
    if existing_profile:
        print_info(f"Raft: already configured (profile: {existing_profile})")
        if not prompt_yes_no("Reconfigure Raft?", False):
            print_info(f"Keeping RAFT_PROFILE={existing_profile}.")
            return

    print_info("Connect Hermes to Raft as an external agent.")
    print_info("Create the External Agent in Raft first, then run:")
    print_info("  raft agent login --server <server-url> --agent <agent-id> --profile-slug <slug>")
    print()

    profile = prompt("Raft profile slug", default=existing_profile or "")
    if not profile:
        print_warning("Raft profile slug is required; skipping Raft setup")
        return

    save_env_value("RAFT_PROFILE", profile.strip())

    print()
    print_success("Raft configuration saved")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="raft",
        label="Raft",
        adapter_factory=lambda cfg: RaftAdapter(cfg),
        check_fn=check_raft_requirements,
        is_connected=_is_connected,
        required_env=["RAFT_PROFILE"],
        install_hint="Install the Raft CLI from https://raft.build",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        emoji="🔔",
        platform_hint=(
            "You are connected to Raft via an external-agent channel. "
            "Run `raft --profile {profile} profile show` to confirm which agent profile is active. "
            "Run `raft --profile {profile} manual get raft-cli-overview` to learn available Raft commands. "
            "Always pass `--profile {profile}` to every raft CLI call."
        ).format(profile=os.environ.get("RAFT_PROFILE", "your-agent-profile")),
    )
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
