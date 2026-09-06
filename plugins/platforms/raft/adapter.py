"""Raft channel platform adapter.

Runs a local wake endpoint, spawns ``raft agent bridge`` as a child process, and injects
content-free wake hints into the normal gateway session pipeline (token/port auto-generated
if unset). The bridge owns Raft message cursors/bodies; the agent uses the Raft CLI.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import functools
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path as _Path
from typing import Any, Deque, Dict, List, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult, merge_pending_message_event
from gateway.session import build_session_key
from gateway.platforms._shared import coerce_port, profile_scoped as _profile_scoped

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
_WAKE_ID_KEYS = ("eventId", "attemptId", "messageId", "delivery_id", "wake_id", "id")
_WAKE_PROMPT = (
    "Raft wake hint received. New Raft messages may be pending. "
    "If you have not read the Raft manual in this session, run "
    "`raft manual get raft-cli-overview` before using Raft commands.")

_CONTENT_FIELD_NAMES = {"body", "content", "message", "messages", "preview", "snippet", "text"}
_SAFE_SCALAR_RE = re.compile(r"^[a-zA-Z0-9._:@/ -]+$")
_MAX_SCALAR_LENGTH = 120
_ACTIVITY_ALLOWED_FIELDS = set(
    "schema eventId sessionId hookEventName status occurredAt toolName toolInput toolOutput "
    "toolInputTruncated toolOutputTruncated truncated errorClass durationMs".split())
_ACTIVE_ADAPTERS: "weakref.WeakSet[RaftAdapter]" = weakref.WeakSet()
_ACTIVE_ADAPTERS_LOCK = threading.Lock()
_RAFT_CONTEXT_LOCK = threading.Lock()
_RAFT_SESSION_IDS: set[str] = set()
_RAFT_TURN_IDS: set[str] = set()
_RAFT_PROMPT_TURN_IDS: set[str] = set()


def _resolve_raft_profile() -> str:
    """Scope-aware ``RAFT_PROFILE``: a secondary multiplex profile configures Raft only via its own ``.env``
    (secret scope) — ``os.environ`` would return the DEFAULT profile's value. Unscoped ``get_secret()`` raises."""
    if _profile_scoped():
        try:
            from agent.secret_scope import get_secret
            return (get_secret("RAFT_PROFILE") or "").strip()
        except Exception:
            return ""
    return os.environ.get("RAFT_PROFILE", "").strip()


def check_raft_requirements() -> bool:
    """Passive ``check_fn`` probe: intentionally silent — it runs on every
    ``load_gateway_config()``; ``create_adapter()`` warns when an adapter is requested."""
    return bool(AIOHTTP_AVAILABLE and shutil.which("raft"))


def _has_content_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(k).strip().lower() in _CONTENT_FIELD_NAMES or _has_content_field(v) for k, v in value.items())
    return isinstance(value, list) and any(_has_content_field(item) for item in value)


def _safe_scalar(value: Any, default: Optional[str] = None) -> Optional[str]:
    ok = isinstance(value, str) and 0 < len(value) <= _MAX_SCALAR_LENGTH and _SAFE_SCALAR_RE.match(value)
    return value if ok else default


def _content_string(value: Any) -> Optional[tuple[str, bool]]:
    if value is None:
        return None
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return None
    return (text[:ACTIVITY_CONTENT_CAP], len(text) > ACTIVITY_CONTENT_CAP) if text else None


def _duration_ms(value: Any) -> Optional[int]:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or int(value) < 0:
        return None
    return int(value)


def _make_activity_event(*, hook_event_name: str, session_id: Any, status: str = "ok", tool_name: Any = None,
                         tool_input: Any = None, tool_output: Any = None, error_class: Any = None,
                         duration_ms: Any = None) -> Dict[str, Any]:
    event: Dict[str, Any] = {"schema": ACTIVITY_EVENT_SCHEMA, "eventId": f"hermes-{uuid.uuid4()}",
                             "sessionId": _safe_scalar(session_id, "unknown") or "unknown",
                             "hookEventName": hook_event_name, "status": "error" if status == "error" else "ok",
                             "occurredAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    for key, raw in (("toolName", tool_name), ("errorClass", error_class)):
        if safe := _safe_scalar(raw):
            event[key] = safe
    if (safe_duration_ms := _duration_ms(duration_ms)) is not None:
        event["durationMs"] = safe_duration_ms
    for key, raw in (("toolInput", tool_input), ("toolOutput", tool_output)):
        if content := _content_string(raw):
            event[key], was_truncated = content
            if was_truncated:
                event[f"{key}Truncated"] = event["truncated"] = True
    return event


_OPTIONAL_FIELD_RULES = (  # checked in this order; first failure wins
    (("toolName", "errorClass"), _safe_scalar, "a safe string"),
    (("durationMs",), lambda v: _duration_ms(v) is not None, "a non-negative number"),
    (("truncated", "toolInputTruncated", "toolOutputTruncated"), lambda v: isinstance(v, bool), "a boolean"))


def _validate_activity_event(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("activity event must be an object")
    if value.get("schema") != ACTIVITY_EVENT_SCHEMA:
        raise ValueError("unsupported activity event schema")
    if unknown := set(value) - _ACTIVITY_ALLOWED_FIELDS:
        raise ValueError(f"activity event field {sorted(unknown)[0]} is not allowed")
    for key in ("eventId", "sessionId", "hookEventName", "occurredAt"):
        if not _safe_scalar(value.get(key)):
            raise ValueError(f"activity event {key} must be a safe non-empty string")
    if value.get("status") not in {"ok", "error"}:
        raise ValueError("activity event status must be ok|error")
    for keys, ok, what in _OPTIONAL_FIELD_RULES:  # optional fields: None passes, anything else must satisfy ``ok``
        for key in keys:
            if value.get(key) is not None and not ok(value.get(key)):
                raise ValueError(f"activity event {key} must be {what}")
    event = dict(value)
    if event.get("durationMs") is not None:
        event["durationMs"] = _duration_ms(event["durationMs"])
    for key in ("toolInput", "toolOutput"):
        content = event.get(key)
        if content is not None and not isinstance(content, str):
            raise ValueError(f"activity event {key} must be a string")
        if content is not None and len(content) > ACTIVITY_CONTENT_CAP:
            event[key] = content[:ACTIVITY_CONTENT_CAP]
            event["truncated"] = event[f"{key}Truncated"] = True
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
            events = [self._events.popleft() for _ in range(min(limit, len(self._events)))]
            dropped, self._dropped_since_drain = self._dropped_since_drain, 0
        return {"schema": ACTIVITY_DRAIN_SCHEMA, "events": events, "dropped": dropped}

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)


def _forget_raft_context(session_id: Any, turn_id: Any = None, *, forget_session: bool = False) -> None:
    safe_session_id, safe_turn_id = _safe_scalar(session_id), _safe_scalar(turn_id)
    with _RAFT_CONTEXT_LOCK:
        if safe_turn_id:
            _RAFT_TURN_IDS.discard(safe_turn_id)
            _RAFT_PROMPT_TURN_IDS.discard(safe_turn_id)
        if forget_session and safe_session_id:
            _RAFT_SESSION_IDS.discard(safe_session_id)


def _is_raft_context(**kwargs: Any) -> bool:
    """True for Raft hook payloads; an explicit platform="raft" also learns the session/turn ids."""
    platform = kwargs.get("platform")
    safe_session_id, safe_turn_id = _safe_scalar(kwargs.get("session_id")), _safe_scalar(kwargs.get("turn_id"))
    with _RAFT_CONTEXT_LOCK:
        if str(getattr(platform, "value", platform) or "") == "raft":
            if safe_session_id:
                _RAFT_SESSION_IDS.add(safe_session_id)
            if safe_turn_id:
                _RAFT_TURN_IDS.add(safe_turn_id)
            return True
        return bool((safe_turn_id and safe_turn_id in _RAFT_TURN_IDS)
                    or (safe_session_id and safe_session_id in _RAFT_SESSION_IDS))


def _emit(hook_event_name: str, kwargs: Dict[str, Any], **fields: Any) -> None:
    """Build an activity event for the hook's session and fan it out to every live adapter."""
    event = _make_activity_event(hook_event_name=hook_event_name, session_id=kwargs.get("session_id"), **fields)
    with _ACTIVE_ADAPTERS_LOCK:
        adapters = list(_ACTIVE_ADAPTERS)
    for adapter in adapters:
        adapter.report_activity(event)


def _raft_hook(fn):
    """Run the hook body only for Raft sessions."""
    @functools.wraps(fn)
    def wrapper(**kwargs: Any) -> None:
        if _is_raft_context(**kwargs):
            fn(**kwargs)
    return wrapper


@_raft_hook
def _on_session_start(**kwargs: Any) -> None:
    try:
        from tools.env_passthrough import register_env_passthrough
        register_env_passthrough(["RAFT_PROFILE"])
    except Exception:
        logger.debug("[raft] failed to register RAFT_PROFILE env passthrough", exc_info=True)
    _emit("SessionStart", kwargs)


@_raft_hook
def _on_pre_llm_call(**kwargs: Any) -> None:
    if safe_turn_id := _safe_scalar(kwargs.get("turn_id")):
        with _RAFT_CONTEXT_LOCK:
            if safe_turn_id in _RAFT_PROMPT_TURN_IDS:
                return
            _RAFT_PROMPT_TURN_IDS.add(safe_turn_id)
    _emit("UserPromptSubmit", kwargs)


@_raft_hook
def _on_pre_tool_call(**kwargs: Any) -> None:
    _emit("PreToolUse", kwargs, tool_name=kwargs.get("tool_name"), tool_input=kwargs.get("args"))


@_raft_hook
def _on_post_tool_call(**kwargs: Any) -> None:
    status = "error" if kwargs.get("status") in {"error", "blocked"} or kwargs.get("error_type") else "ok"
    _emit("PostToolUseFailure" if status == "error" else "PostToolUse", kwargs, status=status,
          tool_name=kwargs.get("tool_name"), tool_input=kwargs.get("args"),
          tool_output=kwargs.get("error_message") or kwargs.get("result"),
          error_class=kwargs.get("error_type") or ("tool_failure" if status == "error" else None),
          duration_ms=kwargs.get("duration_ms"))


@_raft_hook
def _on_post_llm_call(**kwargs: Any) -> None:
    _emit("Stop", kwargs)


@_raft_hook
def _on_session_end(**kwargs: Any) -> None:
    if kwargs.get("interrupted") or kwargs.get("completed") is False:
        _emit("Stop", kwargs, status="error", error_class="interrupted" if kwargs.get("interrupted") else "incomplete")
    _forget_raft_context(kwargs.get("session_id"), kwargs.get("turn_id"))


@_raft_hook
def _on_session_finalize(**kwargs: Any) -> None:
    _emit("SessionEnd", kwargs)
    _forget_raft_context(kwargs.get("session_id"), kwargs.get("turn_id"), forget_session=True)


def _error_response(error: str, status: int) -> "web.Response":
    return web.json_response({"ok": False, "error": error}, status=status)


class RaftAdapter(BasePlatformAdapter):
    """Local HTTP endpoint for Raft channel bridge delivery."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("raft"))
        extra = config.extra or {}
        self._host: str = str(extra.get("host", DEFAULT_HOST))
        self._port: int = int(extra.get("port", DEFAULT_PORT))
        path = str(extra.get("path", DEFAULT_PATH) or DEFAULT_PATH).strip() or DEFAULT_PATH
        self._path: str = path if path.startswith("/") else f"/{path}"
        self._bridge_token: str = str(extra.get("bridge_token", ""))
        self._runtime_session: str = str(extra.get("runtime_session", DEFAULT_RUNTIME_SESSION) or DEFAULT_RUNTIME_SESSION)
        self._max_body_bytes: int = int(extra.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))
        self._runner = None
        self._bridge_process: Optional[subprocess.Popen] = None
        self._activity_queue = ActivityQueue()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self._bridge_token:
            self._bridge_token = secrets.token_hex(32)
            logger.info("[raft] Auto-generated bridge token")
        # client_max_size makes aiohttp enforce the cap on every read path, including
        # chunked bodies with no Content-Length (mirrors gateway/platforms/webhook.py).
        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post(self._path, self._handle_wake)
        app.router.add_post("/activity", self._handle_activity)
        app.router.add_get("/activity/drain", self._handle_activity_drain)
        if self._port:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    sock.connect(("127.0.0.1", self._port))
                logger.error("[raft] Port %d already in use. Set platforms.raft.extra.port in config", self._port)
                return False
            except (ConnectionRefusedError, OSError):
                pass
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
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
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
        if not (raft_bin := shutil.which("raft")):
            logger.warning("[raft] raft CLI not found in PATH; bridge not spawned — wake-only polling mode")
            return
        if not (profile := _resolve_raft_profile()):
            logger.warning("[raft] RAFT_PROFILE not set; bridge not spawned")
            return
        endpoint = f"http://{self._host}:{port}{self._path}"
        cmd: List[str] = [raft_bin, "--profile", profile, "agent", "bridge", "--wake-adapter", "wake-channel",
                          "--wake-channel-endpoint", endpoint]
        try:
            self._bridge_process = subprocess.Popen(
                cmd, env={**os.environ, "RAFT_CHANNEL_TOKEN": self._bridge_token}, stdin=subprocess.DEVNULL)
            logger.info("[raft] Spawned bridge pid=%d profile=%s endpoint=%s", self._bridge_process.pid, profile, endpoint)
        except Exception:
            logger.exception("[raft] Failed to spawn bridge")

    def _stop_bridge(self) -> None:
        proc, self._bridge_process = self._bridge_process, None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
            logger.info("[raft] Bridge process terminated (pid=%d)", proc.pid)
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.warning("[raft] Bridge process killed after timeout (pid=%d)", proc.pid)
        except Exception:
            logger.exception("[raft] Error stopping bridge")

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        logger.debug("[raft] adapter send is a no-op; agent delivers via raft CLI")
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"raft/{chat_id}", "type": "raft"}

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        activity = {"queueSize": self._activity_queue.size, "endpoint": "/activity", "drainEndpoint": "/activity/drain"}
        return web.json_response(
            {"status": "ok", "platform": "raft", "runtimeSession": self._runtime_session, "activity": activity})

    def _authorized(self, request: "web.Request") -> bool:
        token = request.headers.get(BRIDGE_TOKEN_HEADER, "")
        # Compare as bytes: compare_digest raises TypeError on a non-ASCII str header.
        return bool(self._bridge_token and token) and hmac.compare_digest(token.encode(), self._bridge_token.encode())

    async def _read_bridge_body(self, request: "web.Request", *, text: bool) -> tuple[Any, Optional["web.Response"]]:
        """Auth + size-capped body read for wake/activity -> ``(body, None)`` or ``(None, error)``.
        ``text=True``: ``request.text()``, utf-8 length check, exception text in the 400 body; else raw bytes."""
        if not self._authorized(request):
            return None, _error_response("unauthorized", 401)
        if (request.content_length or 0) > self._max_body_bytes:
            return None, _error_response("payload_too_large", 413)
        try:
            body = await (request.text() if text else request.read())
        except web.HTTPRequestEntityTooLarge:
            # client_max_size tripped — chunked or lying Content-Length. Same 413 as above.
            return None, _error_response("payload_too_large", 413)
        except Exception as exc:
            return None, _error_response(str(exc) if text else "bad_request", 400)
        # Defense in depth: cap the actual bytes read even if the server-level limit was bypassed.
        if (len(body.encode("utf-8")) if text else len(body)) > self._max_body_bytes:
            return None, _error_response("payload_too_large", 413)
        return body, None

    async def _handle_wake(self, request: "web.Request") -> "web.Response":
        raw_body, error = await self._read_bridge_body(request, text=False)
        if error is not None:
            return error
        try:
            payload = json.loads(raw_body) if raw_body.strip() else {}
        except json.JSONDecodeError:
            return _error_response("invalid_json", 400)
        if not isinstance(payload, dict):
            return _error_response("invalid_payload", 400)
        # No payload["schema"] gate: the bridge owns schema evolution; Hermes only checks content-free.
        if _has_content_field(payload):
            return _error_response("content_not_allowed", 400)
        not_ready = {"ok": False, "error": "not_ready", "runtimeSession": self._runtime_session}
        if not self._message_handler:
            logger.warning("[raft] Wake received before gateway message handler was attached")
            return web.json_response(not_ready, status=503)
        delivery_id = str(
            next((payload.get(k) for k in _WAKE_ID_KEYS if payload.get(k)), None)
            or f"raft-wake-{int(time.time() * 1000)}")
        source = self.build_source(chat_id=self._runtime_session, chat_name="Raft channel", chat_type="dm",
                                   user_id="raft-bridge", user_name="Raft Bridge")
        event = MessageEvent(text=_WAKE_PROMPT, message_type=MessageType.TEXT, source=source,
                             raw_message=payload, message_id=delivery_id, internal=True)
        try:
            await self.handle_message(event)
        except Exception:
            logger.exception("[raft] Failed to inject wake event")
            return web.json_response(not_ready, status=503)
        return web.json_response({"ok": True, "runtimeSession": self._runtime_session}, status=202)

    async def _handle_activity(self, request: "web.Request") -> "web.Response":
        raw_text, error = await self._read_bridge_body(request, text=True)
        if error is not None:
            return error
        try:
            self._activity_queue.push(json.loads(raw_text))
        except json.JSONDecodeError:
            return _error_response("invalid_json", 400)
        except Exception as exc:
            return _error_response(str(exc), 400)
        return web.json_response({"ok": True}, status=202)

    async def _handle_activity_drain(self, request: "web.Request") -> "web.Response":
        if not self._authorized(request):
            return _error_response("unauthorized", 401)
        max_events = coerce_port(request.query.get("max", "200"), 200)  # int-or-default
        return web.json_response(self._activity_queue.drain(max_events))

    async def handle_message(self, event: MessageEvent) -> None:
        """Accept Raft wake hints without interrupting an active Hermes turn."""
        if not self._message_handler:
            return
        session_key = build_session_key(
            event.source, group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            profile=self._session_key_profile(event.source))
        if session_key in self._active_sessions:
            logger.debug("[raft] Wake queued for busy session %s", session_key)
            merge_pending_message_event(self._pending_messages, session_key, event)
            return
        await super().handle_message(event)

    def report_activity(self, event: Dict[str, Any]) -> None:
        try:
            self._activity_queue.push(event)
        except Exception:
            logger.debug("[raft] activity event dropped during validation", exc_info=True)


def _is_connected(config: PlatformConfig) -> bool:
    extra = config.extra or {}
    return bool(extra.get("enabled") or extra.get("bridge_token"))


def _env_enablement() -> Optional[dict]:
    """Auto-enable during gateway config load when the scope-aware RAFT_PROFILE is set.

    Auto-enables when RAFT_PROFILE is set (the adapter needs it anyway). Scope-aware: consults the active
    profile's own RAFT_PROFILE (env, or a secondary profile's own .env via the secret scope) instead of the
    default profile's bridged env value (mirrors the Buzz/SimpleX fix for 98738) — see
    ``_resolve_raft_profile``. See #98738.
    """
    return {"enabled": True} if _resolve_raft_profile() else None


def interactive_setup() -> None:
    """``hermes gateway setup`` flow: persists ``RAFT_PROFILE`` to the Hermes env file.
    CLI helpers are lazy-imported so the plugin stays importable in gateway runtime and tests."""
    from hermes_cli.cli_output import print_header, print_info, print_success, print_warning, prompt, prompt_yes_no
    from hermes_cli.config import get_env_value, save_env_value
    print_header("Raft")
    existing_profile = get_env_value("RAFT_PROFILE")
    if existing_profile:
        print_info(f"Raft: already configured (profile: {existing_profile})")
        if not prompt_yes_no("Reconfigure Raft?", False):
            print_info(f"Keeping RAFT_PROFILE={existing_profile}.")
            return
    for line in ("Connect Hermes to Raft as an external agent.", "Create the External Agent in Raft first, then run:",
                 "  raft agent login --server <server-url> --agent <agent-id> --profile-slug <slug>"):
        print_info(line)
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
        adapter_factory=RaftAdapter,
        check_fn=check_raft_requirements,
        is_connected=_is_connected,
        required_env=["RAFT_PROFILE"],
        install_hint="Install the Raft CLI from https://raft.build",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        emoji="🔔",
        # Scope-aware: register() runs inside _profile_runtime_scope for a secondary multiplex
        # profile, so this resolves that profile's own RAFT_PROFILE (see _resolve_raft_profile).
        platform_hint=(
            "You are connected to Raft via an external-agent channel. "
            "Run `raft --profile {profile} profile show` to confirm which agent profile is active. "
            "Run `raft --profile {profile} manual get raft-cli-overview` to learn available Raft commands. "
            "Always pass `--profile {profile}` to every raft CLI call."
        ).format(profile=_resolve_raft_profile() or "your-agent-profile"))
    for hook_name, callback in (("on_session_start", _on_session_start), ("pre_llm_call", _on_pre_llm_call),
                                ("pre_tool_call", _on_pre_tool_call), ("post_tool_call", _on_post_tool_call),
                                ("post_llm_call", _on_post_llm_call), ("on_session_end", _on_session_end),
                                ("on_session_finalize", _on_session_finalize)):
        ctx.register_hook(hook_name, callback)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import asyncio  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
