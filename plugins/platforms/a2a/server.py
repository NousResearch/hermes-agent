"""Official A2A 1.0 Starlette routes with a bounded authenticated perimeter."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from . import auth, config

try:
    from a2a.auth.user import User as _A2AUser
except ImportError:
    _A2AUser = object

RPC_PATH = "/a2a"
CARD_PATH = "/.well-known/agent-card.json"
UVICORN_TRANSPORT_GUIDANCE = (
    "Uvicorn owns socket/header parsing: configure --timeout-keep-alive and enforce "
    "header/read deadlines at the reverse proxy; the app additionally bounds ASGI body receive time."
)
_BLOCKED_METHODS = {
    "SendStreamingMessage",
    "SubscribeToTask",
    "CreateTaskPushNotificationConfig",
    "GetTaskPushNotificationConfig",
    "ListTaskPushNotificationConfigs",
    "DeleteTaskPushNotificationConfig",
    "GetExtendedAgentCard",
}
_SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store"),
)


class _SanitizingSDKLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = "A2A protocol processing event"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def _install_sdk_log_filter() -> None:
    names = {
        "a2a.server.routes.jsonrpc_dispatcher",
        "a2a.server.tasks.database_task_store",
    }
    names.update(
        name
        for name in logging.Logger.manager.loggerDict
        if name == "a2a.server" or name.startswith("a2a.server.")
    )
    for name in names:
        logger = logging.getLogger(name)
        if not any(isinstance(item, _SanitizingSDKLogFilter) for item in logger.filters):
            logger.addFilter(_SanitizingSDKLogFilter())


@dataclass(frozen=True)
class ServerLimits:
    max_body_bytes: int = 1_048_576
    max_header_bytes: int = 32_768
    max_response_bytes: int = 2_097_152
    body_receive_timeout_seconds: float = 15.0
    request_timeout_seconds: float = 120.0
    ip_requests_per_minute: int = 120
    principal_requests_per_minute: int = 60
    preauth_concurrency: int = 32
    auth_concurrency: int = 4
    principal_concurrency: int = 2
    limiter_max_keys: int = 4_096

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("A2A server limits must be positive")


@dataclass(frozen=True)
class ResolvedPrincipal:
    name: str
    profile: str
    credential_ref: str


class AuthenticatedA2AUser(_A2AUser):
    def __init__(self, user_name: str):
        self._user_name = user_name

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self._user_name


class A2AAuthError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _header_values(raw_headers: list[tuple[bytes, bytes]], name: bytes) -> list[str]:
    return [value.decode("latin-1") for key, value in raw_headers if key.lower() == name]


class BearerServerCallContextBuilder:
    """Resolve singleton raw headers through the A2A-only credential domain."""

    def __init__(self, target_profile: str):
        self.target_profile = config.validate_name(target_profile, label="target profile")

    def parse_raw_headers(self, raw_headers: list[tuple[bytes, bytes]]) -> str:
        authorizations = _header_values(raw_headers, b"authorization")
        versions = _header_values(raw_headers, b"a2a-version")
        if len(authorizations) != 1 or len(versions) != 1:
            raise A2AAuthError(400, "Exactly one Authorization and A2A-Version header is required")
        scheme, separator, token = authorizations[0].partition(" ")
        if scheme.lower() != "bearer" or not separator or not token.strip():
            raise A2AAuthError(401, "Bearer authentication required")
        return token.strip()

    def authenticate_token(self, token: str) -> ResolvedPrincipal:
        credential_ref = auth.resolve_inbound_token(token)
        if credential_ref is None:
            raise A2AAuthError(401, "Invalid bearer credential")
        matches = [
            ResolvedPrincipal(name=name, profile=entry.get("profile", ""), credential_ref=credential_ref)
            for name, entry in config.load_a2a_settings().principals.items()
            if entry.get("credential_ref") == credential_ref
        ]
        if len(matches) != 1:
            raise A2AAuthError(403, "Credential is not assigned to one principal")
        principal = matches[0]
        if principal.profile != self.target_profile:
            raise A2AAuthError(403, "Principal is not authorized for this profile")
        return principal

    def authenticate_raw_headers(self, raw_headers: list[tuple[bytes, bytes]]) -> ResolvedPrincipal:
        return self.authenticate_token(self.parse_raw_headers(raw_headers))

    def build(self, request):
        from a2a.extensions.common import HTTP_EXTENSION_HEADER, get_requested_extensions
        from a2a.server.context import ServerCallContext

        raw_headers = list(request.scope.get("headers", []))
        principal = request.scope.get("a2a_principal")
        if not isinstance(principal, ResolvedPrincipal):
            try:
                principal = self.authenticate_raw_headers(raw_headers)
            except A2AAuthError as exc:
                from starlette.exceptions import HTTPException

                challenge = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
                raise HTTPException(exc.status_code, str(exc), headers=challenge) from None
        version = _header_values(raw_headers, b"a2a-version")[0]
        extensions = _header_values(raw_headers, HTTP_EXTENSION_HEADER.lower().encode())
        return ServerCallContext(
            user=AuthenticatedA2AUser(principal.name),
            state={
                "headers": {"a2a-version": version},
                "principal": principal.name,
                "profile": principal.profile,
            },
            requested_extensions=get_requested_extensions(extensions),
        )


class _SlidingWindowLimiter:
    def __init__(self, *, max_keys: int = 4_096, ttl_seconds: float = 60.0):
        self.max_keys = max_keys
        self.ttl_seconds = ttl_seconds
        self._events: OrderedDict[str, deque[float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._events)

    def _evict(self, now: float) -> None:
        expired = [key for key, events in self._events.items() if not events or events[-1] <= now - self.ttl_seconds]
        for key in expired:
            self._events.pop(key, None)
        while len(self._events) > self.max_keys:
            self._events.popitem(last=False)

    def allow(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        self._evict(now)
        events = self._events.pop(key, deque())
        while events and events[0] <= now - self.ttl_seconds:
            events.popleft()
        allowed = len(events) < limit
        if allowed:
            events.append(now)
        self._events[key] = events
        while len(self._events) > self.max_keys:
            self._events.popitem(last=False)
        return allowed


class _SanitizingRequestHandler:
    def __init__(self, delegate: Any):
        self._delegate = delegate

    def __getattr__(self, name: str):
        target = getattr(self._delegate, name)

        async def sanitized(*args, **kwargs):
            try:
                return await target(*args, **kwargs)
            except Exception as exc:
                try:
                    from a2a.utils.errors import A2AError
                except ImportError:
                    A2AError = ()
                if isinstance(exc, A2AError):
                    raise
                raise RuntimeError("A2A request failed") from None

        return sanitized


class _ResponseCaptureError(RuntimeError):
    pass


async def _respond(send, status: int, payload: dict[str, Any], *, challenge: bool = False) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *_SECURITY_HEADERS]
    if challenge:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class _SecurityMiddleware:
    def __init__(self, app, *, context_builder: BearerServerCallContextBuilder, limits: ServerLimits, task_store):
        self.app = app
        self.context_builder = context_builder
        self.limits = limits
        self.task_store = task_store
        self.ip_limiter = _SlidingWindowLimiter(max_keys=limits.limiter_max_keys)
        self.principal_limiter = _SlidingWindowLimiter(max_keys=limits.limiter_max_keys)
        self._auth_semaphore = asyncio.Semaphore(limits.auth_concurrency)
        self._active: dict[str, int] = {}
        self._active_lock = asyncio.Lock()
        self.preauth_active = 0
        self._accepting = True

    def stop_accepting(self) -> None:
        """Reject new HTTP ingress while allowing lifespan shutdown to run."""
        self._accepting = False

    async def __call__(self, scope, receive, send):  # noqa: C901, PLR0911, PLR0912
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if not self._accepting:
            await _respond(send, 503, {"error": "Server is shutting down"})
            return
        raw_headers = list(scope.get("headers", []))
        client = scope.get("client") or ("unknown", 0)
        if not self.ip_limiter.allow(str(client[0]), self.limits.ip_requests_per_minute):
            await _respond(send, 429, {"error": "Rate limit exceeded"})
            return
        if self.preauth_active >= self.limits.preauth_concurrency:
            await _respond(send, 503, {"error": "Authentication capacity exceeded"})
            return
        self.preauth_active += 1
        preauth_held = True

        def release_admission() -> None:
            nonlocal preauth_held
            if preauth_held:
                self.preauth_active -= 1
                preauth_held = False

        try:
            if sum(len(key) + len(value) for key, value in raw_headers) > self.limits.max_header_bytes:
                release_admission()
                await _respond(send, 431, {"error": "Request headers too large"})
                return
            path = scope.get("path")
            method = str(scope.get("method", "")).upper()
            if path == CARD_PATH:
                if method not in {"GET", "HEAD"}:
                    release_admission()
                    await _respond(send, 405, {"error": "Method not allowed"})
                    return
                try:
                    messages = await asyncio.wait_for(
                        self._capture(scope, self._replay(b"")),
                        timeout=self.limits.request_timeout_seconds,
                    )
                except TimeoutError:
                    release_admission()
                    await _respond(send, 504, {"error": "Request timed out"})
                    return
                except _ResponseCaptureError:
                    release_admission()
                    await _respond(send, 502, {"error": "Upstream response too large"})
                    return
                release_admission()
                await self._transmit(messages, send)
                return
            if path != RPC_PATH:
                release_admission()
                await _respond(send, 404, {"error": "Not found"})
                return
            if method != "POST":
                release_admission()
                await _respond(send, 405, {"error": "Method not allowed"})
                return
            try:
                token = self.context_builder.parse_raw_headers(raw_headers)
            except A2AAuthError as exc:
                release_admission()
                await _respond(send, exc.status_code, {"error": str(exc)}, challenge=exc.status_code == 401)
                return
            try:
                body = await self._read_body(receive)
            except TimeoutError:
                release_admission()
                await _respond(send, 408, {"error": "Request body timed out"})
                return
            except _ResponseCaptureError:
                release_admission()
                await _respond(send, 413, {"error": "Request body too large"})
                return
            if body is None:
                return
            if auth._parse_inbound_token(token) is None:
                release_admission()
                await _respond(send, 401, {"error": "Invalid bearer credential"}, challenge=True)
                return
            try:
                async with self._auth_semaphore:
                    principal = await asyncio.to_thread(self.context_builder.authenticate_token, token)
            except A2AAuthError as exc:
                release_admission()
                await _respond(send, exc.status_code, {"error": str(exc)}, challenge=exc.status_code == 401)
                return
            release_admission()
            await self._serve_authenticated(scope, body, principal, send)
        finally:
            release_admission()

    async def _read_body(self, receive) -> bytes | None:
        body = bytearray()
        deadline = asyncio.get_running_loop().time() + self.limits.body_receive_timeout_seconds
        more = True
        while more:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError
            message = await asyncio.wait_for(receive(), timeout=remaining)
            if message["type"] == "http.disconnect":
                return None
            body.extend(message.get("body", b""))
            if len(body) > self.limits.max_body_bytes:
                raise _ResponseCaptureError
            more = message.get("more_body", False)
        return bytes(body)

    async def _serve_authenticated(self, scope, body: bytes, principal: ResolvedPrincipal, send) -> None:
        if not self.principal_limiter.allow(principal.name, self.limits.principal_requests_per_minute):
            await _respond(send, 429, {"error": "Rate limit exceeded"})
            return
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, dict) and decoded.get("method") in _BLOCKED_METHODS:
            await _respond(send, 200, {"jsonrpc": "2.0", "id": decoded.get("id"), "error": {"code": -32601, "message": "Method not found"}})
            return
        async with self._active_lock:
            active = self._active.get(principal.name, 0)
            if active >= self.limits.principal_concurrency:
                await _respond(send, 429, {"error": "Concurrency limit exceeded"})
                return
            self._active[principal.name] = active + 1
        scope["a2a_principal"] = principal
        try:
            try:
                messages = await asyncio.wait_for(
                    self._capture(scope, self._replay(body)),
                    timeout=self.limits.request_timeout_seconds,
                )
            except TimeoutError:
                await _respond(send, 504, {"error": "Request timed out"})
                return
            except _ResponseCaptureError:
                await _respond(send, 502, {"error": "Upstream response too large"})
                return
        finally:
            async with self._active_lock:
                remaining = self._active.get(principal.name, 1) - 1
                if remaining:
                    self._active[principal.name] = remaining
                else:
                    self._active.pop(principal.name, None)
        await self._transmit(messages, send)

    @staticmethod
    def _replay(body: bytes):
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        return receive

    async def _capture(self, scope, receive) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        captured_bytes = 0
        starts = 0

        async def capture(message):
            nonlocal captured_bytes, starts
            captured_bytes += 32
            if message["type"] == "http.response.start":
                starts += 1
                if starts > 1:
                    raise _ResponseCaptureError("multiple response starts")
                captured_bytes += sum(
                    len(key) + len(value) for key, value in message.get("headers", [])
                )
            elif message["type"] == "http.response.body":
                captured_bytes += len(message.get("body", b""))
            if captured_bytes > self.limits.max_response_bytes:
                raise _ResponseCaptureError("response too large")
            messages.append(message)

        await self.app(scope, receive, capture)
        if starts != 1:
            raise _ResponseCaptureError("missing response start")
        self._sanitize_jsonrpc_error(messages)
        return messages

    async def _transmit(self, messages: list[dict[str, Any]], send) -> None:
        for message in messages:
            if message["type"] == "http.response.start":
                existing = {key.lower() for key, _value in message.get("headers", [])}
                message["headers"] = list(message.get("headers", [])) + [header for header in _SECURITY_HEADERS if header[0] not in existing]
            await send(message)

    @staticmethod
    def _sanitize_jsonrpc_error(messages: list[dict[str, Any]]) -> None:
        bodies = [message for message in messages if message["type"] == "http.response.body"]
        if len(bodies) != 1:
            return
        try:
            payload = json.loads(bodies[0].get("body", b""))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return
        error.pop("data", None)
        if error.get("code") == -32603:
            error["message"] = "Internal error"
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        bodies[0]["body"] = encoded
        for message in messages:
            if message["type"] == "http.response.start":
                message["headers"] = [(key, str(len(encoded)).encode()) if key.lower() == b"content-length" else (key, value) for key, value in message.get("headers", [])]


def build_agent_card(public_url: str):
    from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, HTTPAuthSecurityScheme, SecurityScheme

    card = AgentCard(
        name="Hermes Agent",
        description="Authenticated Hermes agent-to-agent interface",
        version="1.0",
        supported_interfaces=[AgentInterface(url=public_url, protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False, extended_agent_card=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="text", name="Text request", description="Process a plain-text request", tags=["text"], input_modes=["text/plain"], output_modes=["text/plain"])],
    )
    card.security_schemes["bearer"].CopyFrom(SecurityScheme(http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="bearer", bearer_format="A2A opaque token")))
    card.security_requirements.add().schemes["bearer"].list.extend([])
    return card


def _current_profile_name() -> str:
    from hermes_cli.profiles import get_active_profile_name

    return get_active_profile_name() or "default"


def create_a2a_app(
    request_handler: Any,
    *,
    target_profile: str | None = None,
    production: bool = True,
    limits: ServerLimits | None = None,
    task_store_instance=None,
    agent_card=None,
):
    """Create official routes and own task-store startup/reconciliation lifespan."""
    try:
        from a2a.server.routes.agent_card_routes import create_agent_card_routes
        from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
        from starlette.applications import Starlette
    except ImportError as exc:
        raise RuntimeError("A2A server requires hermes-agent[a2a]") from exc
    from . import task_store as task_store_module

    active_profile = config.validate_name(_current_profile_name(), label="active profile")
    if target_profile is not None and target_profile != active_profile:
        raise ValueError("target profile does not match the active profile")
    public_url = config.configured_public_url(production=production)
    existing_store = getattr(request_handler, "task_store", None)
    existing_card = getattr(request_handler, "_agent_card", None)
    if (
        task_store_instance is not None
        and existing_store is not None
        and existing_store is not task_store_instance
    ):
        raise ValueError("request handler must use the same A2A task store instance")
    if agent_card is not None and existing_card is not None and existing_card is not agent_card:
        raise ValueError("request handler must use the same A2A agent card instance")

    # Validate every supplied/existing identity before allocating defaults so
    # mismatch failures cannot leak a newly-created SQLite store.
    store = task_store_instance or existing_store
    if store is None:
        store = task_store_module.create_task_store()
    card = agent_card or existing_card
    if card is None:
        card = build_agent_card(public_url)
    request_handler.task_store = store
    request_handler._agent_card = card
    _install_sdk_log_filter()
    context_builder = BearerServerCallContextBuilder(active_profile)

    @asynccontextmanager
    async def lifespan(_app):
        await store.initialize()
        await task_store_module.reconcile_orphaned_tasks(store)
        try:
            yield
        finally:
            await store.close()

    routes = create_agent_card_routes(card)
    routes += create_jsonrpc_routes(_SanitizingRequestHandler(request_handler), RPC_PATH, context_builder=context_builder, enable_v0_3_compat=False)
    inner = Starlette(routes=routes, lifespan=lifespan)
    return _SecurityMiddleware(inner, context_builder=context_builder, limits=limits or ServerLimits(), task_store=store)
