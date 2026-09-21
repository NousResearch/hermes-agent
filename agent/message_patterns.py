"""
Request/Reply Message Patterns for Hermes-Agent.

Extends the existing Pub-Sub EventBus with RPC-style Request/Reply support.
All patterns are fully backward-compatible with the existing Pub-Sub model.

Patterns provided:
    - Pub-Sub  (original): fire-and-forget event emission to multiple handlers
    - Request-Reply: RPC-style call with correlated response and timeout

Design principles (Module Independence MI):
    - Does NOT modify the existing EventBus class
    - Composes with EventBus rather than inheriting from it
    - All locking is self-contained in each ReplyChannel
    - Optional EventBus integration: publish lifecycle events without hard dependency

Usage:

    from agent.message_patterns import RequestReplyBus, RR

    bus = RequestReplyBus()

    # --- Pub-Sub (unchanged from EventBus) ---
    def handler(event):
        print(f"Got: {event.type}")

    bus.subscribe("greeting", handler)
    bus.emit_event("greeting", {"text": "hello"})

    # --- Request/Reply ---
    def handle_request(request: RpcRequest) -> RpcResponse:
        return RR.ok(request.correlation_id, {"result": f"echo: {request.payload}"})

    bus.register_handler("echo", handle_request)

    response = bus.call("echo", {"msg": "hello"}, timeout=5.0)
    print(response.payload)  # {"result": "echo: hello"}

    # --- Timeout ---
    try:
        bus.call("unknown", {}, timeout=0.1)
    except RpcTimeoutError as e:
        print(f"Timed out: {e.correlation_id}")

    # --- Error reply ---
    def handle_bad(request):
        return RR.error(request.correlation_id, "Invalid input", code=400)

    bus.register_handler("bad", handle_bad)
    response = bus.call("bad", {})
    print(response.is_error)  # True
    print(response.error_code)  # 400
"""

from __future__ import annotations

import threading
import uuid
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.hermes.analytics import EventBus

logger = logging.getLogger(__name__)


# ─── Reply Convenience ────────────────────────────────────────────────────────


class RR:
    """
    Static helpers to construct RpcResponse objects.

    Usage::
        return RR.ok(correlation_id, {"key": "value"})
        return RR.error(correlation_id, "Something went wrong", code=500)
    """

    @staticmethod
    def ok(correlation_id: str, payload: Any) -> "RpcResponse":
        return RpcResponse(
            correlation_id=correlation_id,
            is_error=False,
            error_message=None,
            error_code=None,
            payload=payload,
        )

    @staticmethod
    def error(
        correlation_id: str,
        message: str,
        code: int | None = None,
        payload: Any = None,
    ) -> "RpcResponse":
        return RpcResponse(
            correlation_id=correlation_id,
            is_error=True,
            error_message=message,
            error_code=code,
            payload=payload,
        )


# ─── RPC Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class RpcRequest:
    """
    An incoming RPC request.

    Attributes:
        handler_name: Identifier for the handler to dispatch to.
        payload:      Arbitrary argument dict passed to the handler.
        correlation_id: Unique ID used to correlate the reply.
        timestamp:     When the request was created.
        session_id:    Session context (optional).
    """

    handler_name: str
    payload: Any
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: str = ""


@dataclass
class RpcResponse:
    """
    A response to an RpcRequest.

    Attributes:
        correlation_id:  Matches the originating RpcRequest.
        is_error:        True if the handler raised an exception or returned an error.
        error_message:   Human-readable error description (set when is_error=True).
        error_code:      Optional integer error code (e.g., HTTP-style).
        payload:         The actual return value from the handler.
    """

    correlation_id: str
    is_error: bool = False
    error_message: Optional[str] = None
    error_code: Optional[int] = None
    payload: Any = None


# ─── ReplyChannel ─────────────────────────────────────────────────────────────


class ReplyChannel:
    """
    A one-shot channel that delivers exactly one RpcResponse (or a timeout error).

    Thread-safe. Awaits a single ``send_reply()`` call then auto-closes.
    Multiple ``wait()`` calls on the same channel all receive the same response.

    Usage::

        ch = ReplyChannel(timeout=5.0)
        # ... hand ch.correlation_id to the callee ...
        response = ch.wait()   # blocks until reply or timeout
    """

    DEFAULT_TIMEOUT: float = 30.0

    def __init__(
        self,
        correlation_id: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.correlation_id: str = correlation_id or str(uuid.uuid4())
        self._timeout: float = (
            timeout if timeout is not None else self.DEFAULT_TIMEOUT
        )
        self._response: Optional[RpcResponse] = None
        self._ready: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()

    # ── Internal (producer side) ─────────────────────────────────────────────

    def send_reply(self, response: RpcResponse) -> bool:
        """
        Deliver a response to all waiters.

        Idempotent: only the first call has effect; subsequent calls are ignored.

        Args:
            response: The RpcResponse to deliver.

        Returns:
            True if this was the first call (reply delivered), False otherwise.
        """
        with self._lock:
            if self._ready.is_set():
                return False
            self._response = response
            self._ready.set()
            return True

    # ── Consumer side ────────────────────────────────────────────────────────

    def wait(self) -> RpcResponse:
        """
        Block until a reply is delivered or the timeout fires.

        Returns:
            The RpcResponse sent via ``send_reply()``.

        Raises:
            RpcTimeoutError: If no reply arrives within the configured timeout.
        """
        if not self._ready.wait(timeout=self._timeout):
            raise RpcTimeoutError(self.correlation_id, self._timeout)
        return self._response  # type: ignore[return-value]

    def wait_if_ready(self) -> RpcResponse | None:
        """
        Non-blocking check.

        Returns:
            The RpcResponse if already set, otherwise None.
        """
        if self._ready.wait(timeout=0):
            return self._response
        return None

    @property
    def is_closed(self) -> bool:
        """True once a reply has been delivered (or timeout occurred)."""
        return self._ready.is_set()


class RpcTimeoutError(Exception):
    """
    Raised when an RPC request receives no reply within the configured timeout.

    Attributes:
        correlation_id: The ID of the timed-out request.
        timeout:        The configured timeout in seconds.
    """

    def __init__(self, correlation_id: str, timeout: float) -> None:
        self.correlation_id = correlation_id
        self.timeout = timeout
        super().__init__(
            f"RPC request {correlation_id!r} timed out after {timeout}s"
        )


# ─── RequestReplyBus ──────────────────────────────────────────────────────────


_RPC_REQUEST_EVENT = "rpc.request"
_RPC_REPLY_EVENT = "rpc.reply"


class RequestReplyBus:
    """
    Composes Pub-Sub (via an injected EventBus) with RPC-style Request/Reply.

    The ``call()`` method sends an RpcRequest and waits for a matching
    RpcResponse.  Handlers are registered per handler_name and receive an
    ``RpcRequest``, returning an ``RpcResponse`` (convenience: use ``RR``).

    Backward compatibility:
        All original EventBus.subscribe/emit methods delegate directly to the
        wrapped EventBus, so existing Pub-Sub subscribers continue to work
        unchanged.

    Thread safety:
        Uses a dedicated lock for handler registry + per-ReplyChannel locks.
        Concurrent ``call()`` invocations are fully isolated.

    Example::

        bus = RequestReplyBus(event_bus=my_event_bus)

        # Existing Pub-Sub subscriber — unchanged
        bus.subscribe("session.start", my_handler)
        bus.emit_event("session.start", {"sid": "s1"})

        # New RPC handler
        def ping(req: RpcRequest) -> RpcResponse:
            return RR.ok(req.correlation_id, {"pong": True})

        bus.register_handler("ping", ping)

        # RPC call
        resp = bus.call("ping", {}, timeout=5.0)
        assert not resp.is_error
    """

    def __init__(
        self,
        event_bus: Optional["EventBus"] = None,
        default_timeout: float = 30.0,
    ) -> None:
        self._event_bus: Optional["EventBus"] = event_bus
        self._default_timeout: float = default_timeout
        self._handlers: Dict[str, Callable[[RpcRequest], RpcResponse]] = {}
        self._handler_lock: threading.Lock = threading.Lock()

        if self._event_bus is not None:
            self._event_bus.subscribe(_RPC_REQUEST_EVENT, self._on_rpc_request)

    # ── Pub-Sub delegation (backward compatibility) ──────────────────────────

    def subscribe(
        self,
        event_type: str,
        handler: Callable[..., None],
    ) -> None:
        """
        Subscribe to a Pub-Sub event type (delegates to EventBus).

        This method is identical in signature and behaviour to EventBus.subscribe,
        preserving full backward compatibility.
        """
        if self._event_bus is None:
            logger.debug("No EventBus configured; subscribe to %s is a no-op", event_type)
            return
        self._event_bus.subscribe(event_type, handler)

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[..., None],
    ) -> None:
        """Unsubscribe from a Pub-Sub event type."""
        if self._event_bus is None:
            return
        self._event_bus.unsubscribe(event_type, handler)

    def emit(self, event: Any) -> None:
        """
        Emit a Pub-Sub event to all subscribed handlers.

        This method is identical in behaviour to EventBus.emit, preserving
        full backward compatibility.
        """
        if self._event_bus is None:
            logger.debug("No EventBus configured; emit is a no-op")
            return
        self._event_bus.emit(event)

    def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        session_id: str = "",
    ) -> None:
        """
        Emit a Pub-Sub event by type and payload (convenience overload).

        Mirrors EventBus.emit_event signature exactly.
        """
        if self._event_bus is None:
            logger.debug("No EventBus configured; emit_event is a no-op")
            return
        self._event_bus.emit_event(event_type, payload, session_id)

    # ── Handler registration ─────────────────────────────────────────────────

    def register_handler(
        self,
        handler_name: str,
        handler: Callable[[RpcRequest], RpcResponse],
    ) -> None:
        """
        Register a handler for RPC requests.

        Args:
            handler_name: Unique identifier for this handler (used in ``call()``).
            handler:     Callable that receives an RpcRequest and returns an
                         RpcResponse.  May raise exceptions — these are caught
                         and returned as error responses automatically.
        """
        with self._handler_lock:
            self._handlers[handler_name] = handler
        logger.debug("RPC handler registered: %s", handler_name)

    def unregister_handler(self, handler_name: str) -> None:
        """Remove a previously registered RPC handler."""
        with self._handler_lock:
            self._handlers.pop(handler_name, None)
        logger.debug("RPC handler unregistered: %s", handler_name)

    # ── RPC call / reply dispatch ────────────────────────────────────────────

    def call(
        self,
        handler_name: str,
        payload: Any,
        timeout: float | None = None,
        session_id: str = "",
    ) -> RpcResponse:
        """
        Send an RPC request and wait for the response.

        Args:
            handler_name: Handler to dispatch to.
            payload:      Arbitrary argument passed to the handler.
            timeout:      Max seconds to wait.  Defaults to ``default_timeout``.
            session_id:   Session context propagated to the RpcRequest.

        Returns:
            The RpcResponse returned by the handler.

        Raises:
            RpcTimeoutError: If no response arrives within ``timeout``.
            RpcNoHandlerError: If no handler is registered for ``handler_name``.
        """
        timeout = timeout if timeout is not None else self._default_timeout
        channel = ReplyChannel(timeout=timeout)

        request = RpcRequest(
            handler_name=handler_name,
            payload=payload,
            correlation_id=channel.correlation_id,
            session_id=session_id,
        )

        self._push_channel(channel)
        self._emit_rpc_request(request)

        response = channel.wait()
        if response.is_error and response.error_code == RpcNoHandlerError.CODE:
            raise RpcNoHandlerError(handler_name)
        return response

    def _emit_rpc_request(self, request: RpcRequest) -> None:
        """
        Publish an ``rpc.request`` event so all handlers can inspect it.

        Publishes via EventBus (if configured) so analytics/subscribers can
        observe RPC traffic, and dispatches directly to registered handlers.
        """
        if self._event_bus is not None:
            self._event_bus.emit_event(
                _RPC_REQUEST_EVENT,
                {
                    "handler_name": request.handler_name,
                    "payload": request.payload,
                    "correlation_id": request.correlation_id,
                    "session_id": request.session_id,
                },
            )

        # Direct dispatch to registered handler (runs in caller's thread)
        self._dispatch_to_handler(request)

    def _dispatch_to_handler(self, request: RpcRequest) -> None:
        """
        Find the handler for ``request.handler_name`` and invoke it.

        Errors raised by the handler are caught and returned as error responses.
        """
        handler: Optional[Callable[[RpcRequest], RpcResponse]]
        with self._handler_lock:
            handler = self._handlers.get(request.handler_name)

        if handler is None:
            response = RpcResponse(
                correlation_id=request.correlation_id,
                is_error=True,
                error_message=f"No handler registered for {request.handler_name!r}",
                error_code=RpcNoHandlerError.CODE,
            )
            self._deliver_reply(request.correlation_id, response)
            return

        try:
            response = handler(request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RPC handler %s raised: %s", request.handler_name, exc)
            response = RpcResponse(
                correlation_id=request.correlation_id,
                is_error=True,
                error_message=str(exc),
                error_code=500,
            )

        self._deliver_reply(request.correlation_id, response)

    def _on_rpc_request(self, event: Any) -> None:
        """
        Handle an ``rpc.request`` event published by another RequestReplyBus.

        Extracts the payload and re-dispatches locally.  This enables
        distributed RPC when multiple RequestReplyBus instances share the
        same EventBus (events are cross-process if the EventBus supports it).
        """
        try:
            payload = event.payload if hasattr(event, "payload") else dict(event)
            request = RpcRequest(
                handler_name=payload.get("handler_name", ""),
                payload=payload.get("payload"),
                correlation_id=payload.get("correlation_id", ""),
                session_id=payload.get("session_id", ""),
            )
            self._dispatch_to_handler(request)
        except Exception as exc:
            logger.warning("_on_rpc_request handling failed: %s", exc)

    # ─── Reply delivery (called by handler or remote bus) ────────────────────

    def reply(
        self,
        correlation_id: str,
        response: RpcResponse,
    ) -> bool:
        """
        Deliver an RpcResponse to the matching ReplyChannel.

        This method is exposed for advanced use-cases where a reply originates
        outside the normal ``call()`` flow (e.g., a separate thread).

        Args:
            correlation_id: The correlation ID to match against a waiting call.
            response:       The RpcResponse to deliver.

        Returns:
            True if a channel was found and the reply was delivered.
        """
        return self._deliver_reply(correlation_id, response)

    # ─── Internal routing table (session-scoped) ─────────────────────────────

    def _deliver_reply(
        self,
        correlation_id: str,
        response: RpcResponse,
    ) -> bool:
        """
        Route a reply to its ReplyChannel using the correlation ID.

        In the current single-process implementation, channels are held in
        ``_channels`` and this method finds the matching one.

        Subclasses or cross-process transports may override this to use
        different routing (e.g., a message queue).
        """
        channel = self._pop_channel(correlation_id)
        if channel is None:
            logger.debug("No pending channel for correlation_id=%s", correlation_id)
            return False
        channel.send_reply(response)
        return True

    # ─── Per-instance channel registry ────────────────────────────────────────

    def _channels(self) -> Dict[str, ReplyChannel]:
        """Return the channel registry (lazily created)."""
        if not hasattr(self, "_channel_map"):
            self._channel_map: Dict[str, ReplyChannel] = {}
        return self._channel_map

    def _push_channel(self, channel: ReplyChannel) -> None:
        """Register a channel so it can be found by correlation_id later."""
        self._channels()[channel.correlation_id] = channel

    def _pop_channel(self, correlation_id: str) -> ReplyChannel | None:
        """Remove and return a channel by correlation_id."""
        return self._channels().pop(correlation_id, None)


class RpcNoHandlerError(Exception):
    """
    Raised by ``RequestReplyBus.call()`` when no handler is registered.

    Attributes:
        handler_name: The name of the missing handler.
        CODE:         Fixed error code (404) for machine-readable use.
    """

    CODE: int = 404

    def __init__(self, handler_name: str) -> None:
        self.handler_name = handler_name
        super().__init__(f"No handler registered for {handler_name!r}")
