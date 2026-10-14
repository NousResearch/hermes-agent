"""One gateway connection per classic-CLI process (CALLMAP §3.1, §4).

A daemon thread owns an asyncio loop and one ``GatewayClient``. It never calls a HermesCLI
method and never blocks: every event frame goes into a thread-safe ``queue.Queue`` that the
agent thread drains (``GatewayAgent.run_conversation``), and ``session.info``-class frames
also update the plain-attribute ``SessionState`` caches the status bar reads.

Callers on any other thread use ``call()``, which blocks on the loop's future; calling it on
the loop thread itself is a deadlock and raises.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading

from hermes_cli.gateway_client import DEFAULT_RPC_TIMEOUT, GatewayClientError

logger = logging.getLogger(__name__)

DISCONNECTED = ("Gateway disconnected; turn outcome is unknown. Resume the printed session ID to check "
                "whether it completed or was interrupted; do not resend the work.")


class SessionState:
    """What the gateway last told this viewer about one session (plain attributes, GIL-atomic
    reads from the prompt_toolkit render thread; never an RPC per paint)."""

    def __init__(self, session_id):
        self.session_id = session_id
        self.info: dict = {}
        self.title: str | None = None
        self.todos: list = []
        self.revision = None
        self.subscription_id = None
        self.execution_generation = None
        self.messages: list = []
        self.usage: dict = {}
        # Called (loop thread) with each ``session.usage`` payload; GatewayAgent maps it onto main's
        # agent attributes, so a frame landing between turns is never dropped with the turn queue.
        self.usage_listeners: list = []
        # A gateway-admitted notification turn the idle CLI queued a header for (gateway_tools_notify).
        self.admitted_turn: dict | None = None

    def adopt_snapshot(self, snapshot):
        self.info = dict(snapshot.get("info") or {})
        self.revision = snapshot.get("revision", self.revision)
        self.subscription_id = snapshot.get("subscription_id", self.subscription_id)
        self.execution_generation = snapshot.get("execution_generation", self.execution_generation)
        self.messages = list(snapshot.get("messages") or [])
        if self.info.get("title"):
            self.title = self.info["title"]

    def apply_event(self, kind, payload):
        if kind in ("session.info", "session.updated") and isinstance(payload, dict):
            self.info.update({k: v for k, v in payload.items() if k != "session_id"})
            if payload.get("title"):
                self.title = payload["title"]
        elif kind == "todo.updated" and isinstance(payload, dict):
            self.todos = list(payload.get("todos") or [])
        elif kind == "session.usage" and isinstance(payload, dict):
            self.apply_usage(payload.get("usage") if isinstance(payload.get("usage"), dict) else payload)

    def apply_usage(self, usage):
        """One session-cumulative usage dict (``session.usage`` payload or ``session.stats`` result)."""
        if not isinstance(usage, dict):
            return
        self.usage = dict(usage)
        for listener in list(self.usage_listeners):
            listener(self.usage)


class GatewayConnection:
    _shared = None
    _shared_lock = threading.Lock()

    def __init__(self, connect=None):
        self._connect = connect
        self.events: queue.Queue = queue.Queue()
        self.states: dict[str, SessionState] = {}
        self.client = None
        self.loop = None
        self._thread = None
        self._ready = threading.Event()
        self._error = None
        self._stop = None
        self._start_lock = threading.Lock()
        self.closed = False
        # /bg and /btw answers by task id (hermes_cli/gateway_side.py); created by whichever of the
        # waiter or the frame comes first, so an answer faster than the verb's reply is kept.
        self._side: dict[str, queue.Queue] = {}
        self._side_lock = threading.Lock()

    @classmethod
    def shared(cls):
        with cls._shared_lock:
            if cls._shared is None or cls._shared.closed:
                cls._shared = cls()
            return cls._shared

    def state(self, session_id) -> SessionState:
        state = self.states.get(session_id)
        if state is None:
            state = self.states[session_id] = SessionState(session_id)
        return state

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self, timeout=90):
        with self._start_lock:
            if self._thread is not None:
                if self._error is not None:
                    raise self._error
                return
            self._thread = threading.Thread(target=self._run, name="hermes-gateway-loop", daemon=True)
            self._thread.start()
        if not self._ready.wait(timeout):
            raise GatewayClientError("Gateway connection timed out; no local fallback")
        if self._error is not None:
            raise self._error

    def _run(self):
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # health: allow BLE001 -- the loop thread reports every failure to the waiting caller
            if self._error is None:
                self._error = exc if isinstance(exc, GatewayClientError) else GatewayClientError(
                    "Gateway connection failed; no local fallback")
        finally:
            self.closed = True
            self._ready.set()

    async def _main(self):
        self.loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        connect = self._connect
        if connect is None:
            from hermes_cli.gateway_client import connect_gateway
            connect = connect_gateway
        try:
            async with connect() as client:
                self.client = client
                self._ready.set()
                pump = asyncio.create_task(self._pump(client))
                stop = asyncio.create_task(self._stop.wait())
                await asyncio.wait({pump, stop}, return_when=asyncio.FIRST_COMPLETED)
                pump.cancel()
                stop.cancel()
        except Exception as exc:  # health: allow BLE001 -- a transport failure becomes the connect error or the disconnect frame
            if not self._ready.is_set():
                self._error = exc if isinstance(exc, GatewayClientError) else GatewayClientError(
                    f"Gateway connection failed; no local fallback ({type(exc).__name__})")
            else:
                self.events.put(GatewayClientError(DISCONNECTED))
        finally:
            self.client = None

    async def _pump(self, client):
        while True:
            frame = await client.events.get()
            if isinstance(frame, Exception):
                self.events.put(frame)
                return
            params = frame.get("params") if isinstance(frame, dict) else None
            if isinstance(params, dict) and self._take_side(params):
                continue
            if isinstance(params, dict):
                sid = params.get("session_id")
                if sid in self.states:
                    try:
                        self.states[sid].apply_event(params.get("type"), params.get("payload"))
                    except Exception:  # pragma: no cover - a cache update must never stop the pump
                        logger.debug("session state update failed", exc_info=True)
            self.events.put(frame)

    def _take_side(self, params):
        from hermes_cli.gateway_side import SIDE_EVENTS
        payload = params.get("payload")
        if params.get("type") not in SIDE_EVENTS or not isinstance(payload, dict) or not payload.get("task_id"):
            return False
        self.side_waiter(payload["task_id"]).put(payload)
        return True

    def side_waiter(self, task_id) -> queue.Queue:
        with self._side_lock:
            return self._side.setdefault(task_id, queue.Queue())

    def drop_side_waiter(self, task_id):
        with self._side_lock:
            self._side.pop(task_id, None)

    def close(self):
        loop, stop = self.loop, self._stop
        if loop is not None and stop is not None and not self.closed:
            try:
                loop.call_soon_threadsafe(stop.set)
            except RuntimeError:  # the loop already ended (the gateway hung up first)
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(5)
        self.closed = True

    # -- requests ----------------------------------------------------------------------------
    def reply(self, request_id, result):
        """Answer one of the gateway's server→client requests (the ``sudo`` prompt): a JSON-RPC
        response frame with the request's own id. Never raises; a lost reply times out gateway-side."""
        import json
        client, loop = self.client, self.loop
        if client is None or loop is None:
            return False
        frame = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result})
        try:
            asyncio.run_coroutine_threadsafe(client.websocket.send(frame), loop).result(10)
        except Exception as exc:  # health: allow BLE001 -- the gateway's own wait bounds a lost answer
            logger.debug("server request reply failed: %s", exc)
            return False
        return True

    def call(self, method, *, timeout=None, **params):
        if self._thread is not None and threading.current_thread() is self._thread:
            raise RuntimeError("GatewayConnection.call() on the gateway loop thread would deadlock")
        self.start()
        client, loop = self.client, self.loop
        if client is None or loop is None:
            raise GatewayClientError(DISCONNECTED)
        budget = DEFAULT_RPC_TIMEOUT if timeout is None else timeout
        future = asyncio.run_coroutine_threadsafe(client.rpc(method, _timeout=budget, **params), loop)
        try:
            return future.result(budget + 5)
        except TimeoutError as exc:
            future.cancel()
            raise GatewayClientError(f"Gateway did not answer {method}") from exc
