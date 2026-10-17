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
import time

from hermes_cli.gateway_client import DEFAULT_RPC_TIMEOUT, GatewayClientError

logger = logging.getLogger(__name__)

DISCONNECTED = ("Gateway disconnected; turn outcome is unknown. Resume the printed session ID to check "
                "whether it completed or was interrupted; do not resend the work.")
# A request refused before it reached any socket: nothing is in doubt, so never the copy above.
NOT_SENT = "Gateway disconnected before the request was sent; nothing was sent."
# Resume refusals meaning the session itself is gone for this viewer (gateway/session_controls).
_GONE = frozenset({"not_found", "permission_denied", "profile_mismatch"})


class CallStopped(GatewayClientError):
    """The caller stopped waiting (:meth:`GatewayConnection.call_until`); the request was sent."""


def lost_before(frame, epoch) -> bool:
    """``frame`` is a transport error from a connection older than ``epoch``: a reconnect replaced it,
    so it is no outcome of anything submitted on ``epoch``."""
    lost = getattr(frame, "connection_epoch", None)
    return isinstance(frame, Exception) and lost is not None and epoch is not None and lost < epoch


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
        # Called (loop thread) with each ``request.cancel`` payload: the agent thread may be parked in a
        # modal answering that very request (another viewer settled it), so it cannot read the queue.
        self.request_listeners: list = []
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

    def runs_route(self, model, provider=""):
        """Whether the session runs ``model`` (and ``provider``, when both sides name one)."""
        current, current_provider = self.info.get("model"), self.info.get("provider")
        return bool(current) and model == current and not (
            provider and current_provider and provider != current_provider)

    def apply_event(self, kind, payload):
        if kind in ("session.info", "session.updated") and isinstance(payload, dict):
            self.info.update({k: v for k, v in payload.items() if k != "session_id"})
            if payload.get("title"):
                self.title = payload["title"]
        elif kind == "todo.updated" and isinstance(payload, dict):
            self.todos = list(payload.get("todos") or [])
        elif kind == "session.usage" and isinstance(payload, dict):
            # The gateway's frame carries the flat counters next to a nested TUI ``usage`` view this
            # reader can't map (``session_usage.usage_frame``): the counters win when present.
            nested = payload.get("usage")
            flat = isinstance(payload.get("api_calls"), int) or not isinstance(nested, dict)
            self.apply_usage(payload if flat else nested)
        elif kind == "request.cancel" and isinstance(payload, dict):
            for listener in list(self.request_listeners):
                listener(payload)

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
        # (client, epoch) of the open socket, swapped as one: a caller never pairs a socket with
        # another life's epoch (``call_tagged``).
        self._live = None
        self.loop = None
        self._thread = None
        self._ready = threading.Event()
        self._error = None
        self._stop = None
        self._start_lock = threading.Lock()
        self.closed = False
        # close() ends the viewer; a gateway hang-up (restart, idle exit, crash) lets the next call
        # dial again. Each connect bumps ``epoch``, and its loss is tagged with it (``lost_before``).
        self._detached = False
        self.epoch = 0
        # /bg and /btw answers by task id (hermes_cli/gateway_side.py); created by whichever of the
        # waiter or the frame comes first, so an answer faster than the verb's reply is kept.
        self._side: dict[str, queue.Queue] = {}
        self._side_lock = threading.Lock()
        # The wake events of :meth:`call_until` commands in flight, each on its own socket.
        self._commands: set[threading.Event] = set()

    @classmethod
    def shared(cls):
        with cls._shared_lock:
            if cls._shared is None or cls._shared.final:
                cls._shared = cls()
            return cls._shared

    @property
    def final(self) -> bool:
        """No call dials again: the viewer closed it, or its first connect failed. ``closed`` is also
        true while the gateway is hung up, which the next ``call()``/``start()`` redials."""
        return self._detached or (self._error is not None and not self.epoch)

    def state(self, session_id) -> SessionState:
        state = self.states.get(session_id)
        if state is None:
            state = self.states[session_id] = SessionState(session_id)
        return state

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self, timeout=90):
        deadline = time.monotonic() + timeout
        while True:
            with self._start_lock:
                if self.final:
                    if self._error is not None:
                        raise self._error
                    return
                thread = self._thread
                if thread is not None and self.closed and thread.is_alive() and thread is not threading.current_thread():
                    # ``_run`` reports ready before its thread returns: that life is over, so let it
                    # end instead of reading it as a dial in progress (and re-raising its stale error).
                    thread.join(max(deadline - time.monotonic(), 0))
                if thread is None or not thread.is_alive():
                    from agent.memory_provider import spawn_context_thread

                    if thread is not None:
                        # The gateway hung up, or the last redial failed: dial again (``connect_gateway``
                        # starts a gateway when none is running) and resubscribe this viewer's sessions.
                        self._error = None
                        self._ready.clear()
                        self.closed = False
                    thread = self._thread = spawn_context_thread(self._run, name="hermes-gateway-loop")
                    thread.start()
            # Every caller waits out a dial in progress, its own or another thread's.
            if not self._ready.wait(max(deadline - time.monotonic(), 0)):
                raise GatewayClientError("Gateway connection timed out; no local fallback")
            if self._error is not None:
                raise self._error
            if self.client is not None or self.final:
                return
            # Hung up right after that dial: let its loop end, then dial again.
            thread.join(max(deadline - time.monotonic(), 0))
            if time.monotonic() >= deadline:
                raise GatewayClientError("Gateway connection timed out; no local fallback")

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
                if self.epoch:
                    await self._resubscribe(client)
                self.epoch += 1
                self.client = client
                self._live = (client, self.epoch)
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
                self._put_loss(GatewayClientError(DISCONNECTED))
        finally:
            self.client = None
            self._live = None

    def _put_loss(self, error):
        error.connection_epoch = self.epoch
        self.events.put(error)

    async def _resubscribe(self, client):
        """A new socket is a new viewer: the gateway streams a session's frames only to connections
        that resumed it. Resume each session this viewer had open; one gone meanwhile is dropped. Any
        other failure fails the reconnect: ready without the subscription, every prompt is refused on a
        live socket that never redials, while a failed redial is retried by the next call."""
        for session_id, state in list(self.states.items()):
            if state.subscription_id is None:
                continue
            try:
                state.adopt_snapshot(await client.rpc("session.resume", session_id=session_id))
            except GatewayClientError as exc:
                if str(exc) not in _GONE:
                    raise
                logger.debug("session.resume %s after reconnect: %s", session_id, exc)
                state.subscription_id = None

    async def _pump(self, client):
        while True:
            frame = await client.events.get()
            if isinstance(frame, Exception):
                self._put_loss(frame)
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
        self._detached = True
        with self._side_lock:
            for wake in self._commands:  # their sockets close too
                wake.set()
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
        return self.call_tagged(method, timeout=timeout, **params)[0]

    def _send(self, method, budget, params):
        """``(future, epoch)`` of one request sent on the live socket (see :meth:`call_tagged`)."""
        if self._thread is not None and threading.current_thread() is self._thread:
            raise RuntimeError("GatewayConnection.call() on the gateway loop thread would deadlock")
        self.start()
        live, loop = self._live, self.loop
        if live is None or loop is None:
            raise GatewayClientError(NOT_SENT)
        client, epoch = live
        return asyncio.run_coroutine_threadsafe(client.rpc(method, _timeout=budget, **params), loop), epoch

    def call_until(self, method, wake, **params):
        """``call`` with no answer deadline, for a verb that lasts as long as the work it runs (a plugin
        command's handler), on a socket of its own to this connection's gateway: the gateway answers a
        socket's requests one at a time, in arrival order (``tui_gateway/ws.py``), so a handler still
        running never holds this connection's reads, prompts and interrupts. One dial and one send,
        never retried. Waits for the answer or the transport's error, unless *wake* (a
        ``threading.Event``, set by Ctrl+C or :meth:`close`) is set first: then raises
        :class:`CallStopped`; the request was sent and its work goes on, but its answer is dropped. The
        socket closes on any of these, and its thread is gone before this returns."""
        import concurrent.futures
        from agent.memory_provider import spawn_context_thread
        if self._thread is not None and threading.current_thread() is self._thread:
            raise RuntimeError("GatewayConnection.call_until() on the gateway loop thread would deadlock")
        self.start()  # this viewer's gateway, dialled (or redialled after a hang-up) as for any call
        if self.final:
            raise GatewayClientError(NOT_SENT)
        answer, sent = concurrent.futures.Future(), threading.Event()
        answer.add_done_callback(lambda _done: wake.set())
        thread = spawn_context_thread(lambda: asyncio.run(self._command(method, params, wake, answer, sent)),
                                      name="hermes-gateway-command")
        with self._side_lock:
            self._commands.add(wake)
        thread.start()
        try:
            wake.wait()
        finally:
            wake.set()  # also on KeyboardInterrupt: the socket closes
            thread.join(10)
            with self._side_lock:
                self._commands.discard(wake)
        if answer.done():
            return answer.result()
        if sent.is_set():
            raise CallStopped(method)
        raise GatewayClientError(NOT_SENT)

    async def _command(self, method, params, wake, answer, sent):
        """The command socket of :meth:`call_until`: dial, send once, settle ``answer`` with the reply or
        the transport's error; a set ``wake`` without an answer cancels the dial or the wait."""
        connect = self._connect
        if connect is None:
            from hermes_cli.gateway_client import connect_gateway
            connect = connect_gateway
        task = asyncio.current_task()
        stop = asyncio.ensure_future(asyncio.to_thread(wake.wait))
        stop.add_done_callback(lambda _stop: answer.done() or task.cancel())
        try:
            async with connect() as client:
                sent.set()  # from here the request may have reached the gateway: never sent again
                answer.set_result(await client.rpc(method, _timeout=float("inf"), **params))
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # health: allow BLE001 -- every transport failure is the caller's error
            if not isinstance(exc, GatewayClientError):
                exc = GatewayClientError(DISCONNECTED if sent.is_set() else NOT_SENT)
            answer.set_exception(exc)
        finally:
            wake.set()  # ends the watcher above
            await asyncio.gather(stop, return_exceptions=True)

    def call_tagged(self, method, *, timeout=None, **params):
        """``(result, epoch)``: the epoch of the socket that carried the request, so work it admitted
        is tied to that gateway life even if another caller redials before this one reads ``epoch``."""
        budget = DEFAULT_RPC_TIMEOUT if timeout is None else timeout
        future, epoch = self._send(method, budget, params)
        try:
            return future.result(budget + 5), epoch
        except TimeoutError as exc:
            future.cancel()
            raise GatewayClientError(f"Gateway did not answer {method}") from exc
