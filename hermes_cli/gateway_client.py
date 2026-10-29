"""Transport-only RPC client. Closing a viewer never stops its authority."""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager, contextmanager, suppress
import contextvars
import json
import os
from pathlib import Path
import time

# The owner's protocol name; session_contract is dependency-free (no config/plugins load).
from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL as GATEWAY_WS_PROTOCOL

GATEWAY_WS_TICKET_PREFIX = "hermes-gateway-ticket."
# Per-request answer budget. Owner verbs are admissions/reads that answer promptly; the one that
# legitimately runs long passes its own budget (``rpc(..., _timeout=...)``).
DEFAULT_RPC_TIMEOUT = 30
# A typed ``--api-key`` on a resume that the owner does not bind (``rebind_launch_key``): every CLI says this.
RESUME_KEY_MISMATCH = ("--api-key on resume must be the key this session was launched with "
                       "(a session launched without --api-key keeps its profile credentials). "
                       "A session never switches keys: start a new one (without --resume) to use a different key.")
# Canonical ``/compress`` answers only after the summary call(s) and the commit: a long session's
# summary routinely takes more than 30 s (``auxiliary.compression.timeout`` defaults to 120 s per
# request, ``compression.context_total_ceiling_seconds`` to 600 s). Same budget as Desktop's
# ``SESSION_COMPRESS_TIMEOUT_MS`` so every surface gives up at the same point.
COMPRESS_RPC_TIMEOUT = 660
# A full event queue parks the socket reader; a consumer that takes no event for this long has
# stalled and is failed. Same bound as the owner's own send deadline (``tui_gateway/ws.py``).
EVENT_STALL_TIMEOUT = 30
# Events read past a full queue while an RPC answer is awaited (the owner sends a re-attached
# session's frames ahead of its ``session.resume`` answer, and a recovery RPC is awaited from the
# queue's only consumer). Bounded: past either cap the reader waits for the consumer again.
EVENT_OVERFLOW_FRAMES = 16384
EVENT_OVERFLOW_BYTES = 8 * 1024 * 1024


def gateway_ws_target(endpoint, ticket):
    """``(url, subprotocols)`` for dialing the owner's canonical WebSocket with a session ticket.

    The owner accepts exactly one ticket subprotocol beside ``GATEWAY_WS_PROTOCOL`` and echoes the
    latter; callers check ``ws.subprotocol == GATEWAY_WS_PROTOCOL`` after the upgrade."""
    url = endpoint.api_origin.replace("https:", "wss:").replace("http:", "ws:") + "/api/ws"
    return url, [GATEWAY_WS_PROTOCOL, GATEWAY_WS_TICKET_PREFIX + ticket]


class GatewayClientError(ValueError):
    pass


class GatewayRPCError(GatewayClientError):
    """A received owner refusal, distinct from an ambiguous transport failure.

    ``str(exc)`` is the bounded reason code; ``code`` is the frame's numeric JSON-RPC code (``None``
    when absent or not an int) and ``data`` its bounded, scalar-only ``error.data`` mapping."""

    def __init__(self, reason, code=None, data=None):
        super().__init__(reason)
        self.code = code
        self.data = data or {}


def _bounded_reason(value):
    return value if isinstance(value, str) and value.replace("_", "").isalnum() and len(value) <= 80 else None


def _bounded_error_data(data):
    """At most 16 short keys whose values are bool/int/None, short strings or short lists of them;
    anything else (nested objects, raw remote diagnostics) is dropped, never forwarded."""
    def scalar(value):
        return value is None or isinstance(value, (bool, int)) or (isinstance(value, str) and len(value) <= 200)
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, value in data.items():
        if len(out) >= 16 or not isinstance(key, str) or len(key) > 64:
            continue
        if scalar(value):
            out[key] = value
        elif isinstance(value, list) and len(value) <= 16 and all(scalar(item) for item in value):
            out[key] = list(value)
    return out


def rpc_error(error):
    """The ``GatewayRPCError`` for one received ``error`` member: only bounded reason codes, never
    raw remote diagnostics. A free-text message (``unknown method: x``) falls back to the bounded
    ``data.reason`` the owner sends beside it, then to ``request_failed``."""
    error = error if isinstance(error, dict) else {}
    data = _bounded_error_data(error.get("data"))
    code = error.get("code")
    reason = _bounded_reason(error.get("message", "request_failed")) or _bounded_reason(data.get("reason"))
    return GatewayRPCError(reason or "request_failed",
                           code if isinstance(code, int) and not isinstance(code, bool) else None, data)


class GatewayUnavailableError(GatewayClientError):
    """No ready owner to dial: ``state`` is ``ensure_gateway_runtime``'s verdict (``draining`` while an
    update holds the install, ``starting`` past the deadline, ...). Nothing was submitted."""

    def __init__(self, message, state, reason_code=None):
        super().__init__(message)
        self.state, self.reason_code = state, reason_code


class EventQueue(asyncio.Queue):
    """The client's bounded event queue plus an ordered overflow behind it: each ``get`` refills the
    queue from the overflow, so items arrive in read order and ``qsize``/``full`` see only the queue."""

    def __init__(self, maxsize):
        super().__init__(maxsize)
        self.overflow = deque()  # (size, item)
        self.overflow_bytes = 0
        self.moved = asyncio.Event()  # an item left the overflow (or an RPC started waiting)

    def _get(self):
        item = super()._get()
        if self.overflow:
            size, held = self.overflow.popleft()
            self.overflow_bytes -= size
            super()._put(held)
        self.moved.set()
        return item

    def hold(self, item, size=0):
        """Queue *item* behind everything already read, never dropping it."""
        if not self.overflow:
            try:
                self.put_nowait(item)
                return
            except asyncio.QueueFull:
                pass
        self.overflow.append((size, item))
        self.overflow_bytes += size


def put_behind(events, item):
    """Queue a local item (ACP's post-ack recovery) behind every frame read so far; never dropped."""
    if isinstance(events, EventQueue):
        events.hold(item)
    else:
        events.put_nowait(item)


class GatewayClient:
    def __init__(self, websocket):
        self.websocket = websocket
        self.events = EventQueue(maxsize=4096)
        self.pending = {}
        self.sequence = 0
        self.reader = None

    def _read_on(self):
        """Keep reading past a full queue: nothing is held, or an awaited answer may sit behind the
        held frames (its waiter may be the queue's only consumer) and the overflow has room."""
        events = self.events
        return not events.overflow or (
            len(events.overflow) < EVENT_OVERFLOW_FRAMES and events.overflow_bytes < EVENT_OVERFLOW_BYTES
            and any(not future.done() for future in self.pending.values()))

    async def _deliver(self, frame, size):
        self.events.hold(frame, size)
        if self._read_on():
            return
        # Frames the socket already buffered (a CLI resumed from Ctrl-Z) are read without yielding,
        # so the consumer has not had a turn yet: wait for it (or for an RPC that needs the reader).
        # Not wait_for: on 3.11 it swallows a close that lands as the wait completes.
        async with asyncio.timeout(EVENT_STALL_TIMEOUT):
            while not self._read_on():
                self.events.moved.clear()
                await self.events.moved.wait()

    async def __aenter__(self):
        self.reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc):
        self.reader.cancel()
        with suppress(asyncio.CancelledError):
            await self.reader

    async def _read(self):
        from websockets.exceptions import ConnectionClosed
        try:
            async for raw in self.websocket:
                frame = json.loads(raw)
                future = self.pending.get(frame.get("id"))
                if future is not None and not future.done():
                    if "error" in frame:
                        future.set_exception(rpc_error(frame["error"]))
                    else:
                        future.set_result(frame.get("result", {}))
                elif "method" in frame:
                    await self._deliver(frame, len(raw))
        except (ConnectionClosed, OSError, ValueError, TimeoutError):
            pass
        finally:
            error = GatewayClientError("Gateway disconnected; turn outcome is unknown. "
                "Resume the printed session ID to check whether it completed or was interrupted; do not resend the work.")
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)
            # A stalled consumer is failed rather than silently losing terminal events: behind
            # whatever is held, or in place of the oldest queued frame when nothing is.
            if self.events.full() and not self.events.overflow:
                self.events.get_nowait()
            self.events.hold(error)

    async def rpc(self, method, *, _timeout=None, **params):
        """Send one request and await its answer; ``_timeout`` (seconds, never sent on the wire)
        overrides ``DEFAULT_RPC_TIMEOUT`` for a verb that legitimately takes longer (``math.inf``: no
        deadline; the answer, a hang-up or the caller's cancel ends the wait)."""
        self.sequence += 1
        rid = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[rid] = future
        self.events.moved.set()  # a reader waiting on the consumer reads on toward this answer
        try:
            await self.websocket.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}))
            budget = DEFAULT_RPC_TIMEOUT if _timeout is None else _timeout
            return await asyncio.wait_for(future, None if budget == float("inf") else budget)
        finally:
            self.pending.pop(rid, None)


def private_control_query(home: Path, request: bytes, timeout: float = 5) -> bytes:
    """Send one request line to *home*'s gateway control socket as this OS user; the raw answer.

    The socket authenticates the peer (same user only), which is what lets a private verb answer."""
    if os.name == "nt":
        from gateway.runtime_bootstrap_windows import query_runtime_control
        return query_runtime_control(home, request, timeout)
    from hermes_cli.gateway_runtime_discovery import connect_private
    deadline = time.monotonic() + timeout
    with connect_private(home, timeout) as peer:
        peer.sendall(request)
        data = bytearray()
        while b"\n" not in data:
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise GatewayClientError("Gateway bootstrap timed out")
            peer.settimeout(budget)
            chunk = peer.recv(4096)
            if not chunk or len(data) + len(chunk) > 65536:
                raise GatewayClientError("Invalid gateway bootstrap response")
            data.extend(chunk)
    return bytes(data)


# An owner that predates ``restart_wait`` refuses the unknown parameter with exactly this answer.
_UNKNOWN_TICKET_PARAMS = "PermissionError: invalid bootstrap parameters"


def _session_ticket(home: Path, endpoint, *, purpose="interactive", scope=None, subject=None,
                    refuse_restart_wait=False) -> str:
    from hermes_cli.gateway_runtime import control_home_for
    from hermes_cli.gateway_runtime_discovery import _identify_response
    # A served secondary's ticket is minted by the multiplexer's socket, bound to the secondary.
    home = control_home_for(home, endpoint)
    # ``subject`` names the principal a same-user caller acts for (the standalone dashboard's
    # authenticated user) instead of the socket UID; the owner's peer gate stays the boundary.
    # ``restart_wait``: the owner refuses the socket at admission if a restart wait has opened by then.
    params = {"profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id, "purpose": purpose,
              **({"scope": scope} if scope else {}), **({"subject": subject} if subject else {}),
              **({"restart_wait": "refuse"} if refuse_restart_wait else {})}
    request = json.dumps({"protocol": 1, "id": 1, "verb": "session-ticket", "params": params}).encode() + b"\n"
    answer = private_control_query(home, request)
    refused = json.loads(answer.split(b"\n", 1)[0]) if refuse_restart_wait else None
    if isinstance(refused, dict) and refused.get("error") == _UNKNOWN_TICKET_PARAMS:
        # An owner older than this client (the code on disk was updated under it) cannot hold the
        # policy; refusing it outright would strand the caller until it restarts. The pre-dial check stands.
        del params["restart_wait"]
        answer = private_control_query(home, json.dumps(
            {"protocol": 1, "id": 1, "verb": "session-ticket", "params": params}).encode() + b"\n")
    grant = _identify_response(answer)
    if any(grant.get(k) != v for k, v in {
        "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id, "runtime_protocol": 1,
    }.items()) or not isinstance(grant.get("ticket"), str) or not grant["ticket"]:
        raise GatewayClientError("Gateway bootstrap identity changed; retry launch")
    return grant["ticket"]


# A restart's after-turn wait admits clients so a human can answer a parked prompt (#139041). A machine
# poller held in it (a Kanban worker reading its card's receipt) keeps the turn the wait is waiting
# on alive, so the restart, and ``hermes update`` behind it, is held until its cap.
_REFUSE_RESTART_WAIT: contextvars.ContextVar[bool] = contextvars.ContextVar("refuse_restart_wait", default=False)


@contextmanager
def outside_restart_wait():
    """``connect_gateway`` calls inside never attach to an owner in its restart's after-turn wait."""
    token = _REFUSE_RESTART_WAIT.set(True)
    try:
        yield
    finally:
        _REFUSE_RESTART_WAIT.reset(token)


@asynccontextmanager
async def connect_gateway(endpoint=None, subject=None, *, max_size=8 * 1024 * 1024):
    """Attach to this home's owner. ``endpoint``: an already-discovered ready owner to dial as-is
    (maintenance that must never start a gateway); default ensures one, starting it if needed.
    ``subject``: mint the ticket for this principal (the caller's authenticated dashboard user)
    rather than the socket identity; only a peer the owner already trusts may name one.
    ``max_size``: the largest frame accepted; a ``session.resume`` answer carries the whole history.
    Inside ``outside_restart_wait()`` an owner in a restart's after-turn wait is refused as
    ``draining`` (``restart_wait``) instead of attached, and the ticket carries that policy so the
    owner also refuses the socket when the wait opens between discovery and the dial."""
    from websockets.asyncio.client import connect
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import attachable, ensure_gateway_runtime, gateway_profile_home
    from hermes_cli.gateway_runtime_discovery import DiscoveryError
    from urllib.parse import urlsplit

    remote = "" if endpoint is not None else os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
    protocols = None
    if endpoint is not None:
        ticket = await asyncio.to_thread(_session_ticket, gateway_profile_home(get_hermes_home()), endpoint,
                                         subject=subject)
        url, protocols = gateway_ws_target(endpoint, ticket)
    elif remote:
        parsed = urlsplit(remote)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise GatewayClientError("Invalid explicit gateway WebSocket URL; no local fallback")
        url = remote
    else:
        # A per-task overlay home is served by the profile its symlinked lock names (#138949).
        home = gateway_profile_home(get_hermes_home())
        # A client's own start: the daemon it may launch ends itself once idle (no client, adapter,
        # cron/kanban work or admission), and the next client starts a fresh one.
        result = await asyncio.to_thread(ensure_gateway_runtime, home, idle_exit=True)
        if not attachable(result) or (_REFUSE_RESTART_WAIT.get() and result.state != "ready"):
            detail = getattr(result, "detail", None)
            if result.reason_code == "runtime_exited" and detail:
                # A multi-line startup report (redacted traceback tail + where the full log is).
                raise GatewayUnavailableError(f"Gateway could not start: {detail}", result.state, result.reason_code)
            detail = f" ({detail})" if detail else ""
            raise GatewayUnavailableError(f"Gateway {result.state}: {result.reason_code or 'not_ready'}{detail}",
                                          result.state, result.reason_code)
        endpoint = result.endpoint
        try:
            policy: dict = {"refuse_restart_wait": True} if _REFUSE_RESTART_WAIT.get() else {}
            ticket = await asyncio.to_thread(_session_ticket, home, endpoint, **policy)
        except DiscoveryError as exc:
            # The control socket refused the ticket: the verdict discovery gives the same refusal.
            detail = exc.bounded_detail(500)
            raise GatewayUnavailableError(f"Gateway inaccessible: {exc.reason}" + (f" ({detail})" if detail else ""),
                                          "inaccessible", exc.reason) from exc
        url, protocols = gateway_ws_target(endpoint, ticket)
    try:
        # The gateway is a loopback (or explicitly named) peer, never something to route through the
        # user's HTTP(S) proxy; websockets>=14 reads HTTP_PROXY/HTTPS_PROXY by default and a proxy that
        # cannot reach 127.0.0.1 turns every launch into a 10 s open timeout.
        async with connect(url, subprotocols=protocols, open_timeout=10, max_size=max_size,
                           proxy=None) as ws:
            if protocols and ws.subprotocol != GATEWAY_WS_PROTOCOL:
                raise GatewayClientError("Gateway protocol mismatch; update/restart required")
            async with GatewayClient(ws) as client:
                yield client
    except GatewayClientError:
        raise
    except (OSError, TimeoutError) as exc:
        raise GatewayClientError("Gateway connection failed; no local fallback") from exc
