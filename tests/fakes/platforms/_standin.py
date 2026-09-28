"""Shared plumbing for local stand-in messaging platform servers.

A stand-in is a real HTTP (and, where the platform needs it, WebSocket) server on 127.0.0.1 that
implements only the endpoints a Hermes adapter actually calls, shaped per the platform's published
API reference. It runs on a private asyncio loop in a daemon thread so a synchronous pytest body can
drive it, records every call it serves (method + decoded params + the response it returned), and
lets a test queue faults per method (an API error body, an HTTP status) that are consumed in order.

Nothing here knows about Hermes: the adapter under test talks to it through its own SDK
(python-telegram-bot, discord.py, slack_sdk) exactly as it would talk to the real platform.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from aiohttp import web


@dataclass
class Call:
    """One request the stand-in served."""

    method: str
    params: Dict[str, Any]
    at: float
    response: Any = None
    faulted: bool = False


@dataclass
class Fault:
    """A queued failure for one API method; ``match`` narrows which call consumes it."""

    method: str
    body: Any
    status: int = 200
    times: int = 1
    match: Optional[Callable[[Dict[str, Any]], bool]] = None
    fired: int = 0


@dataclass
class Visible:
    """What a human would currently see for one bot message in a chat (after edits/deletes)."""

    message_id: str
    text: str
    edits: int = 0
    deleted: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def wait_until(pred: Callable[[], Any], what: str, timeout: float = 60.0, interval: float = 0.05,
               on_timeout: Optional[Callable[[], str]] = None) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        got = pred()
        if got:
            return got
        if time.monotonic() > deadline:
            ctx = on_timeout() if on_timeout else ""
            raise AssertionError(f"timed out after {timeout}s waiting for {what}\n{ctx}")
        time.sleep(interval)


def decode_value(raw: str) -> Any:
    """Bot-API style form fields carry JSON for structured values; plain strings stay strings."""
    text = raw.strip()
    if text[:1] in "[{" or text in ("true", "false", "null"):
        try:
            return json.loads(text)
        except ValueError:
            return raw
    return raw


class StandinServer:
    """Base class: an aiohttp application on a private loop thread with call recording + faults."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.calls: List[Call] = []
        self._faults: List[Fault] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner: Optional[web.AppRunner] = None
        self._ready = threading.Event()
        self.port = 0

    # lifecycle ---------------------------------------------------------------------------------
    def build_app(self) -> web.Application:  # pragma: no cover - abstract
        raise NotImplementedError

    def start(self) -> "StandinServer":
        self._thread = threading.Thread(target=self._serve, name=type(self).__name__, daemon=True)
        self._thread.start()
        if not self._ready.wait(30):
            raise RuntimeError(f"{type(self).__name__} did not start")
        return self

    def _serve(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._astart())
        self._ready.set()
        self._loop.run_forever()

    async def _astart(self) -> None:
        self._runner = web.AppRunner(self.build_app(), access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    def stop(self) -> None:
        if self._loop is None:
            return

        async def _shutdown() -> None:
            await self.on_shutdown()
            if self._runner is not None:
                await self._runner.cleanup()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(15)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(10)
        self._loop = None

    async def on_shutdown(self) -> None:
        """Subclasses close long-lived sockets here."""

    def __enter__(self) -> "StandinServer":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def run_in_loop(self, coro: Any, timeout: float = 15.0) -> Any:
        assert self._loop is not None, "stand-in not running"
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # recording + faults ------------------------------------------------------------------------
    def record(self, method: str, params: Dict[str, Any], response: Any, faulted: bool = False) -> Call:
        call = Call(method=method, params=params, at=time.monotonic(), response=response, faulted=faulted)
        with self._lock:
            self.calls.append(call)
        return call

    def calls_of(self, *methods: str) -> List[Call]:
        with self._lock:
            return [c for c in self.calls if c.method in methods]

    def fail(self, method: str, body: Any, *, status: int = 200, times: int = 1,
             match: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Fault:
        fault = Fault(method=method, body=body, status=status, times=times, match=match)
        with self._lock:
            self._faults.append(fault)
        return fault

    def clear_faults(self) -> None:
        with self._lock:
            self._faults.clear()

    def take_fault(self, method: str, params: Dict[str, Any]) -> Optional[Fault]:
        with self._lock:
            for fault in self._faults:
                if fault.method != method or fault.fired >= fault.times:
                    continue
                if fault.match is not None and not fault.match(params):
                    continue
                fault.fired += 1
                return fault
        return None

    def describe(self, last: int = 40) -> str:
        """Human-readable tail of the call log for assertion messages."""
        with self._lock:
            rows = self.calls[-last:]
        out = []
        for c in rows:
            p = {k: (v[:80] + "…" if isinstance(v, str) and len(v) > 80 else v) for k, v in c.params.items()
                 if k not in ("reply_markup", "blocks", "embeds", "components")}
            out.append(f"  {'!' if c.faulted else ' '}{c.method} {json.dumps(p, default=str)[:240]}")
        return "--- stand-in calls (last %d)\n%s" % (len(rows), "\n".join(out))
