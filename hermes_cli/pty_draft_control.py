"""Narrow draft capability owned by a live PTY, never by a telemetry channel."""
from __future__ import annotations

import asyncio
import hmac
from typing import Any
import json
import secrets
from weakref import WeakValueDictionary
from starlette.websockets import WebSocketDisconnect

DRAFT_CONTROL_PREFIX = b"\x00hermes-draft-v1:"
LIVE_CONTROLLERS: WeakValueDictionary[str, PtyDraftControl] = WeakValueDictionary()


def _valid_identity(value):
    return (isinstance(value, dict)
            and set(value) == {"pty_instance", "connection_generation", "session_id", "draft_id"}
            and all(isinstance(value[key], str) and value[key] for key in ("pty_instance", "session_id", "draft_id"))
            and type(value["connection_generation"]) is int and value["connection_generation"] >= 0)


class PtyDraftControl:
    def __init__(self, channel: str | None = None):
        self.credential = secrets.token_urlsafe(32)
        self.pty_instance = secrets.token_urlsafe(18)
        self.session: Any = None
        self.controller = None
        self.controller_generation = 0
        self.generation = 0
        self.state = None
        self.pending = {}
        self.channel = channel
        self._claim_lock = asyncio.Lock()
        LIVE_CONTROLLERS[self.credential] = self

    def binding(self):
        return {"pty_instance": self.pty_instance, "connection_generation": self.generation}

    async def claim(self, ws, *, credential: str, generation: int) -> bool:
        async with self._claim_lock:
            return await self._claim(ws, credential=credential, generation=generation)

    async def _claim(self, ws, *, credential: str, generation: int) -> bool:
        if (not self.session or not self.session.alive
                or not hmac.compare_digest(credential.encode(), self.credential.encode())
                or type(generation) is not int or generation < 1
                or generation < self.controller_generation
                or (generation == self.controller_generation and self.controller is not None)):
            return False
        old = self.controller
        if old is not None:
            await self.disconnected(old)
        self.controller = ws
        self.controller_generation = generation
        self.pending.clear()
        self.state = None
        self.generation += 1
        if old is not None:
            await self._send(old, {"type": "draft.disconnected"})
            await self._close(old, code=4409)
        await self._send(ws, {"type": "draft.refresh", **self.binding()})
        return True

    async def _send(self, ws, frame):
        try:
            await ws.send_json(frame)
        except (RuntimeError, OSError, WebSocketDisconnect):
            # A disconnected sidecar must not break terminal typing.
            return False
        return True

    async def _close(self, ws, code):
        try:
            await ws.close(code=code)
        except (RuntimeError, OSError, WebSocketDisconnect):
            # Starlette marks a failed sender disconnected before close runs.
            pass

    async def browser_attached(self):
        self.generation += 1
        self.pending.clear()
        self.state = None
        if self.controller is not None:
            await self._send(self.controller, {"type": "draft.refresh", **self.binding()})

    def browser_detached(self):
        self.pending.clear()
        self.state = None
        self.generation += 1
        ws, generation = self.controller, self.generation
        if ws is not None:
            async def notify():
                if self.controller is ws and self.generation == generation:
                    await self._send(ws, {"type": "draft.disconnected"})
            asyncio.create_task(notify())

    async def disconnected(self, ws):
        if self.controller is not ws:
            return
        self.controller = None
        pending = list(self.pending.values())
        self.pending.clear()
        state, self.state = self.state, None
        for request, browser in pending:
            if self.session._ws is browser:
                await self._send(browser, {"type": "draft.result", "request_id": request["request_id"],
                                          "identity": request["expected"], "status": "unavailable"})
        if state is not None and self.session._ws is not None:
            await self._send(self.session._ws, {**state, "available": False})

    async def close(self):
        LIVE_CONTROLLERS.pop(self.credential, None)
        ws = self.controller
        if ws is not None:
            await self.disconnected(ws)
            await self._send(ws, {"type": "draft.disconnected"})
            await self._close(ws, code=4410)

    async def receive(self, ws, frame):
        if (ws is not self.controller or not self.session.alive
                or not isinstance(frame, dict)) :
            return
        if frame.get("type") == "draft.state":
            if any(frame.get(key) != value for key, value in self.binding().items()):
                return
            if not all(isinstance(frame.get(key), str) and frame[key] for key in ("session_id", "draft_id")):
                return
            if not isinstance(frame.get("available"), bool):
                return
            identity = {**self.binding(), "session_id": frame["session_id"], "draft_id": frame["draft_id"]}
            if self.state is not None and self.state["identity"] != identity:
                # These requests can no longer ACK against the current draft.
                self.pending.clear()
            self.state = {"type": "draft.state", "identity": identity, "available": frame["available"]}
            if self.session._ws is not None:
                await self.session._ws.send_json(self.state)
        elif frame.get("type") == "draft.result":
            request_id = frame.get("request_id")
            if not isinstance(request_id, str):
                return
            pending = self.pending.get(request_id)
            if pending is None:
                return
            request, browser = pending
            if (browser is self.session._ws and self.state is not None
                    and frame.get("identity") == request["expected"] == self.state["identity"]
                    and frame.get("status") in ("attached", "stale", "unavailable", "failed")):
                self.pending.pop(request_id)
                result = {key: frame[key] for key in ("type", "request_id", "identity", "status")}
                result.update({key: frame[key] for key in ("path", "label", "error") if isinstance(frame.get(key), str)})
                await browser.send_json(result)

    async def browser_frame(self, ws, raw):
        try:
            frame = json.loads(raw[len(DRAFT_CONTROL_PREFIX):])
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(frame, dict) or frame.get("type") != "draft.attach":
            return
        if self.session._ws is not ws or self.controller is None or self.state is None:
            return
        if not all(isinstance(frame.get(key), str) and frame[key] for key in ("request_id", "path")):
            return
        if not _valid_identity(frame.get("expected")):
            return
        if frame.get("expected") != self.state["identity"] or not self.state["available"]:
            await ws.send_json({"type": "draft.result", "request_id": frame["request_id"],
                               "identity": frame.get("expected"),
                               "status": "stale" if frame.get("expected") != self.state["identity"] else "unavailable"})
            return
        request = {key: frame[key] for key in ("type", "request_id", "expected", "path")}
        previous = self.pending.get(request["request_id"])
        if previous is not None and previous[0] != request:
            return
        self.pending[request["request_id"]] = (request, ws)
        controller = self.controller
        if not await self._send(controller, request):
            await self.disconnected(controller)
