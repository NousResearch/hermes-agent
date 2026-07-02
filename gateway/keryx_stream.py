"""Keryx side-channel stream hub — the server half of Keryx's dual-tier streaming.

Installed as ``gateway/keryx_stream.py`` inside the hermes-agent tree (see install.py; this is a
REINSTALL-FRAGILE patch — re-run install.py after ``hermes update``).

What it does
============
The Keryx Android client opens a transient SSE subscription (``GET /keryx/stream?platform=matrix&
chat_id=<room>`` on the API server, Bearer-authed with API_SERVER_KEY) right before sending a
command into a Matrix room. While that subscriber is attached:

  * every assistant-text delta the ``GatewayStreamConsumer`` receives is mirrored to the SSE
    channel (``event: delta``) for live token rendering in the app;
  * protocol edits to the homeserver are suppressed — the room receives only the single final
    committed message (no m.replace database bloat);
  * ``event: stop`` fires when the turn's stream finishes, telling the client to hold its overlay
    until the final Matrix event syncs in.

When no subscriber is attached and ``FALLBACK_EDITS`` is True, Matrix falls back to
smart-throttled native m.replace edits driven by the normal streaming config
(``streaming.edit_interval`` / ``streaming.buffer_threshold`` — tune to 1.2s / 60 in config.yaml).
Set ``FALLBACK_EDITS = False`` to restore final-message-only behaviour when Keryx is offline.

Thread-safety: ``publish_threadsafe`` is called from the agent's sync worker thread; delivery hops
onto each subscriber's event loop via ``call_soon_threadsafe``. Queues are bounded — a stalled
subscriber drops its own events, never blocks the agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gateway.keryx_stream")

# Opt-in fallback tier: when set (KERYX_STREAM_FALLBACK_EDITS=1), a Matrix chat WITHOUT a live
# side-channel subscriber gets throttled protocol (m.replace) edit streaming instead of the
# buffer-only default. Off by default so this module changes no existing gateway behaviour.
FALLBACK_EDITS = os.getenv("KERYX_STREAM_FALLBACK_EDITS", "").strip().lower() in {"1", "true", "yes", "on"}

# Per-subscriber event buffer. Generous relative to token rate x ping interval; overflow drops
# oldest-first semantics are approximated by dropping the incoming event for that subscriber.
_QUEUE_MAX = 2048


class _Subscription:
    __slots__ = ("queue", "loop")

    def __init__(self, queue: "asyncio.Queue[Tuple[str, Optional[str]]]", loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop


class KeryxStreamHub:
    """In-process pub/sub keyed by (platform, chat_id)."""

    def __init__(self) -> None:
        self._subs: Dict[Tuple[str, str], List[_Subscription]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(platform: str, chat_id: str) -> Tuple[str, str]:
        return (str(platform).strip().lower(), str(chat_id).strip())

    def subscribe(self, platform: str, chat_id: str) -> _Subscription:
        sub = _Subscription(asyncio.Queue(maxsize=_QUEUE_MAX), asyncio.get_running_loop())
        key = self._key(platform, chat_id)
        with self._lock:
            self._subs.setdefault(key, []).append(sub)
        logger.info("keryx subscriber attached: %s", key)
        return sub

    def unsubscribe(self, platform: str, chat_id: str, sub: _Subscription) -> None:
        key = self._key(platform, chat_id)
        with self._lock:
            lst = self._subs.get(key)
            if lst and sub in lst:
                lst.remove(sub)
                if not lst:
                    del self._subs[key]
        logger.info("keryx subscriber detached: %s", key)

    def has_subscribers(self, platform: str, chat_id: str) -> bool:
        with self._lock:
            return bool(self._subs.get(self._key(platform, chat_id)))

    def publish_threadsafe(self, platform: str, chat_id: str, event: str, text: Optional[str]) -> None:
        """Mirror one stream event to every subscriber. Never raises, never blocks."""
        key = self._key(platform, chat_id)
        with self._lock:
            subs = list(self._subs.get(key, ()))
        for sub in subs:
            try:
                sub.loop.call_soon_threadsafe(self._offer, sub.queue, (event, text))
            except Exception:
                # Subscriber's loop is gone — it will be pruned when its handler exits.
                pass

    @staticmethod
    def _offer(queue: "asyncio.Queue[Tuple[str, Optional[str]]]", item: Tuple[str, Optional[str]]) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.debug("keryx subscriber queue full; dropping %s", item[0])


hub = KeryxStreamHub()


def _platform_of(adapter: Any) -> str:
    """Stable lowercase platform key for an adapter ("matrix", "telegram", …)."""
    try:
        return str(adapter.platform.value).lower()
    except Exception:
        return str(getattr(adapter, "name", "")).lower()


def publish_delta(adapter: Any, chat_id: Any, text: str) -> None:
    """Called from GatewayStreamConsumer.on_delta (agent worker thread)."""
    hub.publish_threadsafe(_platform_of(adapter), str(chat_id), "delta", text)


def publish_segment(adapter: Any, chat_id: Any) -> None:
    hub.publish_threadsafe(_platform_of(adapter), str(chat_id), "segment", None)


def publish_stop(adapter: Any, chat_id: Any, final_text: Optional[str] = None) -> None:
    hub.publish_threadsafe(_platform_of(adapter), str(chat_id), "stop", final_text)


def suppress_protocol_edits(adapter: Any, chat_id: Any, default_buffer_only: bool) -> bool:
    """Decide whether the stream consumer should skip interval/threshold homeserver edits.

    Live Keryx subscriber → True (the side-channel carries tokens; commit only the final).
    No subscriber on Matrix with FALLBACK_EDITS → False (throttled m.replace fallback tier).
    Anything else → whatever the gateway decided ([default_buffer_only]).
    """
    platform = _platform_of(adapter)
    if hub.has_subscribers(platform, str(chat_id)):
        return True
    if default_buffer_only and FALLBACK_EDITS and platform == "matrix":
        return False
    return default_buffer_only


def make_stream_handler(check_auth):
    """Build the aiohttp handler for ``GET /keryx/stream`` (wired in api_server.py).

    [check_auth] is ApiServerAdapter._check_auth — same Bearer key as every other route.
    """
    from aiohttp import web

    async def handle_keryx_stream(request: "web.Request") -> "web.StreamResponse":
        auth_err = check_auth(request)
        if auth_err is not None:
            return auth_err
        platform = request.query.get("platform", "matrix")
        chat_id = request.query.get("chat_id", "").strip()
        if not chat_id:
            return web.json_response({"error": {"message": "chat_id is required"}}, status=400)

        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await resp.prepare(request)
        sub = hub.subscribe(platform, chat_id)
        try:
            while True:
                try:
                    event, text = await asyncio.wait_for(sub.queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    # Keepalive: keeps NATs open and lets a dead client surface as a write error.
                    await resp.write(b"event: ping\ndata: {}\n\n")
                    continue
                payload = json.dumps({"text": text} if text is not None else {})
                await resp.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
                if event == "stop":
                    break  # transient channel: one turn per subscription
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            hub.unsubscribe(platform, chat_id, sub)
        try:
            await resp.write_eof()
        except Exception:
            pass
        return resp

    return handle_keryx_stream
