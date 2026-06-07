"""Dashboard plugin API — WebSocket endpoint for live-glass events.

Mounts under ``/api/plugins/live-glass/``.  Clients open a WebSocket at
``/api/plugins/live-glass/events`` and receive a JSON stream of live-glass
events plus periodic heartbeats.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

_HEARTBEAT_SECONDS = float(os.environ.get("HERMES_LIVE_GLASS_WS_HEARTBEAT", "30"))


@router.websocket("/events")
async def websocket_events(ws: WebSocket) -> None:
    """Stream live-glass events to a WebSocket client."""
    await ws.accept()

    # Replay the last frame so the client has an immediate image.
    _replay_last_frame(ws)

    # Build a send-queue that pushes events to the WebSocket.
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def _on_event(event: dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop on backpressure; client is too slow.

    from plugins.observability.live_glass import subscribe
    unsubscribe = subscribe(_on_event)

    heartbeat_task = asyncio.create_task(_heartbeat_loop(queue))

    try:
        while True:
            event = await queue.get()
            if event is None:
                break  # Sentinel from heartbeat on failure.
            try:
                await ws.send_json(event)
            except Exception:
                logger.debug("live-glass WS: send failed, closing", exc_info=True)
                break
    except WebSocketDisconnect:
        logger.debug("live-glass WS: client disconnected")
    except Exception:
        logger.debug("live-glass WS: error", exc_info=True)
    finally:
        unsubscribe()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


def _replay_last_frame(ws: WebSocket) -> None:
    """Send the most recent frame event, if any, without awaiting."""
    try:
        from plugins.observability.live_glass import get_events
        frames = get_events(event_type="frame", limit=1)
        if frames:
            asyncio.create_task(ws.send_json(frames[0]))
    except Exception:
        logger.debug("live-glass WS: replay failed", exc_info=True)


async def _heartbeat_loop(queue: asyncio.Queue) -> None:
    """Push a heartbeat message onto *queue* every ``_HEARTBEAT_SECONDS``."""
    while True:
        await asyncio.sleep(_HEARTBEAT_SECONDS)
        try:
            queue.put_nowait({"type": "heartbeat"})
        except asyncio.QueueFull:
            pass
        except Exception:
            logger.debug("live-glass WS: heartbeat queue closed")
            break
