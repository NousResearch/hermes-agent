"""
Optional long-running EVM log listener over WebSocket JSON-RPC.

Writes normalized rows via EventQueue.enqueue — Hermes does not auto-start
subagents; poll ``dequeue_events`` from the agent or an external worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def run_ws_log_listener(
    *,
    ws_url: str,
    address: str,
    topics: Optional[List[Any]],
    queue_cb,
    max_retries: int = 8,
) -> None:
    """Subscribe to ``logs`` on a WS endpoint; reconnect with exponential backoff."""
    try:
        import websockets
    except ImportError:
        raise RuntimeError("websockets package required for WS listener") from None

    delay = 1.0
    attempt = 0
    sub_id: Optional[str] = None
    while attempt < max_retries:
        try:
            async with websockets.connect(ws_url, ping_interval=20) as ws:
                delay, attempt = 1.0, 0
                req = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": [
                        "logs",
                        {"address": address, "topics": topics or []},
                    ],
                }
                await ws.send(json.dumps(req))
                msg = json.loads(await ws.recv())
                sub_id = msg.get("result")
                if not sub_id:
                    raise RuntimeError(f"subscribe failed: {msg}")
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    params = data.get("params") or {}
                    if params.get("subscription") == sub_id:
                        queue_cb(
                            "evm",
                            {"source": "ws", "log": params.get("result")},
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            logger.warning("ws listener error (attempt %s): %s", attempt, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
            await asyncio.sleep(0)
    logger.error("ws listener exceeded max_retries=%s", max_retries)


def main() -> None:
    """CLI entry: ``python ws_listener.py`` reads env ``WEB3_WS_URL``, ``WEB3_LOG_ADDRESS``."""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from event_queue import EventQueue  # noqa: WPS433 — script path bootstrap

    ws_url = os.environ.get("WEB3_WS_URL", "")
    addr = os.environ.get("WEB3_LOG_ADDRESS", "")
    if not ws_url or not addr:
        raise SystemExit("WEB3_WS_URL and WEB3_LOG_ADDRESS required")
    q = EventQueue()

    def cb(chain: str, payload: Dict[str, Any]) -> None:
        q.enqueue(chain, payload)

    asyncio.run(run_ws_log_listener(ws_url=ws_url, address=addr, topics=None, queue_cb=cb))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
