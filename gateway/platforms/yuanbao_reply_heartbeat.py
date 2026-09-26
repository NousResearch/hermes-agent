"""Yuanbao reply-heartbeat lifecycle + slow-response notifier.

Split byte-verbatim out of ``gateway.platforms.yuanbao`` (2K-law shard, issue #79910):
the RUNNING/FINISH reply heartbeat that keeps a long turn alive on the client, and the
courtesy "please wait" notice sent when the agent stays silent past
``SLOW_RESPONSE_TIMEOUT_S``. The facade re-exports every public name, so
``gateway.platforms.yuanbao.<name>`` still resolves to the same objects.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict

from gateway.platforms.helpers import cancel_task
from gateway.platforms.yuanbao_proto import (
    WS_HEARTBEAT_FINISH, WS_HEARTBEAT_RUNNING,
    encode_send_group_heartbeat, encode_send_private_heartbeat,
)

logger = logging.getLogger("gateway.platforms.yuanbao")  # log-record parity with the origin module


REPLY_HEARTBEAT_INTERVAL_S = 2.0   # RUNNING cadence
REPLY_HEARTBEAT_TIMEOUT_S = 30.0   # auto-FINISH after this much inactivity
SLOW_RESPONSE_TIMEOUT_S = 120.0  # push SLOW_RESPONSE_MESSAGE when the agent is silent this long
SLOW_RESPONSE_MESSAGE = "任务有点复杂，正在努力处理中，请耐心等待..."


def _cancel_all(tasks: Dict[str, asyncio.Task]) -> None:
    """Cancel every unfinished task in *tasks* and clear the dict."""
    for task in list(tasks.values()):
        if not task.done():
            task.cancel()
    tasks.clear()


class HeartbeatManager:
    """Reply heartbeat lifecycle: RUNNING every 2s, auto-FINISH after 30s idle, explicit stop."""
    def __init__(self, adapter: "YuanbaoAdapter") -> None:
        self._adapter = adapter
        self._reply_heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self._reply_hb_last_active: Dict[str, float] = {}

    def _ready(self) -> bool:
        return self._adapter._connection.ws is not None and bool(self._adapter._bot_id)

    async def send_heartbeat_once(self, chat_id: str, heartbeat_val: int) -> None:
        """Send a single heartbeat (RUNNING or FINISH), best effort."""
        adapter = self._adapter
        if not self._ready():
            return
        try:
            if chat_id.startswith("group:"):
                encoded = encode_send_group_heartbeat(from_account=adapter._bot_id, group_code=chat_id[len("group:"):], heartbeat=heartbeat_val)
            else:
                encoded = encode_send_private_heartbeat(from_account=adapter._bot_id, to_account=chat_id.removeprefix("direct:"), heartbeat=heartbeat_val)
            await adapter._connection.ws.send(encoded)
            logger.debug("[%s] Reply heartbeat %s sent: chat=%s", adapter.name,
                         "RUNNING" if heartbeat_val == WS_HEARTBEAT_RUNNING else "FINISH", chat_id)
        except Exception as exc:
            logger.debug("[%s] send_heartbeat_once failed: %s", adapter.name, exc)

    async def start(self, chat_id: str) -> None:
        """Start or renew the periodic RUNNING sender."""
        if not self._ready():
            return
        self._reply_hb_last_active[chat_id] = time.time()
        existing = self._reply_heartbeat_tasks.get(chat_id)
        if not existing or existing.done():
            self._reply_heartbeat_tasks[chat_id] = asyncio.create_task(self._worker(chat_id), name=f"yuanbao-reply-hb-{chat_id}")

    async def _worker(self, chat_id: str) -> None:
        """Send RUNNING every 2s; after 30s without renewal (or WS loss) send FINISH and exit.
        A cancelled worker sends no FINISH — stop() decides that."""
        cancelled = False
        try:
            await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_RUNNING)
            while True:
                await asyncio.sleep(REPLY_HEARTBEAT_INTERVAL_S)
                if (time.time() - self._reply_hb_last_active.get(chat_id, 0) > REPLY_HEARTBEAT_TIMEOUT_S
                        or self._adapter._connection.ws is None):
                    break
                await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_RUNNING)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            pass
        finally:
            if not cancelled:
                await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_FINISH)
            self._reply_heartbeat_tasks.pop(chat_id, None)
            self._reply_hb_last_active.pop(chat_id, None)

    async def stop(self, chat_id: str, send_finish: bool = True) -> None:
        """Stop the RUNNING sender and optionally send FINISH."""
        task = self._reply_heartbeat_tasks.pop(chat_id, None)
        if task and not task.done():
            await cancel_task(task)
        if send_finish:
            await self.send_heartbeat_once(chat_id, WS_HEARTBEAT_FINISH)

    async def close(self) -> None:
        _cancel_all(self._reply_heartbeat_tasks)
        self._reply_hb_last_active.clear()


class SlowResponseNotifier:
    """Per-chat timer that sends a courtesy 'please wait' after SLOW_RESPONSE_TIMEOUT_S without a reply."""
    def __init__(self, adapter: "YuanbaoAdapter", sender: "MessageSender") -> None:
        self._adapter = adapter
        self._sender = sender
        self._tasks: Dict[str, asyncio.Task] = {}

    async def start(self, chat_id: str) -> None:
        self.cancel(chat_id)
        self._tasks[chat_id] = asyncio.create_task(self._notifier(chat_id), name=f"yuanbao-slow-resp-{chat_id}")

    async def _notifier(self, chat_id: str) -> None:
        try:
            await asyncio.sleep(SLOW_RESPONSE_TIMEOUT_S)
            logger.info("[%s] Agent response exceeded %ds for %s, sending wait notice", self._adapter.name, int(SLOW_RESPONSE_TIMEOUT_S), chat_id)
            await self._sender.send_text_chunk(chat_id, SLOW_RESPONSE_MESSAGE)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("[%s] Slow-response notifier failed: %s", self._adapter.name, exc)

    def cancel(self, chat_id: str) -> None:
        task = self._tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def close(self) -> None:
        _cancel_all(self._tasks)
