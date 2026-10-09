"""In-memory messaging platform for the gateway chaos suite, loaded as a REAL user plugin.

The chaos test writes ``$HERMES_HOME/plugins/chaos-fake/`` (``kind: platform``) whose
``register`` is this module's, enables it in config.yaml, and runs the real
``python -m gateway.run`` entry point. The gateway then discovers the plugin, builds this
adapter through the platform registry, connects it and installs its own message handler
— the same path every third-party adapter takes. Nothing in the runner is patched.

Wire to the test process: one loopback TCP connection (port in ``CHAOS_GW_PORT``)
carrying JSON lines.

  test -> adapter   {"op": "msg", "text": ..., "chat": ..., "id": ...}
  adapter -> test   {"ev": "ready"} once connected
                    {"ev": "send", "chat": ..., "text": ...} for every outbound send/edit
                    {"ev": "hb", "max_gap": s, "ticks": n} every ~0.5 s: the largest gap
                    between 100 ms event-loop ticks since the previous report (a sync call
                    blocking the gateway loop shows up here, not in the test's own timing)
                    {"ev": "disconnect"} from the real shutdown path
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType

PLATFORM_NAME = "chaosfake"
PLUGIN_DIR_NAME = "chaos-fake"
ALLOW_ALL_ENV = "CHAOS_FAKE_ALLOW_ALL_USERS"
PORT_ENV = "CHAOS_GW_PORT"
HEARTBEAT_TICK_S = 0.1
HEARTBEAT_REPORT_S = 0.5


class ChaosFakeAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform(PLATFORM_NAME))
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._tasks: list[asyncio.Task] = []
        self._seq = 0

    # -- wire ----------------------------------------------------------------

    def _emit(self, payload: Dict[str, Any]) -> None:
        writer = self._writer
        if writer is None or writer.is_closing():
            return
        payload["t"] = time.time()
        writer.write((json.dumps(payload) + "\n").encode())

    async def _heartbeat(self) -> None:
        last = time.monotonic()
        window_max = 0.0
        last_report = last
        ticks = 0
        while True:
            await asyncio.sleep(HEARTBEAT_TICK_S)
            now = time.monotonic()
            window_max = max(window_max, now - last)
            last = now
            ticks += 1
            if now - last_report >= HEARTBEAT_REPORT_S:
                self._emit({"ev": "hb", "max_gap": round(window_max, 3), "ticks": ticks,
                            "busy": sorted(self._active_sessions)})
                window_max = 0.0
                last_report = now

    async def _read_commands(self) -> None:
        assert self._reader is not None
        while True:
            line = await self._reader.readline()
            if not line:
                return
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cmd.get("op") == "msg":
                await self._inbound(cmd)

    async def _inbound(self, cmd: Dict[str, Any]) -> None:
        chat = str(cmd.get("chat") or "chaos-chat")
        msg_id = str(cmd.get("id") or uuid.uuid4().hex)
        source = self.build_source(
            chat_id=chat, chat_name=chat, chat_type="dm",
            user_id="chaos-user", user_name="chaos-user", message_id=msg_id)
        event = MessageEvent(
            text=str(cmd.get("text") or ""), message_type=MessageType.TEXT,
            source=source, message_id=msg_id)
        await self.handle_message(event)

    # -- BasePlatformAdapter -------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        port = int(os.environ[PORT_ENV])
        self._reader, self._writer = await asyncio.open_connection("127.0.0.1", port)
        self._tasks = [
            asyncio.create_task(self._heartbeat(), name="chaos-heartbeat"),
            asyncio.create_task(self._read_commands(), name="chaos-inbound"),
        ]
        self._mark_connected()
        self._emit({"ev": "ready", "pid": os.getpid()})
        return True

    async def disconnect(self) -> None:
        self._emit({"ev": "disconnect"})
        self._mark_disconnected()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        if self._writer is not None:
            try:
                await self._writer.drain()
                self._writer.close()
            except Exception:
                pass
            self._writer = None

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        self._seq += 1
        message_id = f"chaos-out-{self._seq}"
        self._emit({"ev": "send", "chat": str(chat_id), "text": content, "id": message_id})
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id: str, message_id: str, content: str, **_kw: Any) -> SendResult:
        self._emit({"ev": "send", "chat": str(chat_id), "text": content, "id": message_id, "edit": True})
        return SendResult(success=True, message_id=message_id)

    async def on_processing_start(self, event: MessageEvent) -> None:
        self._emit({"ev": "start", "id": event.message_id})

    async def on_processing_complete(self, event: MessageEvent, outcome: Any) -> None:
        self._emit({"ev": "complete", "id": event.message_id, "outcome": str(outcome)})

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}


def register(ctx) -> None:
    """Plugin entry point (the chaos test writes a plugin dir importing this)."""
    ctx.register_platform(
        name=PLATFORM_NAME,
        label="Chaos Fake",
        adapter_factory=ChaosFakeAdapter,
        check_fn=lambda: True,
        validate_config=lambda _cfg: True,
        is_connected=lambda _cfg: True,
        allow_all_env=ALLOW_ALL_ENV,
    )
