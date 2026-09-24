"""Lane-private child process for ``_drive_gateway.drive_gateway``.

Runs the REAL messaging gateway entrypoint (``gateway.run.main`` — what
``python -m gateway.run`` / the ``hermes gateway run`` service executes: host
attach decision, PID claim, MCP discovery, ``GatewayRunner.start()``, signal
handlers, the graceful shutdown tail) with exactly one substitution: the
Telegram adapter instance is a recording fake. Everything behind the adapter
(authorization, session store, SessionDB, agent cache, AIAgent, hooks, plugins,
MCP) is production code.

The fake adapter injects ONE inbound DM through the adapter's normal
``handle_message`` path once the runner is running, prints every outbound
``send()`` and the ``on_processing_complete`` outcome as ``PARITY-GW <json>``
lines on stdout, and otherwise waits for the parent's normal stop
(planned-stop marker + SIGTERM, exactly like ``hermes gateway stop``).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult

import gateway.run as gateway_run

PARITY_USER_ID = os.environ.get("PARITY_GATEWAY_USER", "424242")
PARITY_CHAT_ID = PARITY_USER_ID  # Telegram private chats use the user id as chat id


# The gateway rebinds sys.stdout during startup; report on a private dup of the original pipe.
_REPORT = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")


def emit(kind: str, **payload: Any) -> None:
    _REPORT.write("PARITY-GW " + json.dumps({"kind": kind, **payload}) + "\n")
    _REPORT.flush()


class ParityTelegramAdapter(BasePlatformAdapter):
    """Telegram-shaped adapter: no network, records outbound text, injects one DM."""

    def __init__(self, config: Any, platform: Platform) -> None:
        super().__init__(config, platform)
        self._sent = 0
        self._inject_task: Optional[asyncio.Task] = None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._mark_connected()
        if self._inject_task is None:
            self._inject_task = asyncio.create_task(self._inject_once())
        return True

    async def _inject_once(self) -> None:
        # Inbound arriving while startup restore runs is queued and replayed later (the first
        # handle_message then "completes" without a turn); inject once the gateway is serving.
        runner = self.gateway_runner
        while not getattr(runner, "_running", False) or getattr(runner, "_startup_restore_in_progress", True):
            await asyncio.sleep(0.05)
        emit("ready")
        source = self.build_source(
            chat_id=PARITY_CHAT_ID, chat_name="parity", chat_type="dm",
            user_id=PARITY_USER_ID, user_name="parity-user", message_id="1",
        )
        event = MessageEvent(
            text=os.environ["PARITY_GATEWAY_PROMPT"], source=source, message_id="1",
            user_id=PARITY_USER_ID, user_name="parity-user",
        )
        await self.handle_message(event)
        emit("accepted", accepted=bool(getattr(event, "_gateway_accepted", False)))

    async def disconnect(self) -> None:
        if self._inject_task is not None and not self._inject_task.done():
            self._inject_task.cancel()
        self._mark_disconnected()

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        self._sent += 1
        message_id = f"parity-{self._sent}"
        emit("send", chat_id=str(chat_id), message_id=message_id, content=content)
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id: str, message_id: str, content: str, *,
                           finalize: bool = False) -> SendResult:
        emit("edit", chat_id=str(chat_id), message_id=message_id, content=content)
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "parity", "type": "dm", "chat_id": chat_id}

    async def on_processing_complete(self, event: MessageEvent, outcome: Any) -> None:
        emit("complete", outcome=str(getattr(outcome, "value", outcome)))


_real_instantiate = gateway_run.GatewayRunner._instantiate_adapter


def _instantiate_adapter(self, platform: Platform, config: Any):
    if platform == Platform.TELEGRAM:
        return ParityTelegramAdapter(config, platform)
    return _real_instantiate(self, platform, config)


def main() -> None:
    # The one seam: adapter construction (``GatewayRunner._create_adapter`` still wires runner +
    # handlers onto whatever this returns, exactly as for the real Telegram adapter).
    gateway_run.GatewayRunner._instantiate_adapter = _instantiate_adapter
    sys.argv = ["gateway.run"]
    gateway_run.main()


if __name__ == "__main__":
    main()
