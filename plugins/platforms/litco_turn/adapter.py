"""Gateway adapter that hosts the LitCo turn server inside the Hermes gateway process.

The adapter owns no inbound chat traffic of its own: LitKit drives turns over HTTP and the
turn server runs each one as an ``AIAgent`` directly (the same pattern the API-server adapter
uses), so structured events stream back instead of chat messages. ``send`` exists for gateway
features that deliver to a platform (cron ``deliver=litco_turn``); those deliveries are logged,
since LitKit reads results from the turn stream, not from pushes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from gateway.config import Platform
from gateway.platforms._shared import get_scoped_secret
from gateway.platforms.base import BasePlatformAdapter, SendResult

logger = logging.getLogger(__name__)

ENV_KEYS = ("LITCO_HOST_SECRET", "LITCO_MATTER_ID", "LITCO_MATTER_HOME", "LITCO_AGENT_TOKEN", "LITCO_INSTANCE_URL",
            "LITCO_TURN_HOST", "LITCO_TURN_PORT", "LITCO_SLOT_PORT")


class LitcoTurnAdapter(BasePlatformAdapter):
    def __init__(self, config, **_kwargs):
        super().__init__(config=config, platform=Platform("litco_turn"))
        self._extra = getattr(config, "extra", {}) or {}
        # Read in the profile scope at construction (multiplex-safe), never from os.environ later.
        self._env = {key: str(get_scoped_secret(key) or "") for key in ENV_KEYS}
        if not self._env["LITCO_MATTER_ID"] and self._extra.get("matter_id"):
            self._env["LITCO_MATTER_ID"] = str(self._extra["matter_id"])
        self._server = None

    @property
    def name(self) -> str:
        return "LitCo Turn Server"

    @property
    def authorization_is_upstream(self) -> bool:
        """Every request is authenticated by the host secret (and user assertion) in the server."""
        return True

    async def connect(self, **_kwargs) -> bool:
        from litco.hermes_runner import HermesTurnRunner
        from litco.turn_server import TurnServer, listen_address

        try:
            port = int(self._extra["port"]) if self._extra.get("port") else None
            host, port = listen_address(self._env, default_host=self._extra.get("host"), default_port=port)
        except ValueError as exc:
            self._set_fatal_error("config", str(exc), retryable=False)
            return False
        server = TurnServer(HermesTurnRunner(), env=self._env)
        if not server.host_secret:
            self._set_fatal_error("config", "LITCO_HOST_SECRET is not set", retryable=False)
            return False
        try:
            await server.start(host, port)
        except OSError as exc:
            logger.error("litco_turn: could not bind %s:%s: %s", host, port, exc)
            self._set_fatal_error("bind_failed", f"litco_turn bind failed: {exc}", retryable=True)
            return False
        self._server = server
        self._mark_connected()
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._server is not None:
            await self._server.stop()
            self._server = None

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        logger.info("litco_turn: out-of-band delivery for %s (%d chars) logged, not pushed",
                    chat_id, len(content or ""))
        return SendResult(success=True, message_id=None)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"litco:{chat_id}", "type": "channel", "chat_id": chat_id}
