"""
Fluxer platform adapter stub.

Fluxer is an open-source Discord alternative (https://github.com/fluxer).
This adapter provides the skeleton for connecting Hermes to a Fluxer
instance via WebSocket gateway (real-time) and REST API (cron/standalone).

Environment variables:
    FLUXER_BOT_TOKEN          Bot authentication token
    FLUXER_API_URL            Base URL of the Fluxer API (e.g. https://fluxer.example.com)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Ensure Hermes core is importable when loaded as a bundled plugin
# ---------------------------------------------------------------------------
_HERMES_ROOT = Path(__file__).resolve().parents[3]
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# ---- Optional dependency guards -------------------------------------------

try:
    import websockets  # noqa: F401

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore[assignment]

try:
    import httpx  # noqa: F401

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

# ---- Class stubs ----------------------------------------------------------


class FluxerGatewayClient:
    """Manages a WebSocket connection to the Fluxer gateway.

    Responsible for:
    - Establishing and maintaining the WebSocket connection
    - Receiving real-time events (messages, presence, etc.)
    - Heartbeat / keep-alive handling
    - Reconnection with exponential backoff
    """

    def __init__(self, token: str, api_url: str) -> None:
        self.token = token
        self.api_url = api_url.rstrip("/")
        self._connected = False

    async def connect(self) -> bool:
        """Open the WebSocket connection and authenticate."""
        raise NotImplementedError  # pragma: no cover

    async def disconnect(self) -> None:
        """Close the WebSocket connection gracefully."""
        raise NotImplementedError  # pragma: no cover

    @property
    def is_connected(self) -> bool:
        return self._connected


class FluxerRESTClient:
    """HTTP client for the Fluxer REST API.

    Used for out-of-process operations such as:
    - Sending messages from cron jobs (standalone sender)
    - Channel / guild metadata queries
    - File uploads
    """

    def __init__(self, token: str, api_url: str) -> None:
        self.token = token
        self.api_url = api_url.rstrip("/")

    async def send_message(
        self, channel_id: str, content: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Post a message to a Fluxer channel via REST."""
        raise NotImplementedError  # pragma: no cover


class FluxerAdapter(BasePlatformAdapter):
    """Platform adapter for the Fluxer messaging platform.

    Integrates Hermes with a Fluxer instance through a WebSocket gateway
    (real-time messaging) and REST API (standalone / cron delivery).
    """

    supports_code_blocks: bool = True
    supports_async_delivery: bool = True
    splits_long_messages: bool = True
    typed_command_prefix: str = "/"

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config)
        self._token: str = os.environ.get("FLUXER_BOT_TOKEN", "")
        self._api_url: str = os.environ.get(
            "FLUXER_API_URL", "http://localhost:8090"
        )
        self._gateway: Optional[FluxerGatewayClient] = None
        self._rest: Optional[FluxerRESTClient] = None

    # ---- BasePlatformAdapter abstract methods -----------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the Fluxer instance.

        Initialises the gateway client and opens the WebSocket connection.
        On reconnect (is_reconnect=True), applies backoff / state recovery.
        """
        self._gateway = FluxerGatewayClient(self._token, self._api_url)
        self._rest = FluxerRESTClient(self._token, self._api_url)
        connected = await self._gateway.connect()
        if connected:
            logger.info(
                "Fluxer adapter %s to %s",
                "reconnected" if is_reconnect else "connected",
                self._api_url,
            )
        return connected

    async def disconnect(self) -> None:
        """Disconnect from the Fluxer instance."""
        if self._gateway is not None:
            await self._gateway.disconnect()
            self._gateway = None

    async def send(
        self,
        message: str,
        channel_id: str,
        message_type: MessageType = MessageType.TEXT,
        event: Optional[MessageEvent] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a message to a Fluxer channel.

        Attempts gateway (WebSocket) delivery first; falls back to REST
        if the gateway is not connected.
        """
        raise NotImplementedError  # pragma: no cover

    async def send_media(
        self,
        file_path: str,
        channel_id: str,
        caption: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Upload and send a media file to a Fluxer channel."""
        raise NotImplementedError  # pragma: no cover

    def is_alive(self) -> bool:
        """Return whether the adapter is currently connected."""
        return self._gateway is not None and self._gateway.is_connected

    def get_me(self) -> Optional[Dict[str, Any]]:
        """Return the bot user's identity metadata, or None if unknown."""
        return None  # pragma: no cover


# ---- Requirements check ---------------------------------------------------


def check_fluxer_requirements() -> bool:
    """Return True when all required dependencies are available.

    Must be silent (no WARNING-level logging) since it is called
    frequently during config loading.
    """
    return WEBSOCKETS_AVAILABLE and HTTPX_AVAILABLE


def _is_connected(config: PlatformConfig) -> bool:
    """Return whether the Fluxer adapter appears configured and connected."""
    token = os.environ.get("FLUXER_BOT_TOKEN", "")
    api_url = os.environ.get("FLUXER_API_URL", "")
    return bool(token) and bool(api_url)


# ---- Plugin entry point ---------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="fluxer",
        label="Fluxer",
        adapter_factory=lambda cfg: FluxerAdapter(cfg),
        check_fn=check_fluxer_requirements,
        is_connected=_is_connected,
        required_env=["FLUXER_BOT_TOKEN", "FLUXER_API_URL"],
        install_hint="pip install 'hermes-agent[fluxer]'",
        emoji="💬",
    )
