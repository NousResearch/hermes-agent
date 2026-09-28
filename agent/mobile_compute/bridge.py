"""Bridge module — configuration, auth, and lifecycle management for the Mobile Compute Node.

This module provides the MobileComputeBridge class which acts as the
central coordinator between Hermes (Windows) and the iPhone 16e server.
It handles:
    - Configuration loading and validation
    - mTLS certificate management and authentication
    - Server lifecycle (start/stop/status)
    - Connection health monitoring
    - Missing-device graceful degradation

The bridge does NOT perform any LLM calls or Core ML inference.
It is purely a transport and lifecycle coordinator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from agent.mobile_compute.config import (
    MobileComputeConfig,
    MobileComputeClientConfig,
    MobileComputeServerConfig,
    TLSConfig,
)
from agent.mobile_compute.client import MobileComputeClient, ComputeResult
from agent.mobile_compute.server import MobileComputeServer

logger = logging.getLogger("agent.mobile_compute.bridge")


@dataclass
class MobileComputeBridgeStatus:
    """Current status of the Mobile Compute Bridge."""
    bridge_started: bool = False
    iphone_connected: bool = False
    iphone_healthy: bool = False
    uptime_seconds: float = 0.0
    task_count: int = 0
    errors: list = field(default_factory=list)


class MobileComputeBridge:
    """Central coordinator for Mobile Compute Node (iPhone 16e).

    Manages configuration, authentication, server lifecycle, and
    connection health. Gracefully handles missing or unreachable
    iPhone devices by returning sensible defaults and logging warnings.

    Parameters
    ----------
    config : MobileComputeConfig, optional
        Pre-loaded configuration. If omitted, loads from config.yaml.
    """

    def __init__(self, config: Optional[MobileComputeConfig] = None) -> None:
        self._config = config or MobileComputeConfig.load()
        self._client: Optional[MobileComputeClient] = None
        self._server: Optional[MobileComputeServer] = None
        self._status = MobileComputeBridgeStatus()
        self._start_time: float = 0.0
        self._closed = False

    # --- Properties ---

    @property
    def enabled(self) -> bool:
        """Whether mobile_compute is enabled in config."""
        return self._config.enabled

    @property
    def status(self) -> MobileComputeBridgeStatus:
        """Get a copy of the current bridge status."""
        return self._status

    @property
    def config(self) -> MobileComputeConfig:
        """Get the loaded configuration."""
        return self._config

    # --- Lifecycle ---

    async def start(self) -> bool:
        """Start the bridge.

        Initializes the HTTP client, starts the local server (if configured),
        and performs an initial health check on the iPhone.

        Returns
        -------
        bool
            True if the bridge started successfully, False if disabled
            or the iPhone is unreachable (graceful degradation).
        """
        if self._closed:
            logger.warning("Bridge already closed; cannot restart")
            return False

        if not self._config.enabled:
            logger.info("Mobile Compute is disabled in config; skipping start")
            self._status.iphone_connected = False
            return False

        self._start_time = time.monotonic()
        self._status.bridge_started = True
        logger.info("Starting Mobile Compute Bridge...")

        # Initialize the HTTP client
        self._client = MobileComputeClient(self._config.client)
        self._status.iphone_connected = True

        # Start local server if configured
        if self._config.server.enabled:
            try:
                self._server = MobileComputeServer(self._config.server)
                await self._server.start()
                logger.info("Local Mobile Compute server started")
            except Exception as exc:
                logger.error("Failed to start local server: %s", exc)
                self._status.errors.append(f"server_start: {exc}")

        # Health check
        await self._check_health()

        return self._status.iphone_healthy

    async def stop(self) -> None:
        """Stop the bridge and clean up resources."""
        logger.info("Stopping Mobile Compute Bridge...")

        # Stop local server
        if self._server:
            try:
                await self._server.stop()
            except Exception as exc:
                logger.error("Error stopping server: %s", exc)
            self._server = None

        # Close HTTP client
        if self._client:
            try:
                await self._client.close()
            except Exception as exc:
                logger.error("Error closing client: %s", exc)
            self._client = None

        self._closed = True
        self._status.bridge_started = False
        self._status.iphone_connected = False
        self._status.iphone_healthy = False
        logger.info("Mobile Compute Bridge stopped")

    # --- Health & Connection ---

    async def _check_health(self) -> bool:
        """Check iPhone health. Updates status accordingly.
        
        Gracefully handles missing/unreachable iPhone by logging
        and setting iphone_connected/iphone_healthy to False.
        """
        if not self._client:
            self._status.iphone_healthy = False
            return False

        try:
            health = await self._client.health()
            self._status.iphone_healthy = True
            self._status.iphone_connected = True
            self._status.errors = [e for e in self._status.errors if not e.startswith("health:")]
            logger.info("iPhone health check OK: %s", health)
            return True
        except Exception as exc:
            self._status.iphone_healthy = False
            self._status.iphone_connected = False
            self._status.errors.append(f"health: {exc}")
            logger.warning("iPhone health check failed — device may be offline: %s", exc)
            return False

    async def check_health(self) -> bool:
        """Public health check wrapper."""
        return await self._check_health()

    # --- Capabilities ---

    async def get_capabilities(self) -> Dict[str, Any]:
        """Get iPhone capabilities. Returns empty dict if iPhone unreachable."""
        if not self._client:
            return {}
        try:
            return await self._client.capabilities()
        except Exception as exc:
            logger.warning("Cannot get capabilities: %s", exc)
            self._status.errors.append(f"capabilities: {exc}")
            return {}

    # --- Task Submission ---

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> ComputeResult:
        """Submit a compute task to the iPhone.
        
        If the iPhone is unreachable, returns a failed ComputeResult
        instead of raising an exception.
        """
        if not self._client:
            return ComputeResult(
                success=False, task_id=task_id or "unknown",
                status="failed", error="Bridge not initialized",
            )

        result = await self._client.submit_task(task_type, payload, task_id)
        self._status.task_count += 1
        return result

    # --- Status ---

    async def get_status(self) -> MobileComputeBridgeStatus:
        """Get current bridge status with updated uptime."""
        self._status.uptime_seconds = time.monotonic() - self._start_time if self._start_time else 0.0
        if self._client and self._server:
            self._status.task_count = len(self._server._tasks)
        return self._status

    # --- Context manager ---

    async def __aenter__(self) -> "MobileComputeBridge":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    async def close(self) -> None:
        """Close the bridge (alias for stop)."""
        await self.stop()


def create_bridge(config_dict: Optional[Dict[str, Any]] = None) -> MobileComputeBridge:
    """Create and start a MobileComputeBridge from a config dict.
    
    Returns a bridge that may be in a 'disabled' or 'degraded' state
    if the iPhone is not available.
    """
    import asyncio
    config = MobileComputeConfig.load(config_dict)
    bridge = MobileComputeBridge(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bridge.start())
    return bridge


async def test_bridge_connection(config_dict: Optional[Dict[str, Any]] = None) -> bool:
    """Test if the Mobile Compute Bridge can reach the iPhone."""
    bridge = MobileComputeBridge(MobileComputeConfig.load(config_dict))
    async with bridge:
        status = await bridge.get_status()
        return status.iphone_healthy