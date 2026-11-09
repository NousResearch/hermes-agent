"""Windows-side client for Mobile Compute Node (iPhone 16e).

This client runs on Windows (Hermes) and communicates with the iPhone server
via HTTP. It's designed to be used by the main Hermes agent logic
without requiring iOS-specific dependencies.

Features:
- Async HTTP client using httpx (already a dependency)
- mTLS support (if configured)
- Automatic retries with exponential backoff
- Simple task submission API
- Health check and capability discovery
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from agent.mobile_compute.config import MobileComputeConfig, MobileComputeClientConfig
from agent.mobile_compute.client import (
    MobileComputeClient,
    ComputeResult,
)
from agent.mobile_compute.bridge import MobileComputeBridge

logger = logging.getLogger("agent.mobile_compute.client_windows")


class MobileComputeClientWindows:
    """Synchronous wrapper around the async MobileComputeClient for Windows use.

    This class provides a blocking interface that can be used in synchronous
    contexts (e.g., during Hermes startup) by running the async methods in
    an event loop.
    """

    def __init__(self, config: MobileComputeClientConfig) -> None:
        self._config = config
        self._client: Optional[MobileComputeClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Create or reuse an event loop for synchronous calls."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def health(self) -> Dict[str, Any]:
        """Check iPhone health. Returns parsed JSON from /health."""
        loop = self._ensure_loop()
        return loop.run_until_complete(self._client.health())

    def capabilities(self) -> Dict[str, Any]:
        """Get iPhone capabilities. Returns parsed JSON from /capabilities."""
        loop = self._ensure_loop()
        return loop.run_until_complete(self._client.capabilities())

    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> ComputeResult:
        """Submit a compute task to the iPhone."""
        loop = self._ensure_loop()
        return loop.run_until_complete(
            self._client.submit_task(task_type, payload, task_id)
        )

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Check status of a previously submitted task."""
        loop = self._ensure_loop()
        return loop.run_until_complete(self._client.get_task_status(task_id))

    def list_tasks(self) -> Dict[str, Any]:
        """List all tasks."""
        loop = self._ensure_loop()
        return loop.run_until_complete(self._client.list_tasks())

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.close()

    def __enter__(self) -> "MobileComputeClientWindows":
        loop = self._ensure_loop()
        self._client = MobileComputeClient(self._config)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._client:
            loop = self._ensure_loop()
            asyncio.run_coroutine_threadsafe(self._client.close(), loop)
        if self._loop:
            self._loop.close()

    async def test_connection(self) -> bool:
        """Test if the iPhone server is reachable. Returns True on success."""
        loop = self._ensure_loop()
        return await test_client_connection(self._config)


def create_client_from_config(
    config_dict: Optional[Dict[str, Any]] = None,
) -> MobileComputeClientWindows:
    """Create a Windows client from config (or defaults)."""
    config = MobileComputeConfig.load(config_dict)
    return MobileComputeClientWindows(config.client)


def create_bridge_from_config(
    config_dict: Optional[Dict[str, Any]] = None,
) -> MobileComputeBridge:
    """Create a MobileComputeBridge from config (or defaults)."""
    config = MobileComputeConfig.load(config_dict)
    return MobileComputeBridge(config)