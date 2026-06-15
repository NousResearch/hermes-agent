"""
Adapter Manager Service — Centralises adapter lifecycle management.

Migrated from gateway/run.py GatewayRunner.
Responsibility: Adapter lifecycle (create, connect, disconnect, reconnect, status).
Does NOT handle message routing or agent execution — those stay in gateway/.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gateway.platforms.api_server import BasePlatformAdapter

logger = logging.getLogger(__name__)


class BaseAdapterManager(ABC):
    """
    Abstract adapter lifecycle manager.

    Defines standard interface for:
    - Adapter creation & initialization
    - Connection lifecycle (connect, disconnect, reconnect)
    - Status monitoring
    - Resource cleanup (disposal)
    """

    @abstractmethod
    async def create_adapter(self, platform: Any, config: Any) -> "BasePlatformAdapter | None":
        """Factory: create a platform-specific adapter instance."""
        ...

    @abstractmethod
    async def connect_adapter(self, adapter: "BasePlatformAdapter") -> bool:
        """
        Connect an adapter with timeout and error handling.
        Returns True if connected, False otherwise.
        """
        ...

    @abstractmethod
    async def disconnect_adapter(self, adapter: "BasePlatformAdapter") -> None:
        """Gracefully disconnect an adapter."""
        ...

    @abstractmethod
    async def dispose_adapter(self, adapter: "BasePlatformAdapter | None") -> None:
        """
        Best-effort dispose for an adapter that never made it onto adapters dict.
        Handles half-constructed adapters that may raise from disconnect().
        """
        ...

    @abstractmethod
    def get_adapter_status(self, adapter: "BasePlatformAdapter") -> dict[str, Any]:
        """Return adapter health / status info."""
        ...

    @abstractmethod
    async def reconnect_adapter(self, platform: Any, config: Any, attempt: int) -> bool:
        """
        Attempt to reconnect a failed adapter with backoff.
        Returns True if reconnected successfully.
        """
        ...


class AdapterManager(BaseAdapterManager):
    """
    Standard implementation of BaseAdapterManager.

    Handles the lifecycle of all platform adapters.
    Actual platform-specific adapters (Telegram, Discord, etc.) are
    instantiated via create_adapter() factory method.
    """

    DEFAULT_CONNECT_TIMEOUT = 30.0
    DEFAULT_DISCONNECT_TIMEOUT = 10.0

    def __init__(self, config: Any = None):
        self.config = config
        self._adapters: dict[str, "BasePlatformAdapter"] = {}
        self._failed_platforms: dict[str, dict[str, Any]] = {}

    # ─── Factory ────────────────────────────────────────────────────────────

    async def create_adapter(self, platform: Any, config: Any) -> "BasePlatformAdapter | None":
        """Create a platform-specific adapter. Override for custom factory logic."""
        # Lazy import to avoid circular dependency
        from gateway.config import load_gateway_config
        from gateway.run import _create_adapter

        full_config = config or load_gateway_config()
        # _create_adapter is the existing factory in gateway/run.py
        return await _create_adapter(platform, full_config)

    # ─── Connection Lifecycle ────────────────────────────────────────────────

    async def connect_adapter(self, adapter: "BasePlatformAdapter") -> bool:
        """Connect with timeout and best-effort error handling."""
        if adapter is None:
            return False

        try:
            await asyncio.wait_for(
                adapter.connect(),
                timeout=self.DEFAULT_CONNECT_TIMEOUT,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Adapter %s connect() timed out after %ss",
                          getattr(adapter, 'name', type(adapter).__name__),
                          self.DEFAULT_CONNECT_TIMEOUT)
            return False
        except Exception as exc:
            logger.debug(
                "Adapter %s connect() raised %r",
                getattr(adapter, 'name', type(adapter).__name__),
                exc,
            )
            return False

    async def disconnect_adapter(self, adapter: "BasePlatformAdapter") -> None:
        """Graceful disconnect with timeout."""
        if adapter is None:
            return

        try:
            await asyncio.wait_for(
                adapter.disconnect(),
                timeout=self.DEFAULT_DISCONNECT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Adapter %s disconnect() timed out after %ss",
                          getattr(adapter, 'name', type(adapter).__name__),
                          self.DEFAULT_DISCONNECT_TIMEOUT)
        except Exception as exc:
            logger.debug(
                "Adapter %s disconnect() raised %r (ignored)",
                getattr(adapter, 'name', type(adapter).__name__),
                exc,
            )

    # ─── Resource Disposal ──────────────────────────────────────────────────

    async def dispose_adapter(self, adapter: "BasePlatformAdapter | None") -> None:
        """
        Best-effort dispose for an adapter that never made it onto self.adapters.

        This is the migrated _dispose_unused_adapter from gateway/run.py.
        Handles half-constructed adapters that may raise from disconnect().
        """
        if adapter is None:
            return

        try:
            await adapter.disconnect()
        except Exception:
            # Half-constructed adapters can raise from disconnect()
            # on objects that never finished initializing.
            # Must not let that escape and abort the watcher loop.
            logger.debug(
                "Adapter dispose raised on unowned adapter %r",
                getattr(adapter, "name", type(adapter).__name__),
                exc_info=True,
            )

    # ─── Status & Monitoring ────────────────────────────────────────────────

    def get_adapter_status(self, adapter: "BasePlatformAdapter") -> dict[str, Any]:
        """Return adapter health / status info."""
        return {
            "name": getattr(adapter, "name", type(adapter).__name__),
            "connected": getattr(adapter, "_connected", False),
            "platform": str(getattr(adapter, "platform", "unknown")),
        }

    async def reconnect_adapter(self, platform: Any, config: Any, attempt: int) -> bool:
        """
        Attempt to reconnect with exponential backoff.
        Backoff logic migrated from _platform_reconnect_watcher.
        """
        from conflict.resolver import ConflictResolver, ConflictEvent

        # Use ConflictResolver for retry policy resolution
        resolver = ConflictResolver()
        event = ConflictEvent(
            source_module="AGENTS",
            conflict_type="reconnect_backoff",
            options={
                "pinned": min(300, 2 ** attempt),  # pinned max 300s
                "default": 30,
            }
        )
        resolution = resolver.resolve(event)
        backoff_secs = resolution.winner_value

        logger.info(
            "Reconnecting %s (attempt %d), backing off %ss",
            platform, attempt, backoff_secs
        )

        await asyncio.sleep(backoff_secs)

        adapter = await self.create_adapter(platform, config)
        if adapter is None:
            return False

        return await self.connect_adapter(adapter)

    # ─── Registry ───────────────────────────────────────────────────────────

    def register_adapter(self, key: str, adapter: "BasePlatformAdapter") -> None:
        """Register a live adapter in the internal registry."""
        self._adapters[key] = adapter

    def unregister_adapter(self, key: str) -> "BasePlatformAdapter | None":
        """Remove an adapter from the registry."""
        return self._adapters.pop(key, None)

    def list_adapters(self) -> list[str]:
        """List all registered adapter keys."""
        return list(self._adapters.keys())

    def is_adapter_connected(self, key: str) -> bool:
        """Check if a registered adapter is currently connected."""
        adapter = self._adapters.get(key)
        if adapter is None:
            return False
        return getattr(adapter, "_connected", False)