"""
Fallback Manager for Trading Operations.

Manages the priority order and health of different trading backends.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class BackendStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class BackendHealth:
    """Health status of a trading backend."""
    name: str
    status: BackendStatus = BackendStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    supported_chains: List[str] = field(default_factory=list)
    supported_actions: List[str] = field(default_factory=list)


class FallbackManager:
    """
    Manage trading backend health and fallback order.

    Tracks:
    - Backend availability
    - Success/failure rates
    - Latency
    - Chain/action support

    Automatically adjusts fallback order based on health.
    """

    # Default priority order
    DEFAULT_PRIORITY = [
        "bankr",      # Primary - handles most cases
        "jupiter",    # Solana DEX aggregator
        "pumpportal", # Pump.fun memecoins
        "0x",         # EVM aggregator
        "direct_rpc", # Last resort
    ]

    # Backend capabilities
    BACKEND_CAPABILITIES = {
        "bankr": {
            "chains": ["base", "ethereum", "solana", "polygon"],
            "actions": ["buy", "sell", "swap", "bridge"],
        },
        "jupiter": {
            "chains": ["solana"],
            "actions": ["swap"],
        },
        "pumpportal": {
            "chains": ["solana"],
            "actions": ["buy", "sell"],
        },
        "0x": {
            "chains": ["ethereum", "base", "polygon", "arbitrum", "optimism"],
            "actions": ["swap"],
        },
        "direct_rpc": {
            "chains": ["*"],  # All chains
            "actions": ["transfer"],  # Limited actions
        },
    }

    def __init__(self):
        self.backends: Dict[str, BackendHealth] = {}
        self.priority: List[str] = self.DEFAULT_PRIORITY.copy()
        self._init_backends()

    def _init_backends(self):
        """Initialize backend health tracking."""
        for name, caps in self.BACKEND_CAPABILITIES.items():
            self.backends[name] = BackendHealth(
                name=name,
                supported_chains=caps["chains"],
                supported_actions=caps["actions"],
            )

    def get_backends_for_request(
        self,
        chain: str,
        action: str
    ) -> List[str]:
        """
        Get backends that can handle a request, in priority order.

        Args:
            chain: Target blockchain
            action: Trade action (buy, sell, swap, etc.)

        Returns:
            List of backend names in priority order
        """
        suitable = []

        for name in self.priority:
            backend = self.backends.get(name)
            if not backend:
                continue

            # Check chain support
            if "*" not in backend.supported_chains and chain.lower() not in backend.supported_chains:
                continue

            # Check action support
            if action.lower() not in backend.supported_actions:
                continue

            # Check health
            if backend.status == BackendStatus.UNHEALTHY:
                continue

            suitable.append(name)

        return suitable

    def record_success(self, backend: str, latency_ms: float):
        """Record a successful operation."""
        if backend not in self.backends:
            return

        health = self.backends[backend]
        health.last_success = datetime.now()
        health.success_count += 1
        health.failure_count = 0  # Reset failure count

        # Update rolling average latency
        health.avg_latency_ms = (
            health.avg_latency_ms * 0.9 + latency_ms * 0.1
        )

        # Update status
        health.status = BackendStatus.HEALTHY
        health.last_check = datetime.now()

    def record_failure(self, backend: str, error: str = None):
        """Record a failed operation."""
        if backend not in self.backends:
            return

        health = self.backends[backend]
        health.failure_count += 1
        health.last_check = datetime.now()

        # Update status based on failure count
        if health.failure_count >= 5:
            health.status = BackendStatus.UNHEALTHY
            logger.warning(f"Backend {backend} marked unhealthy: {error}")
        elif health.failure_count >= 2:
            health.status = BackendStatus.DEGRADED

    async def check_backend_health(self, backend: str) -> BackendStatus:
        """
        Perform a health check on a backend.

        Args:
            backend: Backend name to check

        Returns:
            Current health status
        """
        if backend not in self.backends:
            return BackendStatus.UNKNOWN

        health = self.backends[backend]

        try:
            if backend == "bankr":
                from Cosmos.integration.bankr import get_bankr_client
                client = get_bankr_client()
                is_healthy = await client.health_check()
                health.status = BackendStatus.HEALTHY if is_healthy else BackendStatus.UNHEALTHY

            elif backend == "jupiter":
                # Check Jupiter API
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get("https://quote-api.jup.ag/v6/quote?inputMint=So11111111111111111111111111111111111111112&outputMint=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v&amount=1000000", timeout=5) as resp:
                        health.status = BackendStatus.HEALTHY if resp.status == 200 else BackendStatus.DEGRADED

            elif backend == "pumpportal":
                # Check PumpPortal
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get("https://pumpportal.fun/api/status", timeout=5) as resp:
                        health.status = BackendStatus.HEALTHY if resp.status == 200 else BackendStatus.DEGRADED

            else:
                # Unknown backend
                health.status = BackendStatus.UNKNOWN

        except Exception as e:
            logger.warning(f"Health check failed for {backend}: {e}")
            health.status = BackendStatus.UNHEALTHY

        health.last_check = datetime.now()
        return health.status

    async def check_all_backends(self):
        """Check health of all backends."""
        tasks = [
            self.check_backend_health(name)
            for name in self.backends
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    def reorder_priority(self):
        """
        Reorder priority based on health and performance.

        Healthy backends with lower latency get higher priority.
        """
        def score(name: str) -> tuple:
            health = self.backends.get(name)
            if not health:
                return (3, float('inf'))

            status_score = {
                BackendStatus.HEALTHY: 0,
                BackendStatus.DEGRADED: 1,
                BackendStatus.UNKNOWN: 2,
                BackendStatus.UNHEALTHY: 3,
            }.get(health.status, 3)

            return (status_score, health.avg_latency_ms)

        self.priority.sort(key=score)

    def get_status_report(self) -> Dict[str, Any]:
        """Get a status report of all backends."""
        return {
            name: {
                "status": health.status.value,
                "last_check": health.last_check.isoformat() if health.last_check else None,
                "success_count": health.success_count,
                "failure_count": health.failure_count,
                "avg_latency_ms": round(health.avg_latency_ms, 2),
                "chains": health.supported_chains,
            }
            for name, health in self.backends.items()
        }
