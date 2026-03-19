"""
Cosmos Quantum Trading Broadcaster — AGI v2.1

The broadcaster is how the organism's trading intelligence reaches the outside world.
Manages signal distribution across multiple channels:

1. Nexus Signals (internal) — Shadow agents + DegenTrader subscribe
2. WebSocket Feed (external, token-gated) — Real-time signals to DEX UI
3. REST API (external, token-gated) — Signal/accuracy/correlation endpoints
4. Agent Interface (internal) — Any shadow agent can request a signal

IBM Free Tier: All quantum work delegated to QuantumTradingCortex which handles budgets.
"""

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from loguru import logger


# =============================================================================
# BROADCASTER
# =============================================================================

class QuantumTradingBroadcaster:
    """
    Distributes quantum trading signals across all channels.
    Manages WebSocket subscribers, REST cache, and agent interface.
    """

    def __init__(self, cortex=None, nexus=None):
        self.cortex = cortex
        self.nexus = nexus

        # Signal cache for REST API
        self._latest_signals: Dict[str, dict] = {}  # token_address -> signal dict
        self._all_signals: deque = deque(maxlen=500)  # recent signals across all tokens
        self._correlation_cache: List[dict] = []
        self._accuracy_cache: dict = {}
        self._last_accuracy_update = 0.0

        # WebSocket subscribers: set of (ws_connection, subscribed_tokens)
        self._ws_subscribers: Set = set()

        # Rate limiting
        self._rate_limits: Dict[str, float] = {}  # client_id -> last_request_ts
        self._rate_limit_rpm = 30  # requests per minute

        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize broadcaster and subscribe to Nexus signals."""
        try:
            if self.cortex is None:
                from Cosmos.core.quantum_trading import get_quantum_cortex
                self.cortex = get_quantum_cortex()

            # Subscribe to Nexus for signal events
            if self.nexus:
                try:
                    from Cosmos.core.nexus import SignalType
                    await self.nexus.subscribe(
                        SignalType.QUANTUM_SIGNAL_GENERATED,
                        self._on_signal_generated
                    )
                    await self.nexus.subscribe(
                        SignalType.QUANTUM_CORRELATION_DISCOVERED,
                        self._on_correlation_discovered
                    )
                    await self.nexus.subscribe(
                        SignalType.QUANTUM_ACCURACY_UPDATED,
                        self._on_accuracy_updated
                    )
                except Exception as e:
                    logger.debug(f"Broadcaster: Nexus subscription failed: {e}")

            self._initialized = True
            logger.info("QuantumTradingBroadcaster initialized")
            return True
        except Exception as e:
            logger.error(f"Broadcaster init failed: {e}")
            return False

    # =================================================================
    # NEXUS SIGNAL HANDLERS
    # =================================================================

    async def _on_signal_generated(self, signal):
        """Handle new quantum trading signal from cortex."""
        try:
            payload = signal.payload if hasattr(signal, 'payload') else signal
            if isinstance(payload, dict):
                token_addr = payload.get("token_address", "")
                self._latest_signals[token_addr] = payload
                self._all_signals.append(payload)
                # Broadcast to WebSocket subscribers
                await self._broadcast_ws({
                    "type": "quantum_signal",
                    "data": payload,
                })
        except Exception as e:
            logger.debug(f"Broadcaster: Signal handler error: {e}")

    async def _on_correlation_discovered(self, signal):
        """Handle new correlation discovery."""
        try:
            payload = signal.payload if hasattr(signal, 'payload') else signal
            if isinstance(payload, dict):
                self._correlation_cache = payload.get("pairs", [])
        except Exception as e:
            logger.debug(f"Broadcaster: Correlation handler error: {e}")

    async def _on_accuracy_updated(self, signal):
        """Handle accuracy stats update."""
        try:
            payload = signal.payload if hasattr(signal, 'payload') else signal
            if isinstance(payload, dict):
                self._accuracy_cache = payload
                self._last_accuracy_update = time.time()
        except Exception as e:
            logger.debug(f"Broadcaster: Accuracy handler error: {e}")

    # =================================================================
    # REST API DATA
    # =================================================================

    def get_signal_for_token(self, token_address: str) -> Optional[dict]:
        """Get latest quantum signal for a specific token."""
        return self._latest_signals.get(token_address)

    def get_recent_signals(self, limit: int = 50) -> List[dict]:
        """Get recent signals across all tokens."""
        signals = list(self._all_signals)
        signals.reverse()
        return signals[:limit]

    def get_correlations(self) -> List[dict]:
        """Get discovered cross-token correlations."""
        if self.cortex:
            return [c.to_dict() for c in self.cortex.correlations.values()]
        return self._correlation_cache

    def get_accuracy_stats(self) -> dict:
        """Get public accuracy statistics (proves value, no gate needed)."""
        if self.cortex:
            return self.cortex.accuracy_tracker.get_accuracy_stats()
        return self._accuracy_cache

    def get_full_status(self) -> dict:
        """Get complete quantum trading status."""
        stats = {}
        if self.cortex:
            stats = self.cortex.get_stats()
        stats["broadcaster"] = {
            "initialized": self._initialized,
            "cached_signals": len(self._latest_signals),
            "total_broadcast": len(self._all_signals),
            "correlations": len(self._correlation_cache),
            "ws_subscribers": len(self._ws_subscribers),
        }
        return stats

    # =================================================================
    # AGENT INTERFACE
    # =================================================================

    async def call_quantum_cortex(
        self, token_address: str, price_history: Optional[List[float]] = None,
        current_price: float = 0.0
    ) -> dict:
        """
        Agent interface: any shadow agent can request a quantum trading signal.
        Registered in agent_spawner capabilities.
        """
        if not self.cortex:
            return {"error": "Quantum cortex not initialized"}

        try:
            signal = await self.cortex.generate_signal(
                token_address, price_history, current_price
            )
            return signal.to_dict()
        except Exception as e:
            logger.error(f"Broadcaster: call_quantum_cortex failed: {e}")
            return {"error": str(e)}

    # =================================================================
    # WEBSOCKET MANAGEMENT
    # =================================================================

    async def _broadcast_ws(self, message: dict):
        """Broadcast message to all WebSocket subscribers."""
        if not self._ws_subscribers:
            return

        msg_str = json.dumps(message)
        dead = set()
        for ws in self._ws_subscribers:
            try:
                if hasattr(ws, 'send'):
                    await ws.send(msg_str)
            except Exception:
                dead.add(ws)

        for ws in dead:
            self._ws_subscribers.discard(ws)

    def add_ws_subscriber(self, ws):
        """Add a WebSocket connection to broadcast list."""
        self._ws_subscribers.add(ws)

    def remove_ws_subscriber(self, ws):
        """Remove a WebSocket connection from broadcast list."""
        self._ws_subscribers.discard(ws)

    # =================================================================
    # RATE LIMITING
    # =================================================================

    def is_rate_limited(self, client_id: str) -> bool:
        """Check if a client is rate limited."""
        now = time.time()
        last = self._rate_limits.get(client_id, 0)
        interval = 60.0 / self._rate_limit_rpm

        if now - last < interval:
            return True

        self._rate_limits[client_id] = now
        return False

    def cleanup_rate_limits(self, max_age: float = 600.0):
        """Remove stale rate limit entries."""
        now = time.time()
        to_remove = [k for k, v in self._rate_limits.items() if now - v > max_age]
        for k in to_remove:
            del self._rate_limits[k]


# =============================================================================
# COSMOS TOKEN GATE — Balance verification for premium access
# =============================================================================

COSMOS_TOKEN_MINT = "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"
COSMOS_DECIMALS = 9
MIN_COSMOS_HOLDING = 100_000  # Minimum COSMOS to access quantum signals


async def check_cosmos_balance(wallet_address: str, min_amount: int = MIN_COSMOS_HOLDING) -> dict:
    """
    Check COSMOS token balance for a wallet via Solana RPC.
    Read-only check — no burn required.

    Returns: { has_access: bool, balance: int, required: int }
    """
    import os

    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

    try:
        import aiohttp

        # Get token accounts for this wallet
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                wallet_address,
                {"mint": COSMOS_TOKEN_MINT},
                {"encoding": "jsonParsed"}
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

        accounts = data.get("result", {}).get("value", [])
        total_balance = 0

        for account in accounts:
            info = account.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
            token_amount = info.get("tokenAmount", {})
            amount = int(token_amount.get("amount", "0"))
            decimals = token_amount.get("decimals", COSMOS_DECIMALS)
            total_balance += amount / (10 ** decimals)

        return {
            "has_access": total_balance >= min_amount,
            "balance": int(total_balance),
            "required": min_amount,
            "wallet": wallet_address,
        }

    except Exception as e:
        logger.debug(f"COSMOS balance check failed for {wallet_address[:8]}...: {e}")
        return {
            "has_access": False,
            "balance": 0,
            "required": min_amount,
            "error": str(e),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_broadcaster: Optional[QuantumTradingBroadcaster] = None

def get_quantum_broadcaster() -> QuantumTradingBroadcaster:
    """Get or create the singleton QuantumTradingBroadcaster."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = QuantumTradingBroadcaster()
    return _broadcaster

async def initialize_quantum_broadcaster(cortex=None, nexus=None) -> QuantumTradingBroadcaster:
    """Initialize the broadcaster with organism infrastructure."""
    global _broadcaster
    broadcaster = get_quantum_broadcaster()
    broadcaster.cortex = cortex or broadcaster.cortex
    broadcaster.nexus = nexus or broadcaster.nexus
    await broadcaster.initialize()
    return broadcaster
