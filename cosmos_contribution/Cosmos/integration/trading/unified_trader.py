"""
Unified Trading Interface.

Routes trades through Bankr first, with fallback to chain-specific implementations.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    FALLBACK = "fallback"  # Succeeded via fallback


@dataclass
class TradeRequest:
    """Unified trade request."""
    action: str  # buy, sell, swap, bridge
    token_in: str
    token_out: str
    amount: Decimal
    chain: str = "base"
    slippage_bps: int = 50  # 0.5%
    denomination: str = "token"  # "token" or "usd"

    def to_bankr_prompt(self) -> str:
        """Convert to Bankr natural language prompt."""
        if self.action == "buy":
            if self.denomination == "usd":
                return f"Buy ${self.amount} of {self.token_out} on {self.chain}"
            return f"Buy {self.amount} {self.token_out} on {self.chain}"

        elif self.action == "sell":
            return f"Sell {self.amount} {self.token_in} on {self.chain}"

        elif self.action == "swap":
            return f"Swap {self.amount} {self.token_in} to {self.token_out} on {self.chain}"

        elif self.action == "bridge":
            # Assumes token_in contains source chain info
            return f"Bridge {self.amount} {self.token_out} from {self.token_in} to {self.chain}"

        return f"{self.action} {self.amount} {self.token_in or self.token_out} on {self.chain}"


@dataclass
class TradeResponse:
    """Unified trade response."""
    status: TradeStatus
    tx_hash: Optional[str] = None
    amount_in: Decimal = Decimal(0)
    amount_out: Decimal = Decimal(0)
    price: Optional[Decimal] = None
    chain: str = ""
    handler: str = ""  # "bankr", "jupiter", "pumpportal", etc.
    url: Optional[str] = None  # Explorer URL
    error: Optional[str] = None
    raw_response: Dict = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class UnifiedTrader:
    """
    Unified trading interface across all chains.

    Uses Bankr as the primary handler with fallback to:
    - Jupiter (Solana swaps)
    - PumpPortal (Pump.fun trades)
    - Direct RPC (when APIs fail)
    """

    def __init__(self):
        self._bankr_available = None
        self._bankr_trading = None

    async def _check_bankr(self) -> bool:
        """Check if Bankr is available."""
        if self._bankr_available is not None:
            return self._bankr_available

        try:
            from Cosmos.integration.bankr import get_bankr_client
            from Cosmos.integration.bankr.config import get_bankr_config

            config = get_bankr_config()
            if not config.api_key:
                self._bankr_available = False
                return False

            client = get_bankr_client()
            self._bankr_available = await client.health_check()
            return self._bankr_available

        except ImportError:
            self._bankr_available = False
            return False
        except Exception as e:
            logger.warning(f"Bankr health check failed: {e}")
            self._bankr_available = False
            return False

    async def _get_bankr_trading(self):
        """Get Bankr trading instance."""
        if self._bankr_trading is None:
            try:
                from Cosmos.integration.bankr import BankrTrading, get_bankr_client
                self._bankr_trading = BankrTrading(get_bankr_client())
            except ImportError:
                return None
        return self._bankr_trading

    async def execute(self, request: TradeRequest) -> TradeResponse:
        """
        Execute a trade using the best available method.

        Priority:
        1. Bankr (if available and supports the chain)
        2. Chain-specific fallback (Jupiter, PumpPortal, etc.)
        3. Direct RPC

        Args:
            request: TradeRequest with trade details

        Returns:
            TradeResponse with result
        """
        # Try Bankr first
        if await self._check_bankr():
            try:
                response = await self._execute_via_bankr(request)
                if response.status == TradeStatus.SUCCESS:
                    return response
                logger.warning(f"Bankr trade failed, trying fallback: {response.error}")
            except Exception as e:
                logger.warning(f"Bankr error, trying fallback: {e}")

        # Fallback based on chain
        return await self._execute_fallback(request)

    async def _execute_via_bankr(self, request: TradeRequest) -> TradeResponse:
        """Execute trade via Bankr."""
        trading = await self._get_bankr_trading()
        if not trading:
            return TradeResponse(
                status=TradeStatus.FAILED,
                error="Bankr trading not available",
                handler="bankr",
            )

        prompt = request.to_bankr_prompt()

        if request.action == "buy":
            amount_usd = float(request.amount) if request.denomination == "usd" else 0
            result = await trading.buy_token(
                token=request.token_out,
                amount_usd=amount_usd or float(request.amount),
                chain=request.chain,
            )

        elif request.action == "sell":
            result = await trading.sell_token(
                token=request.token_in,
                amount=float(request.amount),
                chain=request.chain,
            )

        elif request.action == "swap":
            result = await trading.swap(
                from_token=request.token_in,
                to_token=request.token_out,
                amount=float(request.amount),
                chain=request.chain,
            )

        elif request.action == "bridge":
            result = await trading.bridge(
                token=request.token_out,
                amount=float(request.amount),
                from_chain=request.token_in,  # Repurposed field
                to_chain=request.chain,
            )

        else:
            return TradeResponse(
                status=TradeStatus.FAILED,
                error=f"Unknown action: {request.action}",
                handler="bankr",
            )

        if result.success:
            return TradeResponse(
                status=TradeStatus.SUCCESS,
                tx_hash=result.tx_hash,
                amount_in=result.amount_in,
                amount_out=result.amount_out,
                chain=request.chain,
                handler="bankr",
                raw_response=result.raw_response,
            )
        else:
            return TradeResponse(
                status=TradeStatus.FAILED,
                error=result.error,
                handler="bankr",
            )

    async def _execute_fallback(self, request: TradeRequest) -> TradeResponse:
        """Execute trade via chain-specific fallback."""
        chain = request.chain.lower()

        if chain == "solana":
            return await self._execute_solana(request)
        elif chain in ("base", "ethereum", "polygon", "arbitrum"):
            return await self._execute_evm(request)
        else:
            return TradeResponse(
                status=TradeStatus.FAILED,
                error=f"No fallback handler for chain: {chain}",
                handler="none",
            )

    async def _execute_solana(self, request: TradeRequest) -> TradeResponse:
        """Fallback to direct Solana trading."""
        try:
            from Cosmos.integration.solana.trading import solana_trader

            if request.action == "swap":
                # Use Jupiter via existing implementation
                result = await solana_trader.jupiter_swap(
                    input_mint=request.token_in,
                    output_mint=request.token_out,
                    amount_indices=int(float(request.amount) * 10**9),  # Convert to lamports
                    slippage_bps=request.slippage_bps,
                )

                if result.get("status") == "success":
                    return TradeResponse(
                        status=TradeStatus.FALLBACK,
                        tx_hash=result.get("signature"),
                        url=result.get("url"),
                        chain="solana",
                        handler="jupiter",
                        raw_response=result,
                    )
                else:
                    return TradeResponse(
                        status=TradeStatus.FAILED,
                        error=result.get("error"),
                        handler="jupiter",
                    )

            elif request.action in ("buy", "sell"):
                # Try PumpPortal for memecoins
                action = "buy" if request.action == "buy" else "sell"
                result = await solana_trader.pump_fun_trade(
                    action=action,
                    mint=request.token_out if action == "buy" else request.token_in,
                    amount=float(request.amount),
                    denominated_in_sol=request.denomination != "token",
                )

                if result.get("status") == "success":
                    return TradeResponse(
                        status=TradeStatus.FALLBACK,
                        tx_hash=result.get("signature"),
                        url=result.get("url"),
                        chain="solana",
                        handler="pumpportal",
                        raw_response=result,
                    )
                else:
                    return TradeResponse(
                        status=TradeStatus.FAILED,
                        error=result.get("error"),
                        handler="pumpportal",
                    )

            else:
                return TradeResponse(
                    status=TradeStatus.FAILED,
                    error=f"Solana fallback doesn't support: {request.action}",
                    handler="solana",
                )

        except ImportError as e:
            return TradeResponse(
                status=TradeStatus.FAILED,
                error=f"Solana trading module not available: {e}",
                handler="solana",
            )
        except Exception as e:
            return TradeResponse(
                status=TradeStatus.FAILED,
                error=str(e),
                handler="solana",
            )

    async def _execute_evm(self, request: TradeRequest) -> TradeResponse:
        """Fallback for EVM chains - currently limited."""
        # For now, EVM fallback is limited
        # Could integrate with 0x, 1inch, or direct contracts
        return TradeResponse(
            status=TradeStatus.FAILED,
            error=f"EVM fallback not fully implemented for {request.chain}. Use Bankr.",
            handler="evm",
        )

    # Convenience methods

    async def buy(
        self,
        token: str,
        amount: float,
        chain: str = "base",
        in_usd: bool = True
    ) -> TradeResponse:
        """Buy tokens."""
        return await self.execute(TradeRequest(
            action="buy",
            token_in="USD" if in_usd else "",
            token_out=token,
            amount=Decimal(str(amount)),
            chain=chain,
            denomination="usd" if in_usd else "token",
        ))

    async def sell(
        self,
        token: str,
        amount: float,
        chain: str = "base"
    ) -> TradeResponse:
        """Sell tokens."""
        return await self.execute(TradeRequest(
            action="sell",
            token_in=token,
            token_out="USD",
            amount=Decimal(str(amount)),
            chain=chain,
        ))

    async def swap(
        self,
        from_token: str,
        to_token: str,
        amount: float,
        chain: str = "base"
    ) -> TradeResponse:
        """Swap tokens."""
        return await self.execute(TradeRequest(
            action="swap",
            token_in=from_token,
            token_out=to_token,
            amount=Decimal(str(amount)),
            chain=chain,
        ))

    async def get_price(self, token: str) -> float:
        """Get token price."""
        # Try Bankr first
        if await self._check_bankr():
            try:
                from Cosmos.integration.bankr import get_bankr_client
                client = get_bankr_client()
                return await client.get_price(token)
            except Exception as e:
                logger.warning(f"Bankr price failed: {e}")

        # Fallback to DexScreener
        try:
            from Cosmos.integration.financial.dexscreener import dex_screener
            pairs = await dex_screener.search_pairs(token)
            if pairs:
                return float(pairs[0].get("priceUsd", 0))
        except Exception as e:
            logger.warning(f"DexScreener price failed: {e}")

        return 0.0
