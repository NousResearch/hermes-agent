"""
Analyzes market conditions and calculates adaptive slippage parameters.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Tuple, Optional, List
import math
from datetime import datetime
from .models import Order, MarketConditions, ExecutionResult, SlippageProtectionConfig
from Cosmos.core.capability_registry import get_capability_registry
from Cosmos.memory.memory_system import get_memory_system
from loguru import logger

class AdaptiveSlippageAnalyzer:
    def __init__(self, config: Optional[SlippageProtectionConfig] = None):
        self.config = SlippageProtectionConfig() if config is None else config
        self.execution_history: List[ExecutionResult] = []
        self._market_impact_curve_exponent = 1.8
        self._circuit_breaker_threshold = Decimal("0.15")  # 15% ATR
        
    async def calculate_dynamic_slippage_tolerance(
        self,
        order: Order,
        conditions: MarketConditions,
        execution_history: Optional[List[ExecutionResult]] = None
    ) -> int:
        """
        Calculate adaptive slippage tolerance in basis points based on 
        volatility and liquidity conditions.
        
        Returns:
            int: slippage tolerance in basis points
        """
        try:
            # Check for circuit breakers
            if conditions.volatility_atr_24h > Decimal("0.10"):
                logger.warning(f"High volatility detected: {conditions.volatility_atr_24h}")
                
            # Calculate base tolerance
            base_tolerance = self.config.base_tolerance_bps
            
            # Apply volatility adjustment
            volatility_factor = Decimal("1.0")
            if conditions.volatility_atr_24h > Decimal("0.03"):
                volatility_factor = (conditions.volatility_atr_24h / Decimal("0.03")).min(Decimal("2.0"))
                logger.info(f"Volatility factor: {volatility_factor}")
            
            # Apply liquidity adjustment
            liquidity_factor = Decimal("1.0")
            if conditions.liquidity_depth_2pct > Decimal("5000000"):
                liquidity_factor = Decimal("0.5")
            elif conditions.liquidity_depth_2pct < Decimal("500000"):
                liquidity_factor = Decimal("1.5")
            logger.info(f"Liquidity factor: {liquidity_factor}")
            
            # Apply order size adjustment
            size_factor = Decimal("1.0")
            if order.amount > Decimal("1000000"):
                size_factor = (order.amount / Decimal("1000000")).sqrt()
            elif order.amount < Decimal("10000"):
                size_factor = Decimal("2.0")
            logger.info(f"Size factor: {size_factor}")
            
            # Combine factors
            adjusted_tolerance = base_tolerance * volatility_factor * liquidity_factor * size_factor
            
            # Apply hard caps
            adjusted_tolerance = adjusted_tolerance.clip(self.config.min_tolerance_bps, self.config.max_tolerance_bps)
            
            return int(round(adjusted_tolerance))
            
        except Exception as e:
            logger.error(f"Error calculating slippage tolerance: {str(e)}")
            return self.config.base_tolerance_bps

    async def estimate_market_impact(
        self,
        order_value_usd: Decimal,
        liquidity_depth: Decimal,
        curve_exponent: float = 1.8
    ) -> Decimal:
        """
        Estimate price impact using power law: impact = (order/depth)^exponent
        Returns impact as decimal (0.01 = 1%)
        
        Args:
            order_value_usd: Total value of the order
            liquidity_depth: USD liquidity depth in the order book
            curve_exponent: Power law exponent (default: 1.8)
            
        Returns:
            Decimal: estimated price impact
        """
        try:
            if liquidity_depth <= Decimal("0"):
                raise ValueError("Liquidity depth must be positive")
                
            order_depth_ratio = order_value_usd / liquidity_depth
            impact = order_depth_ratio ** Decimal(str(curve_exponent))
            
            # Add slippage estimate
            slippage_estimate = (impact * Decimal("0.001")).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            logger.info(f"Estimated market impact: {impact} (or {slippage_estimate} bps)")
            
            return impact
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {str(e)}")
            return Decimal("0.05")  # default 5% impact

    async def calculate_optimal_chunk_size(
        self,
        total_amount: Decimal,
        conditions: MarketConditions,
        max_impact_bps: int = 25
    ) -> List[Decimal]:
        """
        Split large orders into TWAP chunks to minimize slippage.
        Returns list of order sizes that sum to total_amount.
        
        Args:
            total_amount: Total amount to trade
            conditions: Current market conditions
            max_impact_bps: Maximum allowed price impact per chunk (basis points)
            
        Returns:
            List[Decimal]: List of chunk sizes
        """
        try:
            # Convert max impact to decimal
            max_impact = Decimal(str(max_impact_bps / 10000))
            
            # Estimate market impact for the entire order
            impact_full = await self.estimate_market_impact(total_amount, conditions.liquidity_depth_2pct)
            
            # Calculate number of chunks needed
            num_chunks = max(1, math.ceil(impact_full / max_impact))
            
            # Calculate chunk size
            chunk_size = total_amount / Decimal(str(num_chunks))
            
            # Apply TWAP chunk size adjustments based on market conditions
            chunk_size = chunk_size * (
                Decimal("1.1") if conditions.bid_ask_spread_bps < 10 else
                Decimal("0.9") if conditions.order_book_imbalance > 0.3 else Decimal("1.0")
            )
            
            # Ensure chunk size is reasonable
            chunk_size = chunk_size.max(Decimal("1000"))
            
            # Generate chunks
            chunks = []
            remaining = total_amount
            while remaining > chunk_size:
                chunks.append(chunk_size)
                remaining -= chunk_size
            
            if remaining > Decimal("0"):
                chunks.append(remaining)
            
            logger.info(f"Optimal TWAP chunks: {len(chunks)} chunks of {chunk_size}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error calculating optimal chunk size: {str(e)}")
            return [total_amount]  # fallback to single order

    async def should_retry_with_adjustment(
        self,
        last_result: ExecutionResult,
        attempt: int,
        max_attempts: int = 3
    ) -> Tuple[bool, Optional[Order]]:
        """
        Determine if failed/cancelled order should retry with adjusted params.
        Returns (should_retry, adjusted_order_or_none).
        
        Args:
            last_result: Execution result of previous attempt
            attempt: Current attempt number
            max_attempts: Maximum allowed attempts
            
        Returns:
            Tuple[bool, Optional[Order]]: (should retry, adjusted order or None)
        """
        try:
            if attempt >= max_attempts:
                return False, None
                
            # Calculate slippage percentage
            slippage_percent = (last_result.actual_slippage_bps / 
                               (self.config.base_tolerance_bps * Decimal(str(attempt)))) * Decimal("100")
            
            # Check if slippage exceeds tolerance
            if slippage_percent > Decimal("15"):
                logger.warning(f"High slippage detected: {slippage_percent}%")
                
                # Calculate adjustment
                adjustment_factor = Decimal("1.0") + (slippage_percent / Decimal("100"))
                
                # Adjust order parameters
                new_order = Order(
                    symbol=last_result.order_id.split("-")[0],  # extract symbol
                    side=last_result.order_id.split("-")[1],    # extract side
                    amount=last_result.executed_price * Decimal("1000") * adjustment_factor,
                    order_type="limit",
                    max_slippage_bps=int(self.config.base_tolerance_bps * adjustment_factor)
                )
                
                return True, new_order
                
            return False, None
            
        except Exception as e:
            logger.error(f"Error deciding retry: {str(e)}")
            return False, None