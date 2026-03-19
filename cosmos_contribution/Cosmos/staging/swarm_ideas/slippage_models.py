"""
Data models for trading operations and risk management.
"""

from decimal import Decimal
from datetime import datetime
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, validator, conint

class Order(BaseModel):
    symbol: str = Field(..., regex=r'^[A-Z]{2,5}-[A-Z]{2,5}$')
    side: Literal["buy", "sell"]
    amount: Decimal = Field(..., gt=0)
    price: Optional[Decimal] = None
    order_type: Literal["market", "limit", "twap"] = "market"
    max_slippage_bps: int = Field(default=50, ge=10, le=1000)  # basis points
    
    @validator('max_slippage_bps')
    def validate_slippage(cls, v):
        if v < 0:
            raise ValueError("Slippage must be positive")
        return v

class MarketConditions(BaseModel):
    symbol: str
    volatility_atr_24h: Decimal  # Average True Range
    liquidity_depth_2pct: Decimal  # USD depth within 2% of mid
    bid_ask_spread_bps: conint(ge=0, le=2000)
    last_trade_timestamp: datetime
    order_book_imbalance: Decimal  # -1.0 to 1.0 (sell to buy pressure)
    
    @validator('bid_ask_spread_bps')
    def validate_spread(cls, v):
        if v < 0:
            raise ValueError("Spread cannot be negative")
        return v

class ExecutionResult(BaseModel):
    order_id: str
    executed_price: Decimal
    expected_price: Decimal
    actual_slippage_bps: int
    fill_percentage: Decimal
    timestamp: datetime
    adjustments_applied: List[str]  # ["volatility_boost", "liquidity_split"]
    
    @validator('fill_percentage')
    def validate_fill(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Fill percentage must be between 0 and 100")
        return v

class RiskLimits(BaseModel):
    max_single_order_usd: Decimal
    max_slippage_hard_cap_bps: int
    volatility_threshold_high: Decimal = Decimal("0.05")  # 5% ATR
    
    @validator('volatility_threshold_high')
    def validate_volatility(cls, v):
        if v <= 0:
            raise ValueError("Volatility threshold must be positive")
        return v

class SlippageProtectionConfig(BaseModel):
    base_tolerance_bps: int = 50
    max_tolerance_bps: int = 500
    min_tolerance_bps: int = 20
    volatility_multiplier: Decimal = Decimal("1.5")
    liquidity_threshold_usd: Decimal = Decimal("100000")
    max_order_value_usd: Decimal = Decimal("500000")
    max_attempts: int = 3
    chunk_size_percentage: int = 5  # for TWAP execution

class TWAPChunk(BaseModel):
    size: Decimal
    expected_price: Optional[Decimal] = None
    max_slippage_bps: Optional[int] = None