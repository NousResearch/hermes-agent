from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    symbol: str
    trade_unit: int = 10000
    max_trades: int = 4


DEFAULT_RUNTIME_PROFILE = RuntimeProfile(
    profile_id="olin-688319",
    symbol="688319",
    trade_unit=10000,
    max_trades=4,
)
