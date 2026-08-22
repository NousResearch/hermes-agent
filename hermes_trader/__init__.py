"""Hermes Agentic Trader — autonomous Base/EVM trading on Hermes Agent."""

from hermes_trader.config import TraderConfig, load_trader_config
from hermes_trader.memory import EpisodeStore, retrieve_similar_episodes, load_strategic_rules
from hermes_trader.reflection import run_reflection, run_weekly_calibration
from hermes_trader.loop.executor import OrderExecutor
from hermes_trader.loop.scheduler import (
    CycleResult,
    TradingCycleRunner,
    build_cron_job_spec,
    run_trading_cycle,
)
from hermes_trader.risk import (
    GateDecision,
    Mandate,
    OrderRequest,
    RejectReason,
    RiskGate,
    TradeIntent,
    load_mandate,
    sign_mandate,
    validate_mandate,
)
from hermes_trader.tools import (
    LIVE_WRITE_TOOLS,
    PAPER_MODE_READ_TOOLS,
    ToolPolicy,
    resolve_tool_policy,
)

__all__ = [
    "TraderConfig",
    "load_trader_config",
    "EpisodeStore",
    "retrieve_similar_episodes",
    "load_strategic_rules",
    "run_reflection",
    "run_weekly_calibration",
    "CycleResult",
    "OrderExecutor",
    "TradingCycleRunner",
    "build_cron_job_spec",
    "run_trading_cycle",
    "GateDecision",
    "Mandate",
    "OrderRequest",
    "RejectReason",
    "RiskGate",
    "TradeIntent",
    "load_mandate",
    "sign_mandate",
    "validate_mandate",
    "LIVE_WRITE_TOOLS",
    "PAPER_MODE_READ_TOOLS",
    "ToolPolicy",
    "resolve_tool_policy",
]

__version__ = "0.6.0"