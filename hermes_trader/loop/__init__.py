"""Agent loop: perceive → reason → gate → execute."""

from hermes_trader.loop.executor import ExecutionResult, OrderExecutor
from hermes_trader.loop.intent import hold_intent, parse_trade_intent, validate_trade_intent
from hermes_trader.loop.perceive import perceive_market

__all__ = [
    "ExecutionResult",
    "OrderExecutor",
    "hold_intent",
    "parse_trade_intent",
    "perceive_market",
    "validate_trade_intent",
]


def __getattr__(name: str):
    """Lazy exports for scheduler symbols (avoids memory import cycle)."""
    if name in {
        "CycleResult",
        "TradingCycleRunner",
        "build_cron_job_spec",
        "count_write_tool_calls",
        "cron_schedule_from_config",
        "run_trading_cycle",
    }:
        from hermes_trader.loop import scheduler as _scheduler

        return getattr(_scheduler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")