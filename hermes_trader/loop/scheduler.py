"""Trading cycle orchestration and Hermes cron integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_trader.audit.alerts import AlertStore, evaluate_alerts
from hermes_trader.config import TraderConfig, load_trader_config
from hermes_trader.loop.audit import CycleAuditLog, default_cycle_log_path
from hermes_trader.loop.context import retrieve_context
from hermes_trader.memory.episodes import EpisodeStore
from hermes_trader.loop.executor import ExecutionResult, OrderExecutor
from hermes_trader.loop.intent import hold_intent, parse_trade_intent, validate_trade_intent
from hermes_trader.loop.perceive import McpCallFn, perceive_market
from hermes_trader.loop.reason import build_cycle_prompt, heuristic_reasoner
from hermes_trader.market_state import MarketState
from hermes_trader.risk.gate import GateDecision, RiskGate, TradeIntent
from hermes_trader.tools import LIVE_WRITE_TOOLS

ReasonFn = Callable[[MarketState, TraderConfig, list[dict[str, Any]]], TradeIntent]


@dataclass
class CycleResult:
    market_state: MarketState
    intent: TradeIntent
    decision: GateDecision
    execution: Optional[ExecutionResult] = None
    context: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_state": self.market_state.to_dict(),
            "intent": {
                "action": self.intent.action,
                "chain": self.intent.chain,
                "token_address": self.intent.token_address,
                "size_usd": self.intent.size_usd,
                "confidence": self.intent.confidence,
                "reasoning": self.intent.reasoning,
                "strategy_tag": self.intent.strategy_tag,
            },
            "decision": self.decision.to_dict(),
            "execution": self.execution.to_dict() if self.execution else None,
            "context_count": len(self.context),
        }


def cron_schedule_from_config(config: TraderConfig) -> str:
    """Cron expression from scan_interval_minutes (minimum 1 minute)."""
    minutes = max(1, int(config.scan_interval_minutes))
    if minutes >= 60 and minutes % 60 == 0:
        hours = max(1, minutes // 60)
        return f"0 */{hours} * * *"
    return f"*/{minutes} * * * *"


def build_cron_job_spec(
    config: Optional[TraderConfig] = None,
    *,
    job_id: str = "trader-scan",
) -> dict[str, Any]:
    """Kwargs for ``cron.jobs.create_job`` to schedule LLM-driven scans."""
    cfg = config or load_trader_config()
    prompt = build_cycle_prompt(cfg, MarketState(chain=cfg.primary_chain, captured_at=""))
    return {
        "prompt": prompt,
        "schedule": cron_schedule_from_config(cfg),
        "name": job_id,
        "skill": "hermes-agentic-trader",
        "repeat": None,
        "deliver": "local",
    }


class TradingCycleRunner:
    """Closed-loop perceive → reason → gate → execute."""

    def __init__(
        self,
        config: TraderConfig,
        mcp_call: McpCallFn,
        *,
        reason_fn: Optional[ReasonFn] = None,
        audit_log: Optional[CycleAuditLog] = None,
        episode_store: Optional[EpisodeStore] = None,
        daily_loss_pct: float = 0.0,
    ):
        self.config = config
        self.mcp_call = mcp_call
        self.reason_fn = reason_fn or _default_reason_fn
        self.audit_log = audit_log or CycleAuditLog()
        self.episode_store = episode_store or EpisodeStore()
        self.daily_loss_pct = daily_loss_pct
        self.gate = RiskGate(config=config)
        self.executor = OrderExecutor(config, mcp_call)

    def run_once(self) -> CycleResult:
        state = perceive_market(self.config, self.mcp_call)
        context = retrieve_context(
            state,
            config=self.config,
            episode_store=self.episode_store,
        )
        intent = self.reason_fn(state, self.config, context)
        validate_trade_intent(intent)
        decision = self.gate.evaluate(
            intent,
            market_state=state,
            daily_loss_pct=self.daily_loss_pct,
        )

        execution: Optional[ExecutionResult] = None
        if decision.approved and decision.order is not None:
            execution = self.executor.execute(decision.order)
        else:
            execution = ExecutionResult(
                status="skipped",
                message=decision.message,
            )

        result = CycleResult(
            market_state=state,
            intent=intent,
            decision=decision,
            execution=execution,
            context=context,
        )
        self._log_cycle(result)
        self.episode_store.record_cycle(result)
        self._emit_alerts()
        return result

    def _emit_alerts(self) -> None:
        alerts = evaluate_alerts(
            self.config,
            episode_store=self.episode_store,
            cycles_log_path=self.audit_log.path,
        )
        if alerts:
            AlertStore().emit(alerts)

    def _log_cycle(self, result: CycleResult) -> None:
        self.audit_log.append(
            {
                "mode": self.config.mode,
                "approved": result.decision.approved,
                "reason_code": (
                    result.decision.reason_code.value
                    if result.decision.reason_code
                    else None
                ),
                "intent_action": result.intent.action,
                "execution_status": result.execution.status if result.execution else None,
                "cycle": result.to_dict(),
            }
        )


def _default_reason_fn(
    state: MarketState,
    config: TraderConfig,
    context: list[dict[str, Any]],
) -> TradeIntent:
    _ = context
    return heuristic_reasoner(state, config)


def run_trading_cycle(
    *,
    config: Optional[TraderConfig] = None,
    mcp_call: McpCallFn,
    reason_fn: Optional[ReasonFn] = None,
    audit_log: Optional[CycleAuditLog] = None,
    episode_store: Optional[EpisodeStore] = None,
    daily_loss_pct: float = 0.0,
) -> CycleResult:
    """Run one full trading cycle."""
    runner = TradingCycleRunner(
        config or load_trader_config(),
        mcp_call,
        reason_fn=reason_fn,
        audit_log=audit_log,
        episode_store=episode_store,
        daily_loss_pct=daily_loss_pct,
    )
    return runner.run_once()


def count_write_tool_calls(calls: list[tuple[str, str, dict[str, Any]]]) -> int:
    """Count MCP invocations that target live write tools."""
    return sum(1 for _srv, tool, _args in calls if tool in LIVE_WRITE_TOOLS)


def main() -> int:
    """CLI entry: ``python -m hermes_trader.loop.scheduler`` for cron scripts."""
    import json
    import sys

    from tools.mcp_tool import _make_tool_handler  # lazy — only when MCP is up

    cfg = load_trader_config()

    def mcp_call(server: str, tool: str, args: dict[str, Any]) -> Any:
        handler = _make_tool_handler(server, tool, 60.0)
        raw = handler(args)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    result = run_trading_cycle(config=cfg, mcp_call=mcp_call)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.decision.approved or result.intent.action == "hold" else 0


if __name__ == "__main__":
    raise SystemExit(main())