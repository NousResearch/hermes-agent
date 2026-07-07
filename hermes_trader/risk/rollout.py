"""Config-driven rollout stages with capital and chain caps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from hermes_trader.config import TraderConfig

RolloutStage = str

ROLLOUT_PRESETS: dict[str, dict[str, object]] = {
    "paper": {"capital_cap_usd": 0.0, "chains": ["base"]},
    "canary": {"capital_cap_usd": 50.0, "chains": ["base"]},
    "limited": {"capital_cap_usd": 500.0, "chains": ["base", "ethereum"]},
    "steady": {"capital_cap_usd": None, "chains": None},
}


@dataclass(frozen=True)
class RolloutPolicy:
    stage: str
    capital_cap_usd: Optional[float]
    allowed_chains: List[str]

    def effective_chains(self, config: TraderConfig) -> List[str]:
        preset_chains = self.allowed_chains
        if not preset_chains:
            return list(config.allowed_chains)
        return [c for c in config.allowed_chains if c in preset_chains]

    def max_trade_usd(self, config: TraderConfig) -> Optional[float]:
        if self.capital_cap_usd is not None:
            return float(self.capital_cap_usd)
        if config.rollout_capital_cap_usd is not None:
            return float(config.rollout_capital_cap_usd)
        return None


def resolve_rollout_policy(config: TraderConfig) -> RolloutPolicy:
    stage = (config.rollout_stage or "paper").strip().lower()
    preset = ROLLOUT_PRESETS.get(stage, ROLLOUT_PRESETS["paper"])
    cap = config.rollout_capital_cap_usd
    if cap is None:
        cap = preset.get("capital_cap_usd")
    chains = preset.get("chains")
    return RolloutPolicy(
        stage=stage,
        capital_cap_usd=cap if cap is None else float(cap),
        allowed_chains=list(chains) if isinstance(chains, list) else [],
    )


def chain_allowed_by_rollout(chain: str, config: TraderConfig) -> bool:
    policy = resolve_rollout_policy(config)
    return chain.strip().lower() in policy.effective_chains(config)


def trade_within_rollout_cap(size_usd: float, config: TraderConfig) -> tuple[bool, str]:
    cap = resolve_rollout_policy(config).max_trade_usd(config)
    if cap is None:
        return True, ""
    if size_usd <= cap:
        return True, ""
    return False, f"Trade size ${size_usd:,.2f} exceeds rollout cap ${cap:,.2f} ({config.rollout_stage})"