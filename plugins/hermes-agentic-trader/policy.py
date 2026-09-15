"""Deterministic policy model for Hermes Agentic Trader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

CHAIN_NAME_TO_ID = {
    "ethereum": 1,
    "eth": 1,
    "base": 8453,
    "arbitrum": 42161,
}

ROLLOUT_PRESETS = {
    "paper": {"chains": (8453,), "cap_usd": 0.0},
    "canary": {"chains": (8453,), "cap_usd": 50.0},
    "limited": {"chains": (8453, 1), "cap_usd": 500.0},
    "steady": {"chains": None, "cap_usd": None},
}


@dataclass(frozen=True)
class TraderPolicy:
    mode: str
    rollout_stage: str
    allowed_chain_ids: tuple[int, ...]
    rollout_cap_usd: Optional[float]
    max_position_pct: float
    max_daily_loss_pct: float
    max_slippage_bps: int
    max_write_tools_per_hour: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "TraderPolicy":
        raw = data if isinstance(data, Mapping) else {}

        mode = str(raw.get("mode", "paper")).strip().lower()
        if mode not in {"paper", "live"}:
            raise ValueError("trader.mode must be 'paper' or 'live'")

        stage_default = "paper" if mode == "paper" else "canary"
        stage = str(raw.get("rollout_stage", stage_default)).strip().lower()
        if stage not in ROLLOUT_PRESETS:
            raise ValueError(
                "trader.rollout_stage must be paper, canary, limited, or steady"
            )

        configured = raw.get("allowed_chains", ["base"])
        if isinstance(configured, (str, int)):
            configured = [configured]
        if not isinstance(configured, (list, tuple, set)):
            raise ValueError("trader.allowed_chains must be a list")

        chain_ids: list[int] = []
        for value in configured:
            if isinstance(value, int):
                chain_id = value
            else:
                text = str(value).strip().lower()
                if text.isdigit():
                    chain_id = int(text)
                else:
                    chain_id = CHAIN_NAME_TO_ID.get(text, 0)
            if chain_id <= 0:
                raise ValueError(f"unsupported chain in trader.allowed_chains: {value!r}")
            if chain_id not in chain_ids:
                chain_ids.append(chain_id)

        preset_chains = ROLLOUT_PRESETS[stage]["chains"]
        if preset_chains is not None:
            chain_ids = [chain_id for chain_id in chain_ids if chain_id in preset_chains]
        if not chain_ids:
            raise ValueError("rollout policy leaves no allowed chains")

        preset_cap = ROLLOUT_PRESETS[stage]["cap_usd"]
        configured_cap = _optional_positive_float(raw.get("rollout_capital_cap_usd"))
        if preset_cap is None:
            cap = configured_cap
        elif configured_cap is None:
            cap = float(preset_cap)
        else:
            cap = min(float(preset_cap), configured_cap)

        max_position_pct = float(raw.get("max_position_pct", 5.0))
        max_daily_loss_pct = float(raw.get("max_daily_loss_pct", 3.0))
        max_slippage_bps = int(raw.get("max_slippage_bps", 100))
        max_writes = int(raw.get("max_write_tools_per_hour", 10))

        if not 0 < max_position_pct <= 100:
            raise ValueError("max_position_pct must be >0 and <=100")
        if not 0 < max_daily_loss_pct <= 100:
            raise ValueError("max_daily_loss_pct must be >0 and <=100")
        if not 0 < max_slippage_bps <= 10_000:
            raise ValueError("max_slippage_bps must be >0 and <=10000")
        if not 0 < max_writes <= 10_000:
            raise ValueError("max_write_tools_per_hour must be >0")

        return cls(
            mode=mode,
            rollout_stage=stage,
            allowed_chain_ids=tuple(chain_ids),
            rollout_cap_usd=cap,
            max_position_pct=max_position_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            max_slippage_bps=max_slippage_bps,
            max_write_tools_per_hour=max_writes,
        )


def _optional_positive_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    number = float(value)
    if number <= 0:
        raise ValueError("rollout_capital_cap_usd must be positive")
    return number
