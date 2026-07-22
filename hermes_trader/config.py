"""Configuration loader for Hermes Agentic Trader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional

import yaml

from hermes_trader.tools import TraderMode

DEFAULT_CONFIG_RELATIVE = Path("config") / "hermes_trader.yaml"
TRADER_HOME_SUBDIR = Path("trader")


@dataclass
class TraderConfig:
    mode: TraderMode = "paper"
    primary_chain: str = "base"
    allowed_chains: List[str] = field(default_factory=lambda: ["base", "ethereum", "arbitrum"])
    max_position_pct: float = 5.0
    max_daily_loss_pct: float = 3.0
    max_slippage_bps: int = 100
    scan_interval_minutes: int = 15
    min_pool_liquidity_usd: float = 100_000.0
    mcp_server_name: str = "defi-trading"
    min_confidence: float = 0.6
    memory_retrieval_limit: int = 3
    reflection_loss_threshold_usd: float = 5.0
    calibration_miscalibration_gap: float = 0.15
    rollout_stage: str = "steady"
    rollout_capital_cap_usd: Optional[float] = None
    max_write_tools_per_hour: int = 10
    consecutive_loss_alert_count: int = 3
    gate_reject_spike_threshold: int = 10
    enable_size_modifier: bool = False
    pool_data_fallback: str = "auto"  # auto | defillama | mcp | none

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TraderConfig":
        allowed = data.get("allowed_chains") or ["base", "ethereum", "arbitrum"]
        if isinstance(allowed, str):
            allowed = [allowed]
        mode = str(data.get("mode", "paper")).strip().lower()
        if mode not in ("paper", "live"):
            raise ValueError(f"Invalid trader mode: {mode!r} (expected 'paper' or 'live')")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            primary_chain=str(data.get("primary_chain", "base")).strip().lower(),
            allowed_chains=[str(c).strip().lower() for c in allowed],
            max_position_pct=float(data.get("max_position_pct", 5.0)),
            max_daily_loss_pct=float(data.get("max_daily_loss_pct", 3.0)),
            max_slippage_bps=int(data.get("max_slippage_bps", 100)),
            scan_interval_minutes=int(data.get("scan_interval_minutes", 15)),
            min_pool_liquidity_usd=float(data.get("min_pool_liquidity_usd", 100_000.0)),
            mcp_server_name=str(data.get("mcp_server_name", "defi-trading")),
            min_confidence=float(data.get("min_confidence", 0.6)),
            memory_retrieval_limit=int(data.get("memory_retrieval_limit", 3)),
            reflection_loss_threshold_usd=float(data.get("reflection_loss_threshold_usd", 5.0)),
            calibration_miscalibration_gap=float(data.get("calibration_miscalibration_gap", 0.15)),
            rollout_stage=str(data.get("rollout_stage", "steady")).strip().lower(),
            rollout_capital_cap_usd=_optional_float(data.get("rollout_capital_cap_usd")),
            max_write_tools_per_hour=int(data.get("max_write_tools_per_hour", 10)),
            consecutive_loss_alert_count=int(data.get("consecutive_loss_alert_count", 3)),
            gate_reject_spike_threshold=int(data.get("gate_reject_spike_threshold", 10)),
            enable_size_modifier=bool(data.get("enable_size_modifier", False)),
            pool_data_fallback=str(data.get("pool_data_fallback", "auto")).strip().lower(),
        )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def default_config_paths() -> list[Path]:
    """Search order: env override, repo config, ~/.hermes/trader/."""
    paths: list[Path] = []
    env_path = os.environ.get("HERMES_TRADER_CONFIG", "").strip()
    if env_path:
        paths.append(Path(env_path))
    repo_root = Path(__file__).resolve().parent.parent
    paths.append(repo_root / DEFAULT_CONFIG_RELATIVE)
    paths.append(_hermes_home() / TRADER_HOME_SUBDIR / "hermes_trader.yaml")
    return paths


def load_trader_config(path: Optional[Path | str] = None) -> TraderConfig:
    if path is not None:
        return _load_file(Path(path))
    for candidate in default_config_paths():
        if candidate.is_file():
            return _load_file(candidate)
    return TraderConfig()


def _load_file(path: Path) -> TraderConfig:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config root must be a mapping")
    return TraderConfig.from_mapping(data)