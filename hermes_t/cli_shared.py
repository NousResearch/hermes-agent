"""Shared CLI helpers for hermes_t runtime entrypoints."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from hermes_t.store import TradingStateStore, validate_profile_id


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    symbol: str
    trade_unit: int
    max_trades: int = 4


def _env_or_default(name: str, legacy_name: str, default: str | Path) -> str | Path:
    return os.getenv(name) or os.getenv(legacy_name) or default


def _validated_non_blank_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _validated_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def build_runtime_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hermes_t")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(_env_or_default("HERMES_T_BASE_DIR", "HERMES_OLIN_BASE_DIR", Path.home() / ".hermes_t_runtime")),
    )
    parser.add_argument(
        "--profile-id",
        default=_env_or_default("HERMES_T_PROFILE_ID", "HERMES_OLIN_PROFILE_ID", "default"),
    )
    parser.add_argument(
        "--symbol",
        default=_env_or_default("HERMES_T_SYMBOL", "HERMES_OLIN_SYMBOL", None),
    )
    parser.add_argument(
        "--trade-unit",
        type=int,
        default=_env_or_default("HERMES_T_TRADE_UNIT", "HERMES_OLIN_TRADE_UNIT", None),
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=int(_env_or_default("HERMES_T_MAX_TRADES", "HERMES_OLIN_MAX_TRADES", 4)),
    )
    parser.add_argument(
        "--signal",
        default=_env_or_default("HERMES_T_SIGNAL", "HERMES_OLIN_SIGNAL", "hold"),
    )
    parser.add_argument(
        "--score",
        type=int,
        default=int(_env_or_default("HERMES_T_SCORE", "HERMES_OLIN_SCORE", 50)),
    )
    parser.add_argument("--tech-data-config", type=Path, default=None)
    parser.add_argument("--quote-data-config", type=Path, default=None)
    parser.add_argument("--quote-snapshot-config", type=Path, default=None)
    parser.add_argument("--profiles-config", type=Path, default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--dispatch", action="store_true")
    return parser


def build_runtime_profile_from_args(args: argparse.Namespace) -> RuntimeProfile:
    return RuntimeProfile(
        profile_id=validate_profile_id(args.profile_id),
        symbol=_validated_non_blank_string(args.symbol, "symbol"),
        trade_unit=_validated_positive_int(args.trade_unit, "trade_unit"),
        max_trades=_validated_positive_int(args.max_trades, "max_trades"),
    )


def build_runtime_store(
    base_dir: str | Path | None,
    profile: RuntimeProfile,
    *,
    prefer_legacy_olin_store: bool,
) -> TradingStateStore:
    del prefer_legacy_olin_store
    return TradingStateStore(base_dir=base_dir, profile_id=profile.profile_id)
