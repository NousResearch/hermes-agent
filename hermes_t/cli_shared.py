from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

from hermes_olin.profile import DEFAULT_RUNTIME_PROFILE, RuntimeProfile
from hermes_olin.store import OlinStateStore, TradingStateStore
from hermes_t.orchestrator import _build_runtime_profile_from_item


def parse_trade_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("trade-date must use YYYYMMDD format") from exc
    return value


def build_runtime_parser(
    *,
    description: str,
    default_base_dir_name: str,
    channel_env_vars: Sequence[str],
    chat_id_env_vars: Sequence[str] = (),
    thread_id_env_vars: Sequence[str] = (),
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--base-dir",
        default=str(Path.home() / default_base_dir_name),
        help="Runtime working directory. State files are stored under profiles/<profile-id>/state/realtime/",
    )
    parser.add_argument("--profile-id", default=DEFAULT_RUNTIME_PROFILE.profile_id, help="Runtime profile id used to scope state and metadata.")
    parser.add_argument("--symbol", default=DEFAULT_RUNTIME_PROFILE.symbol, help="Stock symbol for this runtime profile.")
    parser.add_argument("--trade-unit", type=int, default=DEFAULT_RUNTIME_PROFILE.trade_unit, help="Shares per execution unit for this runtime profile.")
    parser.add_argument("--max-trades", type=int, default=DEFAULT_RUNTIME_PROFILE.max_trades, help="Max intraday T trades per side for this runtime profile.")
    parser.add_argument("--trade-date", type=parse_trade_date, default=datetime.now().strftime("%Y%m%d"), help="Effective trade date, format YYYYMMDD")
    parser.add_argument("--signal", choices=["buy", "sell", "hold"], default="hold", help="Summary signal for this cycle")
    parser.add_argument("--score", type=int, default=50, help="Total score used to decide buy/sell thresholds")
    parser.add_argument("--dispatch", action="store_true", help="Dispatch pending signal through Hermes gateway")
    parser.add_argument("--channel", default=_env_first(channel_env_vars, "feishu"), help="Dispatch channel, default feishu.")
    parser.add_argument("--chat-id", default=_env_first(chat_id_env_vars, None), help="Explicit dispatch chat_id.")
    parser.add_argument("--thread-id", default=_env_first(thread_id_env_vars, None), help="Explicit dispatch thread_id.")
    return parser


def build_runtime_profile_from_args(args: argparse.Namespace) -> RuntimeProfile:
    return _build_runtime_profile_from_item(
        item={
            "profile_id": args.profile_id,
            "symbol": args.symbol,
            "trade_unit": args.trade_unit,
            "max_trades": args.max_trades,
        },
        idx=0,
    )


def build_runtime_store(*, base_dir: str | os.PathLike[str], profile: RuntimeProfile, prefer_legacy_olin_store: bool) -> TradingStateStore:
    if prefer_legacy_olin_store and profile == DEFAULT_RUNTIME_PROFILE:
        return OlinStateStore(base_dir)
    return TradingStateStore(base_dir, profile=profile)


def _env_first(names: Sequence[str], fallback: str | None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return fallback
