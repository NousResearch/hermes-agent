"""Fetch Futu option Greeks for held option contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from .adapters import (
    FutuAdapterError,
    FutuOpenDConfig,
    MarketDataAdapter,
    _check_ret,
    _chunks,
    _import_futu,
    _iter_rows,
    _row_get,
    _safe_float,
    _safe_str,
)
from .exposure_ledger import parse_option_leg
from .storage import new_id, utc_now


class OptionMarketDataArtifact(BaseModel):
    artifact_type: str = "option_market_data"
    artifact_id: str = Field(default_factory=lambda: new_id("omd"))
    generated_at: str = Field(default_factory=utc_now)
    source: str = "futu_market_snapshot"
    option_codes: list[str] = Field(default_factory=list)
    underlying_codes: list[str] = Field(default_factory=list)
    data: list[dict[str, Any]] = Field(default_factory=list)
    underlying_quotes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def fetch_option_market_data_from_futu_portfolio(
    futu_portfolio: dict[str, Any],
    *,
    config: FutuOpenDConfig | None = None,
) -> OptionMarketDataArtifact:
    """Fetch option snapshots and underlying quotes for option positions."""

    option_codes, underlying_codes = _option_codes_from_portfolio(futu_portfolio)
    artifact = OptionMarketDataArtifact(
        option_codes=option_codes,
        underlying_codes=underlying_codes,
    )
    if not option_codes:
        artifact.warnings.append("No option positions found in Futu portfolio.")
        return artifact

    futu = _import_futu()
    adapter = MarketDataAdapter(config)
    quote_ctx = futu.OpenQuoteContext(host=adapter.config.host, port=adapter.config.port)
    try:
        artifact.data = _fetch_option_snapshots(adapter, quote_ctx, futu, option_codes, underlying_codes)
        artifact.underlying_quotes = _fetch_underlying_quotes(adapter, quote_ctx, futu, underlying_codes)
    finally:
        quote_ctx.close()
    return artifact


def build_option_market_data_from_files(
    *,
    futu_portfolio_path: str | Path,
    output_dir: str | Path | None = None,
) -> OptionMarketDataArtifact:
    artifact = fetch_option_market_data_from_futu_portfolio(_read_json(Path(futu_portfolio_path)))
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "option_market_data.json", artifact.model_dump(mode="json"))
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        artifact = build_option_market_data_from_files(
            futu_portfolio_path=args.futu_portfolio_path,
            output_dir=args.output_dir,
        )
        payload = artifact.model_dump(mode="json")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"artifact_id: {artifact.artifact_id}")
            print(f"option_codes: {len(artifact.option_codes)}")
            print(f"underlying_codes: {len(artifact.underlying_codes)}")
            print(f"option_snapshots: {len(artifact.data)}")
            if args.output_dir:
                print(f"output_dir: {args.output_dir}")
            if artifact.warnings:
                print("warnings:")
                for warning in artifact.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-option-market-data",
        description="Fetch Futu option Greeks for option positions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="Fetch option Greeks from Futu market snapshots.")
    fetch.add_argument("--futu-portfolio-path", required=True, help="Raw Futu get_portfolio JSON path.")
    fetch.add_argument("--output-dir", help="Directory for option_market_data.json.")
    fetch.add_argument("--json", action="store_true")
    return parser


def _option_codes_from_portfolio(futu_portfolio: dict[str, Any]) -> tuple[list[str], list[str]]:
    option_codes: list[str] = []
    underlyings: set[str] = set()
    for raw_position in futu_portfolio.get("positions") or []:
        leg = parse_option_leg(raw_position)
        if not leg:
            continue
        option_codes.append(leg.raw_code)
        underlyings.add(leg.underlying)
    return sorted(set(option_codes)), sorted(underlyings)


def _fetch_option_snapshots(
    adapter: MarketDataAdapter,
    quote_ctx,
    futu,
    option_codes: list[str],
    underlying_codes: list[str],
) -> list[dict[str, Any]]:
    underlying_set = set(underlying_codes)
    rows: list[dict[str, Any]] = []
    for chunk in _chunks(option_codes, 20):
        ret, data = adapter._quote_call(quote_ctx.get_market_snapshot, chunk)
        _check_ret(futu, ret, data, f"get_market_snapshot(options:{','.join(chunk)})")
        for _, row in _iter_rows(data):
            code = _safe_str(_row_get(row, "code")).upper()
            leg = parse_option_leg({"code": code, "qty": 1})
            underlying = leg.underlying if leg else ""
            record = _snapshot_option_record(row, underlying)
            if underlying and underlying in underlying_set:
                rows.append(record)
    return rows


def _fetch_underlying_quotes(
    adapter: MarketDataAdapter,
    quote_ctx,
    futu,
    underlying_codes: list[str],
) -> dict[str, dict[str, Any]]:
    quotes: dict[str, dict[str, Any]] = {}
    for chunk in _chunks(underlying_codes, 100):
        ret, data = adapter._quote_call(quote_ctx.get_market_snapshot, chunk)
        _check_ret(futu, ret, data, f"get_market_snapshot(underlyings:{','.join(chunk)})")
        for _, row in _iter_rows(data):
            code = _safe_str(_row_get(row, "code")).upper()
            if not code:
                continue
            quotes[code] = {
                "code": code,
                "name": _safe_str(_row_get(row, "name")),
                "last_price": _safe_float(_row_get(row, "last_price")),
                "bid": _safe_float(_row_get(row, "bid_price")),
                "ask": _safe_float(_row_get(row, "ask_price")),
                "update_time": _safe_str(_row_get(row, "update_time")),
            }
    return quotes


def _snapshot_option_record(row, underlying: str) -> dict[str, Any]:
    return {
        "code": _safe_str(_row_get(row, "code")).upper(),
        "name": _safe_str(_row_get(row, "name")),
        "underlying": underlying,
        "last_price": _safe_float(_row_get(row, "last_price")),
        "bid": _safe_float(_row_get(row, "bid_price")),
        "ask": _safe_float(_row_get(row, "ask_price")),
        "volume": int(_safe_float(_row_get(row, "volume"))),
        "turnover": _safe_float(_row_get(row, "turnover")),
        "option_valid": _row_get(row, "option_valid"),
        "option_type": _safe_str(_row_get(row, "option_type")),
        "option_strike_price": _safe_float(_row_get(row, "option_strike_price")),
        "option_implied_volatility": _safe_float(_row_get(row, "option_implied_volatility")),
        "option_delta": _safe_float(_row_get(row, "option_delta")),
        "option_gamma": _safe_float(_row_get(row, "option_gamma")),
        "option_vega": _safe_float(_row_get(row, "option_vega")),
        "option_theta": _safe_float(_row_get(row, "option_theta")),
        "option_rho": _safe_float(_row_get(row, "option_rho")),
        "option_open_interest": int(_safe_float(_row_get(row, "option_open_interest"))),
        "update_time": _safe_str(_row_get(row, "update_time")),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
