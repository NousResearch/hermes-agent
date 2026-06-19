"""CLI for deterministic investment-assistant data mining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data_miner import DEFAULT_LAYERS, FMP_LAYER_SPECS, build_data_files_from_triage


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return _prepare(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-data-miner",
        description="Prepare offline investment-assistant data files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Fetch and persist deterministic source data for selected symbols.",
    )
    prepare.add_argument("--triage", help="Candidate triage JSON path.")
    prepare.add_argument(
        "--queue",
        choices=("deep", "watch", "all"),
        default="deep",
        help="Queue to select from the triage artifact.",
    )
    prepare.add_argument(
        "--symbols",
        nargs="*",
        default=[],
        help="Symbols to fetch directly, e.g. US.MRVL MRVL.",
    )
    prepare.add_argument(
        "--output-root",
        default="data/investment_assistant",
        help="Output root for symbol data files.",
    )
    prepare.add_argument(
        "--layers",
        default=",".join(DEFAULT_LAYERS),
        help=(
            "Comma-separated layers. Core: sec,filing_metadata,filing_text,filing_sections,etf. "
            f"FMP: {','.join(FMP_LAYER_SPECS)} or fmp_all."
        ),
    )
    prepare.add_argument("--market", default="US", help="Default market prefix for bare tickers.")
    prepare.add_argument("--max-symbols", type=int, help="Maximum selected symbols to fetch.")
    prepare.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing files.")
    prepare.add_argument("--force", action="store_true", help="Overwrite files even with --skip-existing.")
    prepare.add_argument("--json", action="store_true", help="Print the full run artifact as JSON.")
    return parser


def _prepare(args: argparse.Namespace) -> int:
    if not args.triage and not args.symbols:
        raise SystemExit("prepare requires --triage or --symbols")
    run = build_data_files_from_triage(
        triage_state_path=args.triage,
        output_root=args.output_root,
        queue=args.queue,
        symbols=args.symbols or None,
        layers=_parse_layers(args.layers),
        max_symbols=args.max_symbols,
        market=args.market,
        skip_existing=args.skip_existing,
        force=args.force,
    )
    if args.json:
        print(json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"run_id: {run.run_id}")
    print(f"output_root: {Path(run.output_root)}")
    print(f"symbols: {', '.join(run.symbols) if run.symbols else '(none)'}")
    print(f"layers: {', '.join(run.layers)}")
    print(f"status_counts: {run.status_counts}")
    if run.warnings:
        print("warnings:")
        for warning in run.warnings:
            print(f"  - {warning}")
    return 0


def _parse_layers(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
