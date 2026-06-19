"""CLI for budgeted websearch investment theme discovery."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Sequence

from .websearch_discovery import (
    build_native_websearch_discovery_plan,
    build_two_stage_websearch_discovery_plan,
    build_websearch_discovery_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.progress:
        os.environ["IA_WEBSEARCH_DISCOVERY_PROGRESS"] = "1"
    try:
        with _operation_timeout(args.timeout_seconds, "websearch-discovery"):
            if args.native:
                builder = build_native_websearch_discovery_plan
            elif args.two_stage:
                builder = build_two_stage_websearch_discovery_plan
            else:
                builder = build_websearch_discovery_plan
            plan = builder(
                args.theme,
                market=args.market,
                theme_description=args.description,
                required_symbols=args.required_symbol,
                max_searches=args.max_searches,
                max_results=args.max_results,
            )
    except TimeoutError as exc:
        print(f"[{time.strftime('%H:%M:%S')}] {exc}", file=sys.stderr)
        return 124
    payload = plan.model_dump(mode="json")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"output: {output_path}")
        print(f"theme: {plan.theme}")
        print(f"seed_count: {len(plan.seed_symbols)}")
        print(f"domain_count: {len(plan.domain_tree)}")
        search_budget = (plan.pydantic_ai or {}).get("search_budget") or {}
        print(f"successful_searches: {search_budget.get('successful_searches', 0)}")
        for domain in plan.domain_tree:
            symbols = []
            for subdomain in domain.subdomains:
                symbols.extend(candidate.symbol for candidate in subdomain.candidates)
            print(f"- {domain.name}: {', '.join(symbols[:12])}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-websearch-discovery",
        description="Run budgeted websearch-only investment theme discovery.",
    )
    parser.add_argument("--theme", required=True, help="Theme key or natural-language theme.")
    parser.add_argument("--market", default="US", help="Listing market, default US.")
    parser.add_argument("--description", default="", help="User request or theme description.")
    parser.add_argument(
        "--required-symbol",
        action="append",
        default=[],
        help="Required/base symbol. Repeatable.",
    )
    parser.add_argument("--max-searches", type=int, default=None, help="Hard search budget.")
    parser.add_argument("--max-results", type=int, default=None, help="Max results per search.")
    parser.add_argument("--output", required=True, help="Output ThemeDiscoveryPlan JSON path.")
    parser.add_argument("--json", action="store_true", help="Print full JSON to stdout.")
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Plan searches first, execute them deterministically, then synthesize candidates.",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Use provider-native web search directly with compact output schema.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=_env_int("IA_WEBSEARCH_DISCOVERY_TIMEOUT_SECONDS", 600),
        help="Wall-clock timeout. Use 0 to disable.",
    )
    parser.add_argument("--progress", action="store_true", help="Enable progress logs.")
    return parser


@contextlib.contextmanager
def _operation_timeout(seconds: int, label: str):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)

    def _timeout_handler(_signum, _frame):
        raise TimeoutError(f"{label} timed out after {seconds}s")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name) or default).strip())
    except ValueError:
        return default


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
