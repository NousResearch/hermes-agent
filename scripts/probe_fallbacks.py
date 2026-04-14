#!/usr/bin/env python3
"""Low-lift concurrent fallback probe for Hermes provider lanes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from agent.fallback_probe import (
    ProbeRoute,
    build_probe_command,
    load_routes_from_config,
    normalize_cli_provider,
    probe_routes,
    run_probe,
    summary_as_json,
)


def _parse_route_arg(raw: str) -> ProbeRoute:
    provider, sep, model = raw.partition(":")
    if not sep or not provider.strip() or not model.strip():
        raise argparse.ArgumentTypeError("route must be provider:model")
    provider = provider.strip()
    return ProbeRoute(provider=provider, cli_provider=normalize_cli_provider(provider), model=model.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe configured Hermes fallback providers with low-lift concurrent PONG checks")
    parser.add_argument("--config", type=Path, default=Path("~/.hermes/config.yaml").expanduser())
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--route", action="append", type=_parse_route_arg, default=[])
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    routes = list(args.route) or load_routes_from_config(args.config)
    if not routes:
        print("No fallback routes configured.", file=sys.stderr)
        return 2

    summary = probe_routes(
        routes,
        repo_root=args.repo_root,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )

    if args.json:
        print(summary_as_json(summary))
    else:
        print(f"Fallback probe summary: {summary['passed']} passed, {summary['failed']} failed")
        for item in summary["results"]:
            status = "PASS" if item["ok"] else "FAIL"
            print(f"- {status} {item['provider']} / {item['model']} [{item['classification']}]")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
