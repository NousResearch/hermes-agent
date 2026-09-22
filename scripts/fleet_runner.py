#!/usr/bin/env python3
"""Run the standalone federated runner against a stock Hermes installation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hermes_cli.fleet_client import FleetClient
from hermes_cli.fleet_protocol import RunnerCapability
from hermes_cli.fleet_runner import FleetRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--token", default=os.environ.get("HERMES_FLEET_TOKEN", ""), help=argparse.SUPPRESS)
    parser.add_argument("--hermes-executable", required=True)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--project", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--tool", action="append", default=["terminal", "git"])
    parser.add_argument("--liveness-file", required=True, help="Marker maintained while local Hermes/Desktop is alive")
    parser.add_argument("--interval", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.token:
        raise SystemExit("HERMES_FLEET_TOKEN is required")
    client = FleetClient(args.coordinator, token=args.token)
    capabilities = [
        RunnerCapability(
            node_id=args.node_id,
            profile=profile,
            models=args.model,
            tools=args.tool,
            projects=args.project,
            platform="windows" if Path(args.hermes_executable).drive else "posix",
        )
        for profile in args.profile
    ]
    runner = FleetRunner(
        args.node_id,
        client,
        args.hermes_executable,
        args.profile,
        args.project,
        capabilities=capabilities,
        liveness_check=lambda: Path(args.liveness_file).is_file(),
    )
    try:
        runner.run_forever(interval=args.interval)
    except KeyboardInterrupt:
        runner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
