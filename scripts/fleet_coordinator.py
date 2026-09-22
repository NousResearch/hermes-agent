#!/usr/bin/env python3
"""Run the federated fleet coordinator on the Mac host."""

from __future__ import annotations

import argparse
import os
import signal
import threading

from hermes_cli.fleet_server import FleetServer
from hermes_cli.fleet_store import FleetStore
from hermes_cli.fleet_mirror import mirror_fleet_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Coordinator SQLite database path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--token", default=os.environ.get("HERMES_FLEET_TOKEN", ""), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.token:
        parser.error("--token or HERMES_FLEET_TOKEN is required")
    server = FleetServer(
        FleetStore(args.db), token=args.token, host=args.host, port=args.port,
        on_task_update=mirror_fleet_task,
    )
    server.start()
    print(f"Hermes fleet coordinator listening at {server.url}", flush=True)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: stopped.set())
    signal.signal(signal.SIGINT, lambda *_args: stopped.set())
    stopped.wait()
    server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
