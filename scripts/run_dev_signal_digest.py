#!/usr/bin/env python3.11
"""Run one Hermes Dev production-signal digest with a simple sentinel lock."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.dev_control.product_events import DevProductEventStore  # noqa: E402
from gateway.dev_control.production_signals import DevProductionSignalStore, run_signal_digest_sources  # noqa: E402
from gateway.dev_control.reliability import DevReliabilityStore  # noqa: E402
from gateway.dev_execution import DevExecutionStore  # noqa: E402
from gateway.subagent_events import SubagentEventStore  # noqa: E402
from hermes_state import DEFAULT_DB_PATH  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the advisory Dev production-signal digest.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument(
        "--sources",
        default=os.getenv("HERMES_DEV_SIGNAL_DIGEST_SOURCES", "deterministic,product,reliability"),
        help="Comma-separated signal sources to run. Defaults to deterministic,product,reliability.",
    )
    parser.add_argument("--window-days", type=float, default=None)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--lock-path", default=os.getenv("HERMES_DEV_SIGNAL_DIGEST_LOCK", "/tmp/hermes_dev_signal_digest.lock"))
    args = parser.parse_args()

    lock_path = Path(args.lock_path)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print(json.dumps({"ok": False, "status": "already_running", "lock_path": str(lock_path)}))
        return 0

    try:
        os.write(fd, str(time.time()).encode("utf-8"))
        os.close(fd)
        db_path = Path(args.db_path)
        signal_store = DevProductionSignalStore(db_path)
        event_store = SubagentEventStore(db_path)
        product_event_store = DevProductEventStore(db_path)
        reliability_store = DevReliabilityStore(db_path)
        execution_store = DevExecutionStore(db_path)
        filters = {"project_id": args.project_id} if args.project_id else {}
        sources = [source.strip() for source in str(args.sources or "").split(",") if source.strip()]
        result = run_signal_digest_sources(
            signal_store=signal_store,
            event_store=event_store,
            product_event_store=product_event_store,
            reliability_store=reliability_store,
            execution_store=execution_store,
            sources=sources,
            window_days=args.window_days,
            filters=filters,
            persist=True,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("ok") else 1
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
