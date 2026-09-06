"""Deterministic local synthetic benchmark; no credentials or inference.

Run: python scripts/benchmark_usage_events.py --events 10000 --queries 100
Uses a temporary profile and the real SessionDB WAL/write/query paths.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def timings(samples):
    ordered = sorted(samples)
    return {"mean_ms": round(statistics.mean(samples) * 1000, 3),
            "p50_ms": round(statistics.median(samples) * 1000, 3),
            "p95_ms": round(ordered[int((len(ordered) - 1) * .95)] * 1000, 3),
            "max_ms": round(max(samples) * 1000, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    if args.events < 1 or args.queries < 1:
        parser.error("counts must be positive")
    with tempfile.TemporaryDirectory(prefix="hermes-usage-bench-") as directory:
        os.environ["HERMES_HOME"] = directory
        from hermes_state import SessionDB
        from hermes_state_usage_events import UsageEvent, WINDOW_US
        from agent.usage_event_capture import observe_execution, record_post_request
        with SessionDB(Path(directory) / "state.db") as db:
            base_bytes = db._read_one("PRAGMA page_count")[0] * db._read_one("PRAGMA page_size")[0]
            as_of = time.time_ns() // 1000
            samples = []
            # Uniform across 6h, away from both edges so benchmark duration cannot evict rows.
            for i in range(args.events):
                event = UsageEvent(str(i), "openai-codex", "synthetic-model", "synthetic-profile",
                                   as_of - WINDOW_US + 60_000_000 + i * (WINDOW_US - 120_000_000) // args.events,
                                   input_tokens=1000, output_tokens=100, cache_read_tokens=500,
                                   cache_write_tokens=None, reasoning_tokens=50, retry_count=0,
                                   status="completed", usage_state="reported")
                before = time.perf_counter()
                assert db.record_usage_event(event)
                samples.append(time.perf_counter() - before)
            query_samples = []
            for _ in range(args.queries):
                before = time.perf_counter()
                result = db.codex_usage_timeline()
                query_samples.append(time.perf_counter() - before)
                assert result["total"]["events"] == args.events
                assert result["total"]["processed_tokens"] == args.events * 1100
                assert sum(b["usage"]["processed_tokens"] for b in result["bins"]) == args.events * 1100
            event_bytes = db._read_one("PRAGMA page_count")[0] * db._read_one("PRAGMA page_size")[0] - base_bytes
            # Measure the whole capture+post-record helper path as well, not just SQL.
            response = SimpleNamespace(usage={"input_tokens": 1000, "output_tokens": 100}, status="completed")
            agent = SimpleNamespace(_session_db=db, provider="openai-codex", api_mode="codex_responses",
                                    model="synthetic-model")
            capture_samples = []
            for _ in range(1000):
                before = time.perf_counter()
                observe_execution(agent, {"model": agent.model}, lambda kw: response, retry_count=0)
                record_post_request(agent, response)
                capture_samples.append(time.perf_counter() - before)
            print(json.dumps({"python": platform.python_version(), "os": platform.system(),
                              "sqlite": sqlite3.sqlite_version, "journal_mode": db._read_one("PRAGMA journal_mode")[0],
                              "synchronous": db._read_one("PRAGMA synchronous")[0],
                              "events": args.events, "queries": args.queries,
                              "ingestion": timings(samples), "query": timings(query_samples),
                              "capture_and_ingestion_1000": timings(capture_samples),
                              "allocated_database_growth_bytes": event_bytes,
                              "all_totals_verified": True}, indent=2))


if __name__ == "__main__":
    main()
