#!/usr/bin/env python3
"""Summarize micro-compaction telemetry from Hermes logs.

Reads the content-free ``micro compaction telemetry:`` JSON lines emitted by
``ContextCompressor._emit_micro_compaction_telemetry`` and reports what the
feature actually did, per session and overall.

Usage:
  python scripts/micro_compaction_report.py [LOGFILE ...]
  python scripts/micro_compaction_report.py --per-session

With no LOGFILE, reads ``$HERMES_HOME/logs/agent.log`` (default ~/.hermes).

Note on reading the numbers: the first pass in a session inserts the summary
marker, which costs a fixed ~400 tokens of scaffolding. That overhead is paid
once; from the second pass on the marker is replaced rather than added, so
each absorbed exchange is pure saving. A session with a single pass can
therefore show a net loss and still be working correctly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

MARKER = "micro compaction telemetry: "


def default_log() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "logs" / "agent.log"


def load(paths: list[Path]) -> list[dict]:
    events: list[dict] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        for line in text.splitlines():
            idx = line.find(MARKER)
            if idx == -1:
                continue
            try:
                events.append(json.loads(line[idx + len(MARKER):]))
            except ValueError:
                continue
    return events


def fmt(n: int | None) -> str:
    return "-" if n is None else f"{n:,}"


def report(events: list[dict], per_session: bool) -> int:
    if not events:
        print("No micro-compaction telemetry found.")
        print("Micro-compaction may be disabled (compression.micro_compact),")
        print("or no session has run long enough to trigger a pass yet.")
        return 1

    by_session: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_session[e.get("session_id") or "(unknown)"].append(e)

    outcomes: dict[str, int] = defaultdict(int)
    for e in events:
        outcomes[e.get("outcome", "?")] += 1

    saved = sum(-(e.get("tokens_delta") or 0) for e in events)
    absorbed = [e for e in events if e.get("outcome") == "absorbed"]
    exchange_tokens = [e.get("exchange_tokens") or 0 for e in absorbed]
    durations = [e.get("duration_ms") or 0 for e in events]

    if per_session:
        print(f"{'session':<38} {'passes':>7} {'saved':>10} {'first':>8} {'last':>8}")
        print("-" * 76)
        for sid, evs in sorted(by_session.items(), key=lambda kv: -len(kv[1])):
            s = sum(-(e.get("tokens_delta") or 0) for e in evs)
            first = evs[0].get("tokens_before")
            last = evs[-1].get("tokens_after")
            print(f"{sid[:38]:<38} {len(evs):>7} {s:>+10,} {fmt(first):>8} {fmt(last):>8}")
        print()

    print(f"sessions                {len(by_session):,}")
    print(f"passes                  {len(events):,}")
    for name, count in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20}  {count:,}")
    print(f"net tokens saved        {saved:+,}")
    if absorbed:
        print(f"exchanges absorbed      {len(absorbed):,}")
        print(f"  mean exchange size    {sum(exchange_tokens) // len(absorbed):,} tokens")
        print(f"  mean saving/pass      {saved // len(events):+,} tokens")
    if durations:
        ordered = sorted(durations)
        print(f"pass duration  median   {ordered[len(ordered) // 2]:,} ms")
        print(f"               max      {ordered[-1]:,} ms")

    multi = {s: e for s, e in by_session.items() if len(e) > 1}
    if multi:
        net = sum(
            (evs[-1].get("tokens_after") or 0) - (evs[0].get("tokens_before") or 0)
            for evs in multi.values()
        )
        print()
        print(f"sessions with >1 pass   {len(multi):,}")
        print(f"  net context change    {net:+,} tokens (negative is shrinkage)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", type=Path, help="log files (default: agent.log)")
    ap.add_argument("--per-session", action="store_true", help="break down by session")
    args = ap.parse_args()

    paths = args.logs or [default_log()]
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"warning: {p} does not exist", file=sys.stderr)
    return report(load([p for p in paths if p.exists()]), args.per_session)


if __name__ == "__main__":
    sys.exit(main())
