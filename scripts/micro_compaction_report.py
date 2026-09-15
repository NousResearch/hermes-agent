#!/usr/bin/env python3
"""Summarize micro-compaction telemetry from Hermes logs.

Reads the content-free JSON lines emitted by
``ContextCompressor._emit_micro_compaction_telemetry`` and reports what the
feature actually bought you.

Usage:
  python scripts/micro_compaction_report.py [LOGFILE ...]
  python scripts/micro_compaction_report.py --per-session

With no LOGFILE, reads ``$HERMES_HOME/logs/agent.log`` (default ~/.hermes).

What to look at
---------------
The point of micro-compaction is not saving tokens or time. It is:

  (a) amortizing the one long batch-compaction pause across many turns, and
  (b) keeping the context window low enough that a session runs much further
      before it needs a hard compaction at all.

So the headline numbers here are OCCUPANCY (how full the window is kept, as a
percentage of the compaction threshold) and BATCH COMPACTIONS (how often the
long pause actually fired). Net tokens saved is reported too, but it is the
least interesting figure -- a session can save nothing on paper and still be a
clear win because the stalls disappeared and the window never filled.

Caveat: running the test suite writes telemetry into the same log. Test lines
cluster inside a sub-second window and carry an empty session_id (they group
as "(unknown)"). Use --per-session to spot them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

MICRO_MARKER = "micro compaction telemetry: "
BATCH_MARKER = "context compression attempt telemetry: "


def default_log() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "logs" / "agent.log"


def load(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    micro: list[dict] = []
    batch: list[dict] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        for line in text.splitlines():
            for marker, sink in ((MICRO_MARKER, micro), (BATCH_MARKER, batch)):
                idx = line.find(marker)
                if idx == -1:
                    continue
                try:
                    sink.append(json.loads(line[idx + len(marker):]))
                except ValueError:
                    pass
                break
    return micro, batch


def pct(values: list[float]) -> tuple[float, float, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[0], ordered[len(ordered) // 2], ordered[-1]


def fmt(n) -> str:
    return "-" if n is None else f"{n:,}"


def report(micro: list[dict], batch: list[dict], per_session: bool) -> int:
    if not micro:
        print("No micro-compaction telemetry found.")
        print("It may be disabled (compression.micro_compact), or no session")
        print("has run long enough to trigger a pass yet.")
        return 1

    by_session: dict[str, list[dict]] = defaultdict(list)
    for e in micro:
        by_session[e.get("session_id") or "(unknown)"].append(e)

    outcomes: dict[str, int] = defaultdict(int)
    for e in micro:
        outcomes[e.get("outcome", "?")] += 1

    occupancies = [e["occupancy_pct"] for e in micro if e.get("occupancy_pct") is not None]
    saved = sum(-(e.get("tokens_delta") or 0) for e in micro)
    absorbed = [e for e in micro if e.get("outcome") == "absorbed"]
    durations = [e.get("duration_ms") or 0 for e in micro]

    if per_session:
        print(f"{'session':<26} {'passes':>6} {'occupancy%':>18} {'batch':>6} {'saved':>10}")
        print("-" * 72)
        batch_by_session: dict[str, int] = defaultdict(int)
        for b in batch:
            batch_by_session[b.get("session_id") or "(unknown)"] += 1
        for sid, evs in sorted(by_session.items(), key=lambda kv: -len(kv[1])):
            occ = [e["occupancy_pct"] for e in evs if e.get("occupancy_pct") is not None]
            spread = pct(occ)
            occ_s = f"{spread[0]:.0f}-{spread[2]:.0f} (med {spread[1]:.0f})" if spread else "-"
            s = sum(-(e.get("tokens_delta") or 0) for e in evs)
            print(f"{sid[:26]:<26} {len(evs):>6} {occ_s:>18} "
                  f"{batch_by_session.get(sid, 0):>6} {s:>+10,}")
        print()

    print("-- headroom ----------------------------------")
    spread = pct(occupancies)
    if spread:
        print(f"context occupancy       min {spread[0]:.0f}%  median {spread[1]:.0f}%  max {spread[2]:.0f}%")
        print("                        (% of the batch-compaction threshold)")
    else:
        print("context occupancy       unavailable (window not resolved when logged)")
    print(f"batch compactions       {len(batch):,}")
    if batch:
        print(f"  micro passes each     {len(micro) / len(batch):.1f}")
    else:
        print("  none fired -- the long pause never happened in this log")

    print()
    print("-- activity ----------------------------------")
    print(f"sessions                {len(by_session):,}")
    print(f"passes                  {len(micro):,}")
    for name, count in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20}  {count:,}")
    if durations:
        ordered = sorted(durations)
        print(f"pass duration           median {ordered[len(ordered) // 2]:,} ms  "
              f"max {ordered[-1]:,} ms")

    print()
    print("-- tokens (least interesting) ----------------")
    print(f"net tokens saved        {saved:+,}")
    if absorbed:
        sizes = [e.get("exchange_tokens") or 0 for e in absorbed]
        print(f"exchanges absorbed      {len(absorbed):,}  "
              f"(mean {sum(sizes) // len(absorbed):,} tokens each)")
    print("note: the first pass in a session costs ~400 tokens of marker")
    print("scaffolding; it pays back from the second pass on.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", type=Path, help="log files (default: agent.log)")
    ap.add_argument("--per-session", action="store_true", help="break down by session")
    args = ap.parse_args()

    paths = args.logs or [default_log()]
    for p in paths:
        if not p.exists():
            print(f"warning: {p} does not exist", file=sys.stderr)
    micro, batch = load([p for p in paths if p.exists()])
    return report(micro, batch, args.per_session)


if __name__ == "__main__":
    sys.exit(main())
