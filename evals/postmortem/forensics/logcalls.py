"""Lane 1b: per-call cache behaviour from ``agent.log`` (OBSERVED provider usage per call).

Hermes logs one line per API call::

    ... INFO [<session_id>] agent.conversation_loop: API call #N: model=... in=<prompt> out=<out> total=... latency=..s cache=<hit>/<total>

Given the rotated logs, this reproduces: prompt-size distribution, cache hit-ratio buckets, the
"plateau" signature of a broken cache prefix (hit count stuck at the previous call's breakpoint while
``in`` grows), the share of uncached input those plateaus explain, and the sawtooth replay of an absolute
context cap on REAL per-call prompt sizes (the figure the tokens lane reported as -49%).

    python -m evals.postmortem.forensics.logcalls --db state_copy.db --logs ~/.hermes/logs/agent.log*

Coverage caveat: logs rotate; report the fraction of the run's calls that were found before quoting
anything from this lane, and treat extrapolations as upper bounds.
"""
from __future__ import annotations

import collections
import glob
import re
import statistics
from typing import Any, Dict, List

from evals.postmortem.forensics.common import Run

_LINE = re.compile(
    r"^(\S+ \S+) INFO \[(\S+)\] agent\.conversation_loop: API call #(\d+): model=(\S+) provider=\S+ "
    r"in=(\d+) out=(\d+) total=\d+ latency=([\d.]+)s cache=(\d+)/(\d+)"
)


def parse_logs(paths: List[str], sids: set) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for path in paths:
        try:
            fh = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                m = _LINE.match(line)
                if m and m.group(2) in sids:
                    calls.append({"ts": m.group(1), "sid": m.group(2), "n": int(m.group(3)), "model": m.group(4),
                                  "inp": int(m.group(5)), "out": int(m.group(6)), "lat": float(m.group(7)),
                                  "hit": int(m.group(8))})
    return calls


def sawtooth_real(by_sid: Dict[str, List[Dict[str, Any]]], cap: int, floor: int) -> float:
    """Replay on REAL prompt sizes: appended = in[i] - in[i-1]; compress to floor when above cap."""
    total = 0.0
    for calls in by_sid.values():
        calls = sorted(calls, key=lambda c: c["n"])
        ctx = None
        for i, c in enumerate(calls):
            appended = c["inp"] if i == 0 else max(0, c["inp"] - calls[i - 1]["inp"])
            ctx = c["inp"] if ctx is None else ctx + appended
            if ctx > cap:
                ctx = float(floor)
            total += ctx
    return total


def main(argv=None) -> int:
    ap = Run.parser((__doc__ or "").split("\n\n")[0])
    ap.add_argument("--logs", nargs="+", required=True, help="agent.log files (globs ok)")
    ap.add_argument("--cap", type=int, default=200_000)
    ap.add_argument("--floor", type=int, default=65_000)
    a = ap.parse_args(argv)
    run = Run.open(a.db, root=a.root, out=a.out)
    paths = sorted(p for g in a.logs for p in glob.glob(g))
    calls = parse_logs(paths, set(run.in_run))
    total_calls = run.summary()["api_calls"]
    if not calls:
        print("[logcalls] no matching API-call lines found in", paths); return 1
    inp = [c["inp"] for c in calls]
    hit_total, in_total = sum(c["hit"] for c in calls), sum(inp)
    buckets = collections.Counter()
    for c in calls:
        r = c["hit"] / c["inp"] if c["inp"] else 0
        buckets["<50%" if r < .5 else "50-90%" if r < .9 else "90-97%" if r < .97 else "97-99%" if r < .99 else ">=99%"] += 1
    by_sid: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for c in calls:
        by_sid[c["sid"]].append(c)
    # Two definitions, reported separately (an independent review caught them being conflated):
    #   strict plateau: hit count EXACTLY equal to the previous call's (prefix stuck at the old breakpoint)
    #   non-advancing:  hit count did not grow while input did (includes partial misses of other causes)
    strict_pairs = strict_uncached = loose_pairs = loose_uncached = pairs = uncached_total = 0
    for cs in by_sid.values():
        cs.sort(key=lambda c: c["n"])
        for prev, cur in zip(cs, cs[1:]):
            pairs += 1
            u = max(0, cur["inp"] - cur["hit"]); uncached_total += u
            if cur["inp"] > prev["inp"]:
                if cur["hit"] == prev["hit"]:
                    strict_pairs += 1; strict_uncached += u
                if cur["hit"] <= prev["hit"]:
                    loose_pairs += 1; loose_uncached += u
    price = run.price_per_token
    real = sum(inp); capped = sawtooth_real(by_sid, a.cap, a.floor)
    report = {
        "coverage": {"calls_found": len(calls), "run_api_calls": total_calls, "fraction": round(len(calls) / total_calls, 4) if total_calls else None,
                     "first_ts": min(c["ts"] for c in calls), "last_ts": max(c["ts"] for c in calls),
                     "note": "rotated logs; extrapolations from this window are upper bounds"},
        "observed": {
            "prompt_tokens": {"median": int(statistics.median(inp)), "p90": sorted(inp)[int(.9 * len(inp))], "max": max(inp)},
            "share_of_calls_above": {t: round(sum(1 for x in inp if x > t) / len(inp), 3) for t in (150_000, 200_000, 300_000)},
            "cache_hit_ratio_overall": round(hit_total / in_total, 4),
            "hit_ratio_buckets": dict(buckets),
            "strict_plateau": {"pairs_share": round(strict_pairs / pairs, 4) if pairs else None,
                               "share_of_uncached_input": round(strict_uncached / uncached_total, 4) if uncached_total else None},
            "non_advancing_hit": {"pairs_share": round(loose_pairs / pairs, 4) if pairs else None,
                                  "share_of_uncached_input": round(loose_uncached / uncached_total, 4) if uncached_total else None},
            "uncached_input_usd_in_window": round(uncached_total * price["cache_write_tokens"], 2),
        },
        "modeled": {
            "sawtooth_on_real_prompt_sizes": {
                "cap": a.cap, "floor": a.floor, "prompt_tokens_ratio": round(capped / real, 3) if real else None,
                "caveats": ["compression-call cost excluded", "read/write ratio assumed unchanged", "window only"],
            }
        },
    }
    path = run.write("logcalls.json", report)
    o = report["observed"]
    print(f"[logcalls] coverage {len(calls):,}/{total_calls:,} calls ({report['coverage']['fraction']:.1%}) from {len(paths)} file(s)")
    print(f"[logcalls] OBSERVED median prompt {o['prompt_tokens']['median']:,} p90 {o['prompt_tokens']['p90']:,}; >200K: {o['share_of_calls_above'][200000]:.0%}; hit ratio {o['cache_hit_ratio_overall']:.1%}")
    print(f"[logcalls] OBSERVED strict plateau (hit unchanged): {o['strict_plateau']['pairs_share']:.1%} of pairs, {o['strict_plateau']['share_of_uncached_input']:.1%} of uncached input; "
          f"non-advancing hit: {o['non_advancing_hit']['pairs_share']:.1%} / {o['non_advancing_hit']['share_of_uncached_input']:.1%}; uncached in window ${o['uncached_input_usd_in_window']:,}")
    print(f"[logcalls] MODELED sawtooth cap {a.cap}/{a.floor} on real sizes: prompt volume x{report['modeled']['sawtooth_on_real_prompt_sizes']['prompt_tokens_ratio']}")
    print(f"[logcalls] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
