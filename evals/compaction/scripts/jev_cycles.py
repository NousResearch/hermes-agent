#!/usr/bin/env python3
"""Simulate repeated Jev compaction over a growing session.

Usage: jev_cycles.py <lineage_json> <threshold_tokens> <max_cycles>

Feed a lineage chronologically; whenever the estimated context crosses the
threshold, compact with the Jev arm and record the cycle. Stops when the
transcript ends, Jev cannot fit its state (plugin fallback), or a compaction
frees nothing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evals.compaction.fixtures import estimate_tokens, load_transcript  # noqa: E402
from evals.compaction.jev_arm import JevCompactor, JevOptions, collect_tool_calls  # noqa: E402

path, threshold, max_cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
name = Path(path).stem
msgs = load_transcript(path)
ctx, i, cycles, total_jev_usd = [], 0, [], 0.0


def tokens(ms):
    return sum(estimate_tokens(m) for m in ms)


def text_floor(ms):
    return sum(estimate_tokens(m) for m in ms if m.get("role") != "tool" and not m.get("tool_calls"))


while i < len(msgs) and len(cycles) < max_cycles:
    ctx.append(msgs[i]); i += 1
    if tokens(ctx) < threshold or msgs[i - 1].get("tool_calls"):
        continue  # only compact on a well-formed boundary (result rows present)
    jc = JevCompactor(options=JevOptions())
    before = tokens(ctx)
    try:
        out = jc.compress(ctx)
    except ValueError as e:
        cycles.append({"cycle": len(cycles) + 1, "at_msg": i, "before": before, "fallback": str(e)[:90],
                       "floor": text_floor(ctx),
                       "calls": len(collect_tool_calls(ctx, jc.opt.preserve_recent_messages))})
        break
    total_jev_usd += jc.usage.cost_usd
    after = tokens(out)
    cycles.append({"cycle": len(cycles) + 1, "at_msg": i, "before": before, "after": after,
                   "freed_pct": round(100 * (before - after) / before, 1), "floor": text_floor(out),
                   "candidates": len(jc.decisions) - jc.stats["pinned"], "dropped": jc.stats["calls_dropped"],
                   "stage": jc.stats["state_stage"], "state_tok": jc.stats["state_tokens"],
                   "requests": jc.stats["requests"], "usd": round(jc.usage.cost_usd, 4)})
    if after >= threshold:
        cycles[-1]["stuck"] = True
        break
    ctx = out

print(json.dumps({"transcript": name, "threshold": threshold, "raw_tokens_consumed": tokens(msgs[:i]),
                  "raw_total": tokens(msgs), "msgs_consumed": i, "cycles": cycles,
                  "jev_usd_total": round(total_jev_usd, 4)}, indent=1))
