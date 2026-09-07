#!/usr/bin/env python3
"""Replay a ~500K-token prefix of a reconstructed lineage through compaction.

Usage: run_lineage_compaction.py <checkout> <lineage_json> <out_json> [cap_tokens]

Takes the chronological prefix of the lineage at the token cap (default 500K =
the 50% trigger on a 1M-context model), aligned to a tool-group boundary, and
runs ContextCompressor.compress() exactly as the live trigger would.
"""
import copy
import json
import sys
import time
from pathlib import Path

CHECKOUT = sys.argv[1]
LINEAGE = sys.argv[2]
OUT = sys.argv[3]
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 500_000

sys.path.insert(0, CHECKOUT)

data = json.load(open(LINEAGE, encoding="utf-8"))
msgs = data["messages"]

def tok(m):
    t = len(m.get("content") or "") // 4
    tc = m.get("tool_calls")
    if tc:
        t += len(json.dumps(tc, default=str)) // 4
    return t

# chronological prefix up to CAP tokens
prefix = []
total = 0
for m in msgs:
    t = tok(m)
    if total + t > CAP and len(prefix) > 10:
        break
    prefix.append(m)
    total += t

# align the end: never end on an assistant msg with tool_calls whose results
# were cut off; drop trailing orphans
while prefix and prefix[-1].get("tool_calls"):
    prefix.pop()
# also drop trailing tool results with no preceding assistant tool_calls kept
# (compress()'s _sanitize_tool_pairs would handle it, but keep input clean)

before_tokens = sum(tok(m) for m in prefix)
print(f"[{Path(CHECKOUT).name}] {Path(LINEAGE).stem}: prefix {len(prefix)} msgs ~{before_tokens:,} tok (cap {CAP:,})")

from agent.context_compressor import ContextCompressor  # noqa: E402

model = "anthropic/claude-fable-5"
comp = ContextCompressor(model=model, quiet_mode=True)
before = copy.deepcopy(prefix)
t0 = time.time()
compressed = comp.compress(prefix, current_tokens=before_tokens, force=True)
dt = time.time() - t0
after_tokens = sum(tok(m) for m in compressed)
print(f"  -> {len(compressed)} msgs ~{after_tokens:,} tok in {dt:.1f}s  (err={getattr(comp,'_last_summary_error',None)})")

json.dump({
    "meta": {
        "checkout": Path(CHECKOUT).name,
        "session_id": data["root"],
        "title": f"lineage {data['root']} ({len(data['chain'])} rotations)",
        "model": model,
        "elapsed_s": round(dt, 1),
        "before_msgs": len(before),
        "after_msgs": len(compressed),
        "before_tokens_est": before_tokens,
        "after_tokens_est": after_tokens,
        "summary_error": getattr(comp, "_last_summary_error", None),
        "todo_injection_block": None,
    },
    "before": before,
    "after": compressed,
}, open(OUT, "w", encoding="utf-8"), default=str)
print(f"  wrote {OUT}")
