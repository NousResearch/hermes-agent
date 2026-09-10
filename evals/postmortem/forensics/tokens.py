"""Lane 1: where the money went, context sizes, and the compression-cap counterfactual.

Reproduces (for any run): cost by bucket and depth, the share of calls above context thresholds,
the excess-cache-write proxy, and the sawtooth replay of an absolute subagent context cap.

    python -m evals.postmortem.forensics.tokens --db state_copy.db [--cap 200000 --floor 65000]

Every figure is labeled in the output as OBSERVED (from usage rows) or MODELED (reconstruction /
replay). The per-call context reconstruction estimates tokens from message character counts
(chars/3.5) plus the system-prompt and tool-schema sizes; it is a diagnostic, not a measurement.
"""
from __future__ import annotations

import collections
import statistics
from typing import Any, Dict, List

from evals.postmortem.forensics.common import Run

CHARS_PER_TOKEN = 3.5
TOOL_SCHEMA_TOKENS = 12_000


def _per_call_context(run: Run, sid: str) -> List[Dict[str, Any]]:
    """Reconstructed prompt size at each assistant turn (MODELED)."""
    ctx = run.system_prompt_len(sid) / CHARS_PER_TOKEN + TOOL_SCHEMA_TOKENS
    calls, appended = [], 0.0
    for m in run.messages(sid, "role, length(coalesce(content,'')) AS lc, length(coalesce(tool_calls,'')) AS ltc"):
        if m["role"] == "assistant":
            calls.append({"ctx": ctx, "appended_since_last": appended})
            out = (m["lc"] + m["ltc"]) / CHARS_PER_TOKEN
            ctx += out; appended = out
        else:
            ctx += m["lc"] / CHARS_PER_TOKEN; appended += m["lc"] / CHARS_PER_TOKEN
    return calls


def sawtooth(calls: List[Dict[str, Any]], cap: int, floor: int) -> float:
    """Prompt tokens if the session compressed to ``floor`` whenever ctx exceeded ``cap`` (MODELED)."""
    total, ctx = 0.0, None
    for c in calls:
        ctx = c["ctx"] if ctx is None else ctx + c["appended_since_last"]
        if ctx > cap:
            ctx = float(floor)
        total += ctx
    return total


def main(argv=None) -> int:
    ap = Run.parser((__doc__ or "").split("\n\n")[0])
    ap.add_argument("--cap", type=int, default=200_000)
    ap.add_argument("--floor", type=int, default=65_000)
    a = ap.parse_args(argv)
    run = Run.open(a.db, root=a.root, out=a.out)
    price = run.price_per_token
    report: Dict[str, Any] = {"summary": run.summary()}

    # OBSERVED: cost by depth, by duration bucket, top-N concentration
    by_depth = collections.defaultdict(float)
    for sid in run.in_run:
        by_depth[run.depth[sid]] += run.cost(sid)
    costs = sorted((run.cost(s) for s in run.in_run), reverse=True)
    total = sum(costs) or 1.0
    long_sessions = [s for s in run.in_run if (float(run.sessions[s].get("ended_at") or 0) - float(run.sessions[s].get("started_at") or 0)) > 3600]
    report["observed"] = {
        "cost_by_depth_usd": {d: round(v, 2) for d, v in sorted(by_depth.items())},
        "top30_share": round(sum(costs[:30]) / total, 3),
        "sessions_over_60min": len(long_sessions),
        "sessions_over_60min_cost_share": round(sum(run.cost(s) for s in long_sessions) / total, 3),
    }

    # MODELED: per-call context distribution, excess cache-write proxy, sawtooth replay
    all_calls, per_session = [], {}
    est_total = actual_total = appended_total = 0.0
    for sid in run.in_run:
        calls = _per_call_context(run, sid)
        if not calls:
            continue
        per_session[sid] = calls
        all_calls.extend(c["ctx"] for c in calls)
        appended_total += sum(c["appended_since_last"] for c in calls) + calls[0]["ctx"]
        est_total += sum(c["ctx"] for c in calls)
        s = run.sessions[sid]
        actual_total += float(s.get("cache_read_tokens") or 0) + float(s.get("cache_write_tokens") or 0) + float(s.get("input_tokens") or 0)
    n = len(all_calls) or 1
    thresholds = {t: round(sum(1 for c in all_calls if c > t) / n, 3) for t in (100_000, 150_000, 200_000, 300_000)}
    cache_write_tokens = sum(float(run.sessions[s].get("cache_write_tokens") or 0) for s in run.in_run)
    excess = max(0.0, cache_write_tokens - appended_total)
    base_prompt = sum(sum(c["ctx"] for c in v) for v in per_session.values())
    capped_prompt = sum(sawtooth(v, a.cap, a.floor) for v in per_session.values())
    ctx_spend = sum(float(run.sessions[s].get(c) or 0) * price[c] for s in run.in_run for c in ("cache_read_tokens", "cache_write_tokens"))
    report["modeled"] = {
        "note": "reconstructed from message sizes at chars/3.5; diagnostic, not a measurement",
        "calls_reconstructed": len(all_calls),
        "median_prompt_tokens": int(statistics.median(all_calls)) if all_calls else 0,
        "share_of_calls_above": thresholds,
        "reconstruction_vs_actual_ratio": round(est_total / actual_total, 3) if actual_total else None,
        "excess_cache_write_proxy": {
            "ideal_tokens_if_only_appended": int(appended_total), "actual_cache_write_tokens": int(cache_write_tokens),
            "excess_tokens": int(excess), "excess_usd": round(excess * price["cache_write_tokens"], 2),
            "caveats": ["token counts estimated", "ideal omits initial system/tool prefix writes", "reasoning accumulation omitted"],
        },
        "sawtooth_replay": {
            "cap": a.cap, "floor": a.floor,
            "prompt_tokens_ratio_capped_vs_actual": round(capped_prompt / base_prompt, 3) if base_prompt else None,
            "context_spend_usd_observed": round(ctx_spend, 2),
            "context_spend_usd_if_capped": round(ctx_spend * capped_prompt / base_prompt, 2) if base_prompt else None,
            "caveats": ["excludes compression-call cost", "assumes read/write ratio unchanged", "ignores re-reads and quality effects"],
        },
    }
    path = run.write("tokens.json", report)
    o, m = report["observed"], report["modeled"]
    print(f"[tokens] {run.summary()['sessions']} sessions, ${run.summary()['cost_usd']:,} | buckets {run.summary()['cost_by_bucket_usd']}")
    print(f"[tokens] OBSERVED depth-2 share {by_depth.get(2,0)/total:.0%}, >60min share {o['sessions_over_60min_cost_share']:.0%}, top30 {o['top30_share']:.1%}")
    print(f"[tokens] MODELED calls >150K ctx {m['share_of_calls_above'][150000]:.0%}; excess cache-write proxy ${m['excess_cache_write_proxy']['excess_usd']:,}")
    print(f"[tokens] MODELED sawtooth cap {a.cap}: prompt volume x{m['sawtooth_replay']['prompt_tokens_ratio_capped_vs_actual']} -> context spend ${m['sawtooth_replay']['context_spend_usd_observed']:,} -> ${m['sawtooth_replay']['context_spend_usd_if_capped']:,}")
    print(f"[tokens] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
