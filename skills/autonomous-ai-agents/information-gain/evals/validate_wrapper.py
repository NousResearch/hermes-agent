#!/usr/bin/env python3
"""validate_wrapper.py — #21 end-to-end: does the wrapper beat a no-clarification baseline?

For each prompt, produce two final responses with the iterate wrapper:
  - baseline : respond directly, answering NO clarifying questions
  - wrapper  : answer the top-K ranked questions via grounded research, then respond
A blind judge (randomized A/B order to kill position bias) picks which better serves the user.
Reframed from top-K-vs-bottom-K (the ranking is already validated via realized_change; the open
question is the product one — does the loop help?). Optional --with-bottom adds the ranking check.

Must run INSIDE the hermes container (grounded answerer shells out to `hermes`). Incremental,
append-across-invocations writes (run per-prompt to survive a container restart):

  docker exec -e OLLAMA_URL=http://host.docker.internal:11434/api/chat -e HERMES_HOME=/opt/data hermes \
    /opt/hermes/.venv/bin/python <this> --ids add-auth --k 1 --out /opt/data/wrapper_validation.json
"""

import argparse
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import iterate  # noqa: E402
import pipeline  # noqa: E402
import testbank  # noqa: E402


def judge(problem, resp_a, resp_b, model, timeout):
    p = ("Two responses, A and B, to the SAME task. Which better serves the user — more correct, "
         "specific, actionable, and appropriately scoped? Penalize vagueness, padding, and wrong "
         "assumptions; reward responses that resolve real uncertainty.\n\n"
         f"TASK:\n{problem}\n\nRESPONSE A:\n{resp_a}\n\nRESPONSE B:\n{resp_b}\n\n"
         'Return ONLY JSON: {"winner": "A", "reason": "one sentence"}  (winner ∈ A|B|tie).')
    obj, _ = pipeline._call_json(model, p, timeout, num_predict=220)
    return obj if isinstance(obj, dict) else {"winner": "tie", "reason": "judge parse error"}


def _blind(problem, left, right, left_name, right_name, rng, model, timeout):
    """Judge left vs right with randomized presentation; return (winner_name, reason)."""
    flip = rng.random() < 0.5
    a, b = (right, left) if flip else (left, right)
    a_name, b_name = (right_name, left_name) if flip else (left_name, right_name)
    v = judge(problem, a, b, model, timeout)
    w = (v.get("winner") or "tie").strip().upper()[:1]
    name = {"A": a_name, "B": b_name}.get(w, "tie")
    return name, v.get("reason", "")


def run_prompt(pr, k, judge_model, timeout, with_bottom, prog, cfg=None):
    base = iterate.validate_selection(pr["problem"], "baseline", 0, cfg=cfg, progress=prog)
    top = iterate.validate_selection(pr["problem"], "top", k, cfg=cfg, progress=prog)
    rng = random.Random(hash(pr["id"]) & 0xFFFF)  # deterministic per-prompt order
    win, reason = _blind(pr["problem"], base["final"], top["final"], "baseline", "wrapper",
                         rng, judge_model, timeout)
    row = {
        "prompt": pr["id"], "cat": pr.get("cat"), "k": k,
        "winner": win, "reason": reason,
        "top_selected": top.get("selected"), "top_values": top.get("values"),
        "top_answered": sum(1 for t in top["tombstones"] if t["status"] == "ANSWERED"),
        "top_gaps": sum(1 for t in top["tombstones"] if t["status"] == "NOT_FOUND"),
        "baseline_len": len(base["final"]), "wrapper_len": len(top["final"]),
        "baseline_final": base["final"], "wrapper_final": top["final"],
    }
    if with_bottom:
        bot = iterate.validate_selection(pr["problem"], "bottom", k, progress=prog)
        rw, rr = _blind(pr["problem"], bot["final"], top["final"], "bottom", "top",
                        rng, judge_model, timeout)
        row["rank_winner"], row["rank_reason"] = rw, rr  # top vs bottom (ranking check)
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="*", help="prompt ids from testbank (default: a small agentic set).")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--judge-model", default="deepseek")
    p.add_argument("--timeout", type=int, default=200)
    p.add_argument("--with-bottom", action="store_true", help="also judge top-K vs bottom-K (ranking).")
    p.add_argument("--cwd", help="pin BOTH answerer and responder to this project dir (de-confounds).")
    p.add_argument("--responder-tools", default="", help="responder toolsets (e.g. 'file') for the test.")
    p.add_argument("--out")
    args = p.parse_args(argv)

    cfg = {"answer_cwd": args.cwd, "responder_cwd": args.cwd, "responder_toolsets": args.responder_tools}

    default_ids = ["research-compare", "add-auth", "gmail-triage"]  # researchable / spec / just-do-it
    ids = args.ids or default_ids
    prompts = [testbank.BY_ID[i] for i in ids if i in testbank.BY_ID]
    judge_model = pipeline.resolve_alias(args.judge_model)
    prog = lambda m: print(f"… {m}", file=sys.stderr, flush=True)

    rows = []
    if args.out and os.path.exists(args.out):  # append across per-prompt invocations
        try:
            rows = json.load(open(args.out)).get("rows", [])
        except Exception:
            rows = []
    done = {r["prompt"] for r in rows}

    t0 = time.time()
    for pr in prompts:
        if pr["id"] in done:
            print(f"… skip {pr['id']} (already in {args.out})", file=sys.stderr)
            continue
        print(f"… === {pr['id']} ({pr['cat']}) ===", file=sys.stderr, flush=True)
        try:
            row = run_prompt(pr, args.k, judge_model, args.timeout, args.with_bottom, prog, cfg)
        except Exception as e:
            row = {"prompt": pr["id"], "cat": pr.get("cat"), "error": str(e)}
        rows.append(row)
        print(f"  -> winner: {row.get('winner', row.get('error'))} "
              f"(answered={row.get('top_answered')}, gaps={row.get('top_gaps')})", file=sys.stderr)
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"rows": rows, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2, default=str)

    # summary
    real = [r for r in rows if "winner" in r]
    wins = sum(1 for r in real if r["winner"] == "wrapper")
    bl = sum(1 for r in real if r["winner"] == "baseline")
    tie = sum(1 for r in real if r["winner"] == "tie")
    print(f"\n=== wrapper vs baseline: wrapper {wins} · baseline {bl} · tie {tie}  (n={len(real)}) ===")
    for r in real:
        print(f"  {r['prompt']:<16} {r['cat']:<14} winner={r['winner']:<9} "
              f"answered={r.get('top_answered')} gaps={r.get('top_gaps')}  | {r.get('reason','')[:70]}")
        if "rank_winner" in r:
            print(f"      ranking (top vs bottom): {r['rank_winner']} | {r.get('rank_reason','')[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
