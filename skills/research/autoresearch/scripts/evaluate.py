#!/usr/bin/env python3
"""Autoresearch experiment evaluation - scoring rubric and decision logic.

Usage:
    python evaluate.py score <evidence> <accuracy> <depth> <relevance> <net_improvement>
    python evaluate.py log-result <run_dir> <exp_id> <description> <type> <target> <decision> <reason> [--scores E,A,D,R,N]
    python evaluate.py log-result-ml <run_dir> <exp_id> <description> <metric_name> <metric_value> <prev_best> <decision> <reason>
    python evaluate.py read-results <run_dir> [--last N]
    python evaluate.py stats <run_dir>
"""
import json, os, sys
from _util import now_iso

def score_knowledge(evidence, accuracy, depth, relevance, net_improvement):
    total = evidence + accuracy + depth + relevance + net_improvement
    scores = {"evidence": evidence, "accuracy": accuracy, "depth": depth,
              "relevance": relevance, "net_improvement": net_improvement, "total": total, "max": 25}
    if evidence == 1: return {"scores": scores, "decision": "REVERT", "reason": "Evidence=1: no unsourced claims"}
    if net_improvement == 1: return {"scores": scores, "decision": "REVERT", "reason": "NetImprovement=1: made doc worse"}
    if total >= 18: return {"scores": scores, "decision": "MERGE", "reason": f"Strong (total={total}/25)"}
    elif total >= 13:
        if evidence >= 3 and relevance >= 3:
            return {"scores": scores, "decision": "MERGE", "reason": f"Acceptable (total={total}/25)"}
        return {"scores": scores, "decision": "REVERT", "reason": "Borderline weak evidence/relevance"}
    return {"scores": scores, "decision": "REVERT", "reason": f"Below threshold (total={total}/25)"}

def score_ml(metric_value, prev_best, lower_is_better=True):
    improved = metric_value < prev_best if lower_is_better else metric_value > prev_best
    delta = (prev_best - metric_value) if lower_is_better else (metric_value - prev_best)
    decision = "MERGE" if improved else "REVERT"
    return {"decision": decision, "metric_value": metric_value, "prev_best": prev_best,
            "delta": delta, "reason": f"Metric {'improved' if improved else 'not improved'} by {delta:.6f}"}

def log_result(run_dir, exp_id, description, exp_type, target, decision, reason, scores=None):
    entry = f"\n## Experiment {exp_id}: {description}\nTime: {now_iso()}\nType: {exp_type}\nTarget: {target}\n"
    if scores: entry += f"Scores: {scores}\n"
    entry += f"Decision: {decision}\nReason: {reason}\n---\n"
    with open(os.path.join(run_dir, "results.log"), "a") as f: f.write(entry)
    print(json.dumps({"status": "logged", "experiment_id": exp_id, "decision": decision}))

def log_result_ml(run_dir, exp_id, description, metric_name, metric_value, prev_best, decision, reason):
    entry = f"\n## Experiment {exp_id}: {description}\nTime: {now_iso()}\nMetric: {metric_name}={metric_value} (previous best: {prev_best})\nDecision: {decision}\nReason: {reason}\n---\n"
    with open(os.path.join(run_dir, "results.log"), "a") as f: f.write(entry)
    print(json.dumps({"status": "logged", "experiment_id": exp_id, "decision": decision}))

def read_results(run_dir, last_n=None):
    try:
        with open(os.path.join(run_dir, "results.log")) as f: content = f.read()
    except FileNotFoundError: print("No results yet."); return
    if last_n:
        entries = [e.strip() for e in content.split("---\n") if e.strip()]
        print("\n---\n".join(entries[-last_n:]))
    else: print(content)

def stats(run_dir):
    try:
        with open(os.path.join(run_dir, "results.log")) as f: content = f.read()
    except FileNotFoundError: print(json.dumps({"total": 0})); return
    entries = [e.strip() for e in content.split("---\n") if e.strip()]
    merged = sum(1 for e in entries if "Decision: MERGE" in e)
    reverted = sum(1 for e in entries if "Decision: REVERT" in e)
    types = {}
    for e in entries:
        for line in e.split("\n"):
            if line.startswith("Type: "): t = line[6:].strip(); types[t] = types.get(t, 0) + 1
    print(json.dumps({"total": len(entries), "merged": merged, "reverted": reverted,
                       "merge_rate": f"{merged/len(entries)*100:.0f}%" if entries else "0%", "by_type": types}, indent=2))

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args: print(__doc__); sys.exit(1)
    cmd = args[0]
    if cmd == "score": print(json.dumps(score_knowledge(int(args[1]),int(args[2]),int(args[3]),int(args[4]),int(args[5])), indent=2))
    elif cmd == "log-result":
        scores_str = None
        if "--scores" in args: idx = args.index("--scores"); scores_str = args[idx+1]; args = args[:idx]+args[idx+2:]
        log_result(args[1], int(args[2]), args[3], args[4], args[5], args[6], args[7], scores_str)
    elif cmd == "log-result-ml": log_result_ml(args[1], int(args[2]), args[3], args[4], float(args[5]), float(args[6]), args[7], args[8])
    elif cmd == "read-results":
        last_n = int(args[args.index("--last")+1]) if "--last" in args else None
        read_results(args[1], last_n)
    elif cmd == "stats": stats(args[1])
    else: print(f"Unknown: {cmd}"); sys.exit(1)
