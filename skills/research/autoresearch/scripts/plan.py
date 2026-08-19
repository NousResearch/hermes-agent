#!/usr/bin/env python3
"""Autoresearch experiment planning and tracking.

Usage:
    python plan.py write <run_dir> <experiments_json>
    python plan.py read <run_dir>
    python plan.py update-experiment <run_dir> <exp_id> <status> [--reason TEXT]
    python plan.py add-experiment <run_dir> <type> <hypothesis> <target_section>
    python plan.py next-pending <run_dir>
    python plan.py summary <run_dir>

Experiment types: investigate, deepen, verify, synthesize
Statuses: pending, in_progress, merged, reverted, failed
"""
import json, os, sys
from _util import now_iso, atomic_write, read_json

def _plan_path(run_dir): return os.path.join(run_dir, "plan.json")

def write_plan(run_dir, experiments_json):
    experiments = json.loads(experiments_json)
    valid_types = {"investigate", "deepen", "verify", "synthesize"}
    for exp in experiments:
        if "id" not in exp: raise ValueError(f"Experiment missing 'id': {exp}")
        if exp.get("type", "investigate") not in valid_types:
            raise ValueError(f"Invalid type '{exp.get('type')}'")
        exp.setdefault("status", "pending"); exp.setdefault("type", "investigate")
        exp.setdefault("target_section", ""); exp.setdefault("hypothesis", "")
    atomic_write(_plan_path(run_dir), {"experiments": experiments, "created": now_iso(), "last_updated": now_iso()})
    print(json.dumps({"status": "plan_written", "count": len(experiments)}))

def read_plan(run_dir):
    print(json.dumps(read_json(_plan_path(run_dir)), indent=2))

def update_experiment(run_dir, exp_id, status, reason=None):
    valid = {"pending", "in_progress", "merged", "reverted", "failed"}
    if status not in valid: print(json.dumps({"error": "Invalid status"})); sys.exit(1)
    data = read_json(_plan_path(run_dir)); found = False
    for exp in data.get("experiments", []):
        if exp["id"] == exp_id:
            exp["status"] = status; exp["updated"] = now_iso()
            if reason: exp["reason"] = reason
            found = True; break
    if not found: print(json.dumps({"error": f"Experiment {exp_id} not found"})); sys.exit(1)
    data["last_updated"] = now_iso(); atomic_write(_plan_path(run_dir), data)
    print(json.dumps({"status": "updated", "experiment_id": exp_id, "new_status": status}))

def add_experiment(run_dir, exp_type, hypothesis, target_section):
    data = read_json(_plan_path(run_dir)); experiments = data.get("experiments", [])
    max_id = max((e["id"] for e in experiments), default=0)
    new_exp = {"id": max_id+1, "type": exp_type, "hypothesis": hypothesis,
               "target_section": target_section, "status": "pending", "added_during_run": True}
    experiments.append(new_exp); data["experiments"] = experiments; data["last_updated"] = now_iso()
    atomic_write(_plan_path(run_dir), data)
    print(json.dumps({"status": "added", "experiment": new_exp}))

def next_pending(run_dir):
    for exp in read_json(_plan_path(run_dir)).get("experiments", []):
        if exp["status"] == "pending": print(json.dumps(exp)); return
    print(json.dumps({"status": "all_done"}))

def summary(run_dir):
    exps = read_json(_plan_path(run_dir)).get("experiments", [])
    counts = {}; types = {}
    for e in exps:
        s = e.get("status", "unknown"); counts[s] = counts.get(s, 0) + 1
        t = e.get("type", "unknown"); types[t] = types.get(t, 0) + 1
    print(json.dumps({"total": len(exps), "by_status": counts, "by_type": types}, indent=2))

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args: print(__doc__); sys.exit(1)
    cmd = args[0]
    if cmd == "write": write_plan(args[1], args[2])
    elif cmd == "read": read_plan(args[1])
    elif cmd == "update-experiment":
        reason = args[args.index("--reason")+1] if "--reason" in args else None
        update_experiment(args[1], int(args[2]), args[3], reason)
    elif cmd == "add-experiment": add_experiment(args[1], args[2], args[3], args[4] if len(args)>4 else "")
    elif cmd == "next-pending": next_pending(args[1])
    elif cmd == "summary": summary(args[1])
    else: print(f"Unknown: {cmd}"); sys.exit(1)
