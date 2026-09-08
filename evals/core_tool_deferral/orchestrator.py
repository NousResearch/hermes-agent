#!/usr/bin/env python3
"""Orchestrate the PR #97979 A/B battery. Resume-safe; per-run wall timeout.

Usage: orchestrator.py <model_slug> <reps> [--tasks id1,id2] [--arms base,pr] [--parallel N]
Results land in results/<model_short>/<arm>__<task>__rep<r>.json (override
the results root with ABDEFER_RESULTS).
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARNESS)
import tasks as taskmod

MODEL = sys.argv[1]
REPS = int(sys.argv[2])
task_ids = [t["id"] for t in taskmod.TASKS]
arms = ["base", "pr"]
parallel = 4
for a in sys.argv[3:]:
    if a.startswith("--tasks="):
        task_ids = a.split("=", 1)[1].split(",")
    elif a.startswith("--arms="):
        arms = a.split("=", 1)[1].split(",")
    elif a.startswith("--parallel="):
        parallel = int(a.split("=", 1)[1])

short = MODEL.split("/")[-1]
RESULTS = os.path.join(os.environ.get("ABDEFER_RESULTS", os.path.join(HARNESS, "results")), short)
os.makedirs(RESULTS, exist_ok=True)
PY = os.environ.get("ABDEFER_PYTHON", sys.executable)

cells = []
for task_id in task_ids:
    for arm in arms:
        for rep in range(1, REPS + 1):
            out = f"{RESULTS}/{arm}__{task_id}__rep{rep}.json"
            if os.path.exists(out):
                try:
                    with open(out, encoding="utf-8") as f:
                        rec = json.load(f)
                    if rec.get("error") is None or rec.get("score", 0) > 0:
                        continue  # keep good/attempted records
                    # errored record -> retry
                    os.remove(out)
                except Exception:
                    os.remove(out)
            cells.append((arm, task_id, rep, out))

print(f"model={MODEL} cells to run: {len(cells)} (parallel={parallel})", flush=True)

def run_cell(cell):
    arm, task_id, rep, out = cell
    timeout = taskmod.TASKS_BY_ID[task_id].get("timeout", 600)
    cmd = [PY, os.path.join(HARNESS, "worker.py"), arm, MODEL, task_id, str(rep), out]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60,
                           env=os.environ.copy())
        if p.returncode == 3:
            return (cell, "INFRA_ABORT", p.stderr[-500:])
        if p.returncode != 0 and not os.path.exists(out):
            rec = {"arm": arm, "model": MODEL, "task": task_id, "rep": rep,
                   "score": 0.0, "error": f"worker exit {p.returncode}",
                   "notes": [p.stderr[-400:]], "api_turns": None,
                   "total_tokens": None, "wall_s": round(time.time() - t0, 1),
                   "bridge_calls": None, "tool_calls_total": None,
                   "tool_counts": {}, "raw_xml_noise": False}
            with open(out, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1)
            return (cell, "WORKER_ERR", p.stderr[-300:])
        return (cell, "OK", p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "")
    except subprocess.TimeoutExpired:
        rec = {"arm": arm, "model": MODEL, "task": task_id, "rep": rep,
               "score": 0.0, "error": "wall timeout", "notes": ["hard wall timeout"],
               "api_turns": None, "total_tokens": None,
               "wall_s": round(time.time() - t0, 1), "bridge_calls": None,
               "tool_calls_total": None, "tool_counts": {}, "raw_xml_noise": False}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        return (cell, "TIMEOUT", "")

done = 0
infra_aborts = 0
with ThreadPoolExecutor(max_workers=parallel) as ex:
    futs = {ex.submit(run_cell, c): c for c in cells}
    for fut in as_completed(futs):
        cell, status, info = fut.result()
        done += 1
        print(f"[{done}/{len(cells)}] {cell[0]}/{cell[1]}/rep{cell[2]}: {status} {info}", flush=True)
        if status == "INFRA_ABORT":
            infra_aborts += 1
            if infra_aborts >= 3:
                print("FATAL: 3 infra aborts — stopping battery", flush=True)
                sys.exit(3)
print("BATTERY COMPLETE", flush=True)
