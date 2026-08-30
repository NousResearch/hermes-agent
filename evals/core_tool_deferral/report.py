#!/usr/bin/env python3
"""Aggregate A/B results. Usage: report.py [model_short ...]"""
import json
import glob
import os
import statistics
import sys

BASE = os.environ.get("ABDEFER_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
models = sys.argv[1:] or sorted(
    d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)) and d != "smoke")

def load(model):
    recs = []
    for p in glob.glob(f"{BASE}/{model}/*.json"):
        if p.endswith(".transcript.json"):
            continue
        with open(p) as f:
            recs.append(json.load(f))
    return recs

def fmt(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

for model in models:
    recs = load(model)
    if not recs:
        continue
    tasks = sorted({r["task"] for r in recs})
    print(f"\n{'='*100}\nMODEL: {model}   (runs: {len(recs)})\n{'='*100}")
    hdr = f"{'task':<28} | {'arm':<4} | {'n':>1} | {'score':>10} | {'turns':>6} | {'tok(k)':>7} | {'wall':>6} | {'bridge':>6} | {'err':>3}"
    print(hdr)
    print("-" * len(hdr))
    agg = {"base": {"s": [], "t": [], "k": [], "w": []}, "pr": {"s": [], "t": [], "k": [], "w": []}}
    for task in tasks:
        for arm in ("base", "pr"):
            rs = [r for r in recs if r["task"] == task and r["arm"] == arm]
            if not rs:
                continue
            scores = [r["score"] for r in rs]
            ok = [r for r in rs if not r.get("error")]
            turns = [r["api_turns"] for r in ok if r.get("api_turns")]
            toks = [r["total_tokens"] for r in ok if r.get("total_tokens")]
            walls = [r["wall_s"] for r in ok if r.get("wall_s")]
            bridges = [r.get("bridge_calls") or 0 for r in ok]
            nerr = sum(1 for r in rs if r.get("error"))
            smean = statistics.mean(scores)
            sspread = f"{smean:.2f} [{min(scores):.1f}-{max(scores):.1f}]"
            print(f"{task:<28} | {arm:<4} | {len(rs)} | {sspread:>10} | "
                  f"{fmt(statistics.mean(turns) if turns else None):>6} | "
                  f"{fmt(statistics.mean(toks)/1000 if toks else None):>7} | "
                  f"{fmt(statistics.mean(walls) if walls else None):>6} | "
                  f"{fmt(statistics.mean(bridges) if bridges else None):>6} | {nerr:>3}")
            agg[arm]["s"].append(smean)
            if turns: agg[arm]["t"].append(statistics.mean(turns))
            if toks: agg[arm]["k"].append(statistics.mean(toks))
            if walls: agg[arm]["w"].append(statistics.mean(walls))
    print("-" * len(hdr))
    for arm in ("base", "pr"):
        a = agg[arm]
        if a["s"]:
            print(f"{'MEAN-OF-TASK-MEANS':<28} | {arm:<4} |   | {statistics.mean(a['s']):>10.3f} | "
                  f"{fmt(statistics.mean(a['t']) if a['t'] else None):>6} | "
                  f"{fmt(statistics.mean(a['k'])/1000 if a['k'] else None):>7} | "
                  f"{fmt(statistics.mean(a['w']) if a['w'] else None):>6} |")
    noise = [r for r in recs if r.get("raw_xml_noise")]
    errs = [r for r in recs if r.get("error")]
    if noise:
        print(f"raw-XML noise runs: {len(noise)} -> " + ", ".join(f"{r['arm']}/{r['task']}/r{r['rep']}" for r in noise))
    if errs:
        print(f"errored runs: {len(errs)} -> " + ", ".join(f"{r['arm']}/{r['task']}/r{r['rep']}: {r['error'][:60]}" for r in errs))
