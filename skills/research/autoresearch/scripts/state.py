#!/usr/bin/env python3
"""Autoresearch state management - atomic JSON I/O, research ID generation, budget checks.

Usage:
    python state.py init <run_dir> <goal> <domain> <scope> <depth> <max_experiments>
    python state.py status <run_dir>
    python state.py update-status <run_dir> <phase> [--experiments-done N] [--merged N] [--reverted N]
    python state.py control <run_dir> [--action ACTION] [--addendum TEXT]
    python state.py read-control <run_dir>
    python state.py checkpoint <run_dir> <last_completed> <next>
    python state.py read-checkpoint <run_dir>
    python state.py check-budget <run_dir> [--tokens N]
    python state.py gen-id <domain>
"""
import json, os, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path
from _util import now_iso, atomic_write, read_json

def gen_id(domain):
    h = hashlib.sha256(f"{domain}{now_iso()}{os.getpid()}".encode()).hexdigest()[:6]
    return f"{domain}_{h}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

def init(run_dir, goal, domain, scope, depth, max_exp, max_duration=180, max_tokens=2000000):
    rd = Path(run_dir); (rd/"workspace").mkdir(parents=True, exist_ok=True)
    atomic_write(str(rd/"config.json"), {"goal":goal,"domain":domain,"scope":scope,"depth":depth,
        "max_experiments":max_exp,"max_experiments_hard_cap":int(max_exp*1.5),
        "max_duration_minutes":max_duration,"max_tokens":max_tokens,"created":now_iso()})
    atomic_write(str(rd/"status.json"), {"phase":"planning","experiments_done":0,"experiments_total":0,
        "experiments_merged":0,"experiments_reverted":0,"experiments_failed":0,
        "current_experiment":None,"last_updated":now_iso()})
    atomic_write(str(rd/"control.json"), {"action":"none"})
    atomic_write(str(rd/"plan.json"), {"experiments":[]})
    (rd/"results.log").touch()
    print(json.dumps({"status":"initialized","run_dir":str(rd),"workspace":str(rd/"workspace")}, indent=2))

def status(run_dir):
    d = read_json(os.path.join(run_dir, "status.json"))
    if not d: print(json.dumps({"error":"status.json not found"})); sys.exit(1)
    print(json.dumps(d, indent=2))

def update_status(run_dir, phase, **kw):
    p = os.path.join(run_dir, "status.json"); d = read_json(p)
    d["phase"] = phase; d["last_updated"] = now_iso()
    for k, v in kw.items():
        if v is not None:
            # Normalize --merged/--reverted to experiments_merged/experiments_reverted
            if k == "merged": k = "experiments_merged"
            elif k == "reverted": k = "experiments_reverted"
            d[k] = v
    atomic_write(p, d); print(json.dumps(d, indent=2))

def write_control(run_dir, action="none", addendum=None):
    d = {"action": action, "timestamp": now_iso()}
    if addendum: d["addendum"] = addendum
    atomic_write(os.path.join(run_dir, "control.json"), d); print(json.dumps(d, indent=2))

def read_control(run_dir):
    print(json.dumps(read_json(os.path.join(run_dir, "control.json")), indent=2))

def checkpoint(run_dir, last, nxt):
    d = {"last_completed": last, "next": nxt, "timestamp": now_iso()}
    atomic_write(os.path.join(run_dir, "checkpoint.json"), d); print(json.dumps(d, indent=2))

def read_checkpoint(run_dir):
    d = read_json(os.path.join(run_dir, "checkpoint.json"))
    print(json.dumps(d if d else {"error": "no checkpoint"}, indent=2))

def check_budget(run_dir, tokens=None):
    cfg = read_json(os.path.join(run_dir, "config.json"))
    st = read_json(os.path.join(run_dir, "status.json"))
    md, mt = cfg.get("max_duration_minutes",180), cfg.get("max_tokens",2000000)
    cap = cfg.get("max_experiments_hard_cap",30); done = st.get("experiments_done",0)
    # Auto-read token usage from usage.json when not explicitly provided
    if tokens is None:
        usage = read_json(os.path.join(run_dir, "usage.json"))
        tokens = usage.get("total_tokens", 0)
    v = []
    if md and cfg.get("created"):
        try:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(cfg["created"])).total_seconds()/60
            if elapsed > md: v.append(f"time_exceeded: {elapsed:.0f}min > {md}min")
        except Exception: pass
    if mt and tokens > mt: v.append(f"tokens_exceeded: {tokens} > {mt}")
    if done >= cap: v.append(f"experiments_exceeded: {done} >= {cap}")
    print(json.dumps({"exceeded":len(v)>0,"violations":v,
        "budget":{"max_duration_minutes":md,"max_tokens":mt,"max_experiments_hard_cap":cap},
        "current":{"experiments_done":done,"tokens_used":tokens}}, indent=2))

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(1)
    c = a[0]
    if c == "gen-id": print(gen_id(a[1] if len(a)>1 else "general"))
    elif c == "init":
        mx_dur,mx_tok=180,2000000; i=7
        while i<len(a):
            if a[i]=="--max-duration": mx_dur=int(a[i+1]); i+=2
            elif a[i]=="--max-tokens": mx_tok=int(a[i+1]); i+=2
            else: i+=1
        init(a[1],a[2],a[3],a[4],a[5],int(a[6]),mx_dur,mx_tok)
    elif c == "status": status(a[1])
    elif c == "update-status":
        kw={}; i=3
        while i<len(a):
            if a[i].startswith("--"): k=a[i][2:].replace("-","_"); v=a[i+1]
            else: i+=1; continue
            try: v=int(v)
            except ValueError: pass
            kw[k]=v; i+=2
        update_status(a[1],a[2],**kw)
    elif c == "control":
        act,add="none",None; i=2
        while i<len(a):
            if a[i]=="--action": act=a[i+1]; i+=2
            elif a[i]=="--addendum": add=a[i+1]; i+=2
            else: i+=1
        write_control(a[1],act,add)
    elif c == "read-control": read_control(a[1])
    elif c == "checkpoint": checkpoint(a[1],int(a[2]),int(a[3]))
    elif c == "read-checkpoint": read_checkpoint(a[1])
    elif c == "check-budget":
        t = int(a[a.index("--tokens")+1]) if "--tokens" in a else None
        check_budget(a[1],t)
    else: print(f"Unknown: {c}"); sys.exit(1)
