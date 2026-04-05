#!/usr/bin/env python3
"""Autoresearch run registry - multi-user tracking.
Usage:
    python registry.py register <rid> <user_id> <platform> <chat_id> <goal> <cron_job_id> [--watchdog-job-id ID]
    python registry.py list [--user-id U] [--platform P] [--active-only]
    python registry.py get <rid>
    python registry.py update <rid> [--phase P] [--cron-job-id ID]
    python registry.py remove <rid>
    python registry.py find-by-job <cron_job_id>
"""
import json, os, sys
from _util import now_iso, atomic_write, read_json, hermes_home


def _registry_path():
    return os.path.join(hermes_home(), "autoresearch", "registry.json")


def _read_registry():
    try:
        with open(_registry_path()) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"runs": {}}


def register(rid, uid, plat, cid, goal, jid, wid=None):
    reg = _read_registry()
    run_dir = os.path.join(hermes_home(), "autoresearch", rid)
    reg["runs"][rid] = {
        "research_id": rid, "user_id": uid, "platform": plat,
        "chat_id": cid, "goal": goal, "cron_job_id": jid,
        "watchdog_job_id": wid, "run_dir": run_dir,
        "created": now_iso(), "phase": "starting",
    }
    atomic_write(_registry_path(), reg)
    print(json.dumps({"status": "registered", "research_id": rid}))


def list_runs(uid=None, plat=None, active_only=False):
    runs = list(_read_registry().get("runs", {}).values())
    if uid:
        runs = [r for r in runs if r.get("user_id") == uid]
    if plat:
        runs = [r for r in runs if r.get("platform") == plat]
    for run in runs:
        try:
            with open(os.path.join(run["run_dir"], "status.json")) as f:
                status = json.load(f)
            run["phase"] = status.get("phase", "?")
            run["experiments_done"] = status.get("experiments_done", 0)
            run["experiments_total"] = status.get("experiments_total", 0)
        except Exception:
            pass
    if active_only:
        runs = [r for r in runs if r.get("phase") in ("planning", "executing", "starting")]
    print(json.dumps({"count": len(runs), "runs": runs}, indent=2))


def get_run(rid):
    run = _read_registry().get("runs", {}).get(rid)
    if not run:
        print(json.dumps({"error": f"Not found: {rid}"}))
        sys.exit(1)
    try:
        with open(os.path.join(run["run_dir"], "status.json")) as f:
            run["status"] = json.load(f)
    except Exception:
        run["status"] = None
    print(json.dumps(run, indent=2))


def update_run(rid, **kw):
    reg = _read_registry()
    if rid not in reg.get("runs", {}):
        print(json.dumps({"error": "Not found"}))
        sys.exit(1)
    for k, v in kw.items():
        if v is not None:
            reg["runs"][rid][k] = v
    atomic_write(_registry_path(), reg)
    print(json.dumps({"status": "updated"}))


def remove_run(rid):
    reg = _read_registry()
    if rid in reg.get("runs", {}):
        del reg["runs"][rid]
        atomic_write(_registry_path(), reg)
        print(json.dumps({"status": "removed"}))
    else:
        print(json.dumps({"error": "Not found"}))


def find_by_job(jid):
    for run in _read_registry().get("runs", {}).values():
        if run.get("cron_job_id") == jid or run.get("watchdog_job_id") == jid:
            print(json.dumps(run, indent=2))
            return
    print(json.dumps({"error": f"No run for job {jid}"}))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    cmd = args[0]
    if cmd == "register":
        wid = args[args.index("--watchdog-job-id") + 1] if "--watchdog-job-id" in args else None
        register(args[1], args[2], args[3], args[4], args[5], args[6], wid)
    elif cmd == "list":
        uid = args[args.index("--user-id") + 1] if "--user-id" in args else None
        plat = args[args.index("--platform") + 1] if "--platform" in args else None
        list_runs(uid, plat, "--active-only" in args)
    elif cmd == "get":
        get_run(args[1])
    elif cmd == "update":
        kw = {}
        i = 2
        while i < len(args):
            if args[i].startswith("--"):
                kw[args[i][2:].replace("-", "_")] = args[i + 1]
                i += 2
            else:
                i += 1
        update_run(args[1], **kw)
    elif cmd == "remove":
        remove_run(args[1])
    elif cmd == "find-by-job":
        find_by_job(args[1])
    else:
        print(f"Unknown: {cmd}")
        sys.exit(1)
