#!/usr/bin/env python3
"""Autoresearch token usage and cost tracking.
Usage:
    python usage.py session-cost <session_id>
    python usage.py job-cost <cron_job_id>
    python usage.py research-cost <run_dir>
    python usage.py track <run_dir> <exp_id> <input_tokens> <output_tokens> [--cost USD]
    python usage.py summary <run_dir>
"""
import json, os, sys, sqlite3
from _util import now_iso, atomic_write, read_json, hermes_home


def _sessions_db():
    return os.path.join(hermes_home(), "sessions.db")


def _registry_path():
    return os.path.join(hermes_home(), "autoresearch", "registry.json")


def session_cost(sid):
    db = _sessions_db()
    if not os.path.exists(db):
        print(json.dumps({"error": "No SessionDB"}))
        return
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    r = conn.execute(
        "SELECT id,model,input_tokens,output_tokens,cache_read_tokens,"
        "cache_write_tokens,reasoning_tokens,estimated_cost_usd "
        "FROM sessions WHERE id=?", (sid,)
    ).fetchone()
    conn.close()
    if not r:
        print(json.dumps({"error": f"Session {sid} not found"}))
        return
    print(json.dumps({
        "session_id": r["id"], "model": r["model"],
        "input_tokens": r["input_tokens"] or 0,
        "output_tokens": r["output_tokens"] or 0,
        "total_tokens": (r["input_tokens"] or 0) + (r["output_tokens"] or 0),
        "estimated_cost_usd": r["estimated_cost_usd"],
    }, indent=2))


def job_cost(jid):
    db = _sessions_db()
    if not os.path.exists(db):
        print(json.dumps({"error": "No SessionDB"}))
        return
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT input_tokens,output_tokens,estimated_cost_usd "
        "FROM sessions WHERE id LIKE ? ORDER BY started_at",
        (f"cron_{jid}_%",)
    ).fetchall()
    conn.close()
    if not rows:
        print(json.dumps({"error": f"No sessions for job {jid}"}))
        return
    ti = sum(r["input_tokens"] or 0 for r in rows)
    to = sum(r["output_tokens"] or 0 for r in rows)
    tc = sum(r["estimated_cost_usd"] or 0 for r in rows)
    print(json.dumps({
        "cron_job_id": jid, "sessions": len(rows),
        "input_tokens": ti, "output_tokens": to,
        "total_tokens": ti + to,
        "estimated_cost_usd": round(tc, 4) if tc else None,
    }, indent=2))


def track(run_dir, exp_id, inp, out, cost=None):
    up = os.path.join(run_dir, "usage.json")
    u = read_json(up)
    if "experiments" not in u:
        u = {"total_input_tokens": 0, "total_output_tokens": 0,
             "total_tokens": 0, "estimated_cost_usd": 0.0, "experiments": {}}
    u["experiments"][str(exp_id)] = {
        "input_tokens": inp, "output_tokens": out,
        "cost_usd": cost, "timestamp": now_iso(),
    }
    u["total_input_tokens"] = sum(e["input_tokens"] for e in u["experiments"].values())
    u["total_output_tokens"] = sum(e["output_tokens"] for e in u["experiments"].values())
    u["total_tokens"] = u["total_input_tokens"] + u["total_output_tokens"]
    costs = [e["cost_usd"] for e in u["experiments"].values() if e["cost_usd"] is not None]
    u["estimated_cost_usd"] = round(sum(costs), 4) if costs else 0.0
    atomic_write(up, u)
    print(json.dumps({
        "status": "tracked",
        "cumulative_tokens": u["total_tokens"],
        "estimated_cost_usd": u["estimated_cost_usd"],
    }))


def usage_summary(run_dir):
    rid = os.path.basename(run_dir.rstrip("/"))
    reg = read_json(_registry_path())
    run = reg.get("runs", {}).get(rid, {})
    result = {"research_id": rid, "source": "local"}
    u = read_json(os.path.join(run_dir, "usage.json"))
    if u:
        result["local_tracking"] = {
            "total_tokens": u.get("total_tokens", 0),
            "estimated_cost_usd": u.get("estimated_cost_usd"),
        }
    db = _sessions_db()
    cid = run.get("cron_job_id")
    if cid and os.path.exists(db):
        try:
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT input_tokens,output_tokens,estimated_cost_usd "
                "FROM sessions WHERE id LIKE ?", (f"cron_{cid}_%",)
            ).fetchall()
            conn.close()
            if rows:
                result["source"] = "session_db"
                result["session_db_tracking"] = {
                    "total_tokens": sum((r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in rows),
                    "estimated_cost_usd": round(sum(r["estimated_cost_usd"] or 0 for r in rows), 4),
                }
        except Exception:
            pass
    wid = run.get("watchdog_job_id")
    if wid and os.path.exists(db):
        try:
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT input_tokens,output_tokens,estimated_cost_usd "
                "FROM sessions WHERE id LIKE ?", (f"cron_{wid}_%",)
            ).fetchall()
            conn.close()
            if rows:
                result["watchdog_cost"] = {
                    "total_tokens": sum((r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in rows),
                    "estimated_cost_usd": round(sum(r["estimated_cost_usd"] or 0 for r in rows), 4),
                }
        except Exception:
            pass
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(1)
    c = a[0]
    if c == "session-cost": session_cost(a[1])
    elif c == "job-cost": job_cost(a[1])
    elif c == "research-cost": usage_summary(a[1])
    elif c == "track":
        cost = float(a[a.index("--cost") + 1]) if "--cost" in a else None
        track(a[1], int(a[2]), int(a[3]), int(a[4]), cost)
    elif c == "summary": usage_summary(a[1])
    else: print(f"Unknown: {c}"); sys.exit(1)
