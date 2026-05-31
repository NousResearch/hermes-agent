#!/usr/bin/env python3
"""Para-Soul 定时同步守护进程

每 10 分钟：同步到 Paragate + 人格健康检查。
输出写入 .para/sync/sync_daemon.log
健康报告写入 .para/state/health.json

Usage:
  python3 sync_daemon.py       # Start (runs forever)
  
Env vars:
  PARA_HOME      — path to .para/ directory (default: ~/.para)
  PARA_KEYS_DIR  — path to private key (default: ~/.config/paragate/keys)
  PARAGATE_URL   — Paragate server URL (default: http://paragate.cc)
"""

import subprocess
import time
import os
import json
from datetime import datetime, timedelta

SYNC_INTERVAL = 600  # 10 分钟
LOG_FILE = None  # Set by main() from PARA_HOME

ENV_DEFAULTS = {
    "PARA_HOME": os.path.expanduser("~/.para"),
    "PARA_KEYS_DIR": os.path.expanduser("~/.config/paragate/keys"),
    "PARAGATE_URL": "http://paragate.cc",
}

CORE_PY = "core.py"  # Resolved relative to script dir or from PARA_HOME parent


def _latest_growth_log(para_home):
    """Return the most recently modified .md file in growth-log/, or the current month's file path if directory is empty."""
    log_dir = os.path.join(para_home, "growth-log")
    if not os.path.isdir(log_dir):
        return os.path.join(log_dir, datetime.now().strftime("%Y-%m") + ".md")
    md_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".md")]
    if not md_files:
        return os.path.join(log_dir, datetime.now().strftime("%Y-%m") + ".md")
    return max(md_files, key=os.path.getmtime)


# ── Health check thresholds ──
FILE_STALENESS_RULES = {
    "growth-log": {
        "check": lambda para_home: _latest_growth_log(para_home),
        "max_hours": 24,
        "description": "Daily growth log (most recent entry)"
    },
    "human-relationship": {
        "check": lambda para_home: os.path.join(para_home, "human-relationship.md"),
        "max_hours": 24,
        "description": "Session log + trust index"
    },
    "mental-models": {
        "check": lambda para_home: os.path.join(para_home, "mental-models.md"),
        "max_hours": 120,  # 5 days
        "description": "Reflect output for mental model updates"
    },
    "long-term-memory": {
        "check": lambda para_home: os.path.join(para_home, "long-term-memory.md"),
        "max_hours": 720,  # 30 days
        "description": "Archived milestones (low urgency)"
    },
}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def health_check(para_home):
    """Check personality file freshness and write health.json for all agents."""
    now = datetime.now()
    report = {
        "checked_at": now.isoformat(),
        "files": {},
        "needs_write_cycle": False,
        "needs_reflect": False,
        "stale_files": [],
    }

    for name, rule in FILE_STALENESS_RULES.items():
        path = rule["check"](para_home)
        max_hours = rule["max_hours"]

        if not os.path.exists(path):
            report["files"][name] = {
                "exists": False,
                "path": path,
                "stale_hours": None,
                "max_hours": max_hours,
                "description": rule["description"],
            }
            report["needs_write_cycle"] = True
            report["stale_files"].append(name)
            continue

        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        stale_hours = (now - mtime).total_seconds() / 3600
        is_stale = stale_hours > max_hours

        report["files"][name] = {
            "exists": True,
            "path": path,
            "last_modified": mtime.isoformat(),
            "stale_hours": round(stale_hours, 1),
            "max_hours": max_hours,
            "is_stale": is_stale,
            "description": rule["description"],
        }

        if is_stale:
            report["stale_files"].append(name)

    # Aggregate flags for agents to consume
    report["needs_write_cycle"] = any(
        report["files"].get(f, {}).get("is_stale", True)
        for f in ["growth-log", "human-relationship"]
    )
    report["needs_reflect"] = report["files"].get("mental-models", {}).get("is_stale", True)
    report["stale_files"] = sorted(report["stale_files"])

    # Write to state/health.json so any future agent can read it
    state_dir = os.path.join(para_home, "state")
    os.makedirs(state_dir, exist_ok=True)
    health_path = os.path.join(state_dir, "health.json")

    with open(health_path, "w") as f:
        json.dump(report, f, indent=2)

    if report["needs_write_cycle"]:
        log(f"⚠️  HEALTH: Write-cycle needed ({len(report['stale_files'])} files stale: {', '.join(report['stale_files'])})")


def sync_once(core_path):
    env = os.environ.copy()
    for k, v in ENV_DEFAULTS.items():
        if k not in env:
            env[k] = v
    try:
        result = subprocess.run(
            ["python3", core_path, "sync"],
            capture_output=True, text=True, timeout=30, env=env
        )
        if "✅" in result.stdout:
            log("✅ Sync OK")
        else:
            log(f"❌ Sync FAIL: {result.stderr.strip()[:150] or result.stdout.strip()[:150]}")
    except Exception as e:
        log(f"❌ Sync ERROR: {e}")


def main():
    global LOG_FILE
    
    para_home = os.environ.get("PARA_HOME", ENV_DEFAULTS["PARA_HOME"])
    LOG_FILE = os.path.join(para_home, "sync", "sync_daemon.log")
    
    # Find core.py — try script dir parent first, then PARA_HOME parent
    script_dir = os.path.dirname(os.path.abspath(__file__))
    core_path = os.path.join(os.path.dirname(script_dir), "core.py")
    if not os.path.exists(core_path):
        core_path = os.path.join(os.path.dirname(para_home), "core.py")
    
    log(f"Daemon started. Interval: {SYNC_INTERVAL}s ({SYNC_INTERVAL//60}min)")
    log(f"PARA_HOME: {para_home}")
    log(f"PARAGATE_URL: {os.environ.get('PARAGATE_URL', ENV_DEFAULTS['PARAGATE_URL'])}")

    # 首次立即 sync + health check
    health_check(para_home)
    sync_once(core_path)

    while True:
        time.sleep(SYNC_INTERVAL)
        health_check(para_home)
        sync_once(core_path)


if __name__ == "__main__":
    main()
