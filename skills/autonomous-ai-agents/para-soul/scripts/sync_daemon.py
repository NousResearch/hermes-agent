#!/usr/bin/env python3
"""Para-Soul Sync Daemon — Local-first health + optional cloud sync

Every 10 minutes:
  1. Local health check (mtime-based, writes health.json)      ← ALWAYS
  2. Auto-fix stale files (memsync, reflect, index)            ← ALWAYS
  3. If DID configured: push changed files, pull remote updates ← OPT-IN

Cloud is passive storage only. No server-driven health logic.
"""

import subprocess
import time
import os
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

SYNC_INTERVAL = 600               # 10 minutes
HEARTBEAT_INTERVAL = 12 * 3600    # 12 hours (cloud mode only)
LOG_FILE = None

ENV_DEFAULTS = {
    "PARA_HOME": os.path.expanduser("~/.para"),
    "PARA_KEYS_DIR": os.path.expanduser("~/.config/paragate/keys"),
    "PARAGATE_URL": "http://paragate.cc",
}

# ── Monitored files + thresholds ────────────────────

MONITORED_FILES = [
    "growth-log", "human-relationship.md", "memory.md",
    "skills.json", "mental-models.md", "keywords.json",
    "long-term-memory.md", "principles.md", "soul.md",
]

STALENESS_THRESHOLDS = {
    "growth-log": 24,
    "human-relationship.md": 24,
    "memory.md": 48,
    "skills.json": 120,
    "mental-models.md": 120,
    "keywords.json": 120,
    "long-term-memory.md": 120,
    "principles.md": 120,
    "soul.md": 120,
}

AUTO_FIX_FILES = {
    "memory.md": "memsync",
    "skills.json": "memsync",
    "mental-models.md": "reflect",
    "keywords.json": "index",
}

MARK_ONLY_FILES = {"growth-log", "human-relationship.md",
                    "long-term-memory.md", "principles.md", "soul.md"}


# ── Logging ──────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


# ── Local health check ───────────────────────────────

def _latest_growth_log(para_home):
    log_dir = Path(para_home) / "growth-log"
    if not log_dir.is_dir():
        return None
    md_files = sorted(
        [f for f in log_dir.iterdir() if f.suffix == ".md"],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    return md_files[0] if md_files else None


def _compute_file_hash(path):
    if not path or not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_health_check(para_home):
    """Check all monitored files' freshness. Write health.json."""
    now = datetime.now()
    report = {"checked_at": now.isoformat(), "files": {},
              "needs_write_cycle": False, "needs_reflect": False, "stale_files": []}

    for name in MONITORED_FILES:
        if name == "growth-log":
            path = _latest_growth_log(para_home)
        else:
            path = Path(para_home) / name

        max_h = STALENESS_THRESHOLDS.get(name, 120)

        if not path or not path.exists():
            report["files"][name] = {"exists": False, "stale_hours": None, "max_hours": max_h,
                                      "is_stale": True}
            report["stale_files"].append(name)
            report["needs_write_cycle"] = True
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        stale_h = (now - mtime).total_seconds() / 3600
        is_stale = stale_h > max_h

        report["files"][name] = {
            "exists": True,
            "last_modified": mtime.isoformat(),
            "stale_hours": round(stale_h, 1),
            "max_hours": max_h,
            "is_stale": is_stale,
        }
        if is_stale:
            report["stale_files"].append(name)

    report["needs_write_cycle"] = any(
        report["files"].get(f, {}).get("is_stale", True)
        for f in ["growth-log", "human-relationship.md"]
    )
    report["needs_reflect"] = report["files"].get("mental-models", {}).get("is_stale", True)
    report["stale_files"] = sorted(report["stale_files"])

    state_dir = Path(para_home) / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "health.json").write_text(json.dumps(report, indent=2))

    if report["stale_files"]:
        log(f"⚠️  HEALTH: {len(report['stale_files'])} stale ({', '.join(report['stale_files'])})")

    return report


# ── Auto-fix actions ─────────────────────────────────

def run_memsync(para_home, core_path):
    memsync_path = Path(core_path).resolve().parent / "scripts" / "memsync.py"
    if not memsync_path.exists():
        log("⚠️  memsync: script not found")
        return
    try:
        result = subprocess.run(
            ["python3", str(memsync_path)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PARA_HOME": str(para_home)}
        )
        if result.returncode == 0:
            log("✅ memsync OK")
        else:
            log(f"⚠️  memsync: {result.stderr.strip()[:100]}")
    except Exception as e:
        log(f"⚠️  memsync ERROR: {e}")


def run_reflect(core_path, para_home):
    try:
        result = subprocess.run(
            ["python3", str(core_path), "reflect", "--save"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PARA_HOME": str(para_home)}
        )
        if result.returncode == 0:
            log("✅ reflect OK")
    except Exception as e:
        log(f"⚠️  reflect ERROR: {e}")


def run_index(core_path, para_home):
    try:
        subprocess.run(
            ["python3", str(core_path), "index"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PARA_HOME": str(para_home)}
        )
        log("✅ index OK")
    except Exception as e:
        log(f"⚠️  index ERROR: {e}")


def execute_auto_fix(report, core_path, para_home):
    """Run auto-fix for files that can be auto-generated."""
    ran = set()
    for name in report.get("stale_files", []):
        action = AUTO_FIX_FILES.get(name)
        if not action or action in ran:
            continue
        ran.add(action)
        stale_h = report["files"].get(name, {}).get("stale_hours", 0)

        if action == "memsync":
            log(f"🔧 auto_memsync ({name}: stale {stale_h:.0f}h)")
            run_memsync(para_home, core_path)
        elif action == "reflect":
            log(f"🔧 auto_reflect ({name}: stale {stale_h:.0f}h)")
            run_reflect(core_path, para_home)
        elif action == "index":
            log(f"🔧 auto_index ({name}: stale {stale_h:.0f}h)")
            run_index(core_path, para_home)

    for name in report.get("stale_files", []):
        if name in MARK_ONLY_FILES:
            h = report["files"].get(name, {}).get("stale_hours", 0)
            log(f"⚠️  {name} stale {h:.0f}h — needs manual write (daemon cannot auto-generate)")


# ── Cloud sync (opt-in, DID-passthrough) ─────────────

def has_did(para_home):
    profile_path = Path(para_home) / "profile.json"
    if not profile_path.exists():
        return False
    try:
        profile = json.loads(profile_path.read_text())
        return bool(profile.get("identity", {}).get("did", ""))
    except Exception:
        return False


def cloud_sync(core_path, para_home):
    """Push changed files + pull remote updates. Passive storage only."""
    try:
        # Push
        result = subprocess.run(
            ["python3", str(core_path), "sync", "--full"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PARA_HOME": str(para_home)}
        )
        if result.returncode == 0 and "✅" in result.stdout:
            log("☁️  Push OK")
        else:
            log(f"⚠️  Push: {result.stderr.strip()[:100] or result.stdout.strip()[:100]}")

        # Pull
        result2 = subprocess.run(
            ["python3", str(core_path), "pull"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PARA_HOME": str(para_home)}
        )
        if result2.returncode == 0:
            out = result2.stdout.strip()
            if out and "❌" not in out:
                log(f"☁️  Pull: {out[:100]}")
    except Exception as e:
        log(f"⚠️  Cloud sync error: {e}")


# ── Main ────────────────────────────────────────────

def main():
    global LOG_FILE

    para_home = Path(os.environ.get("PARA_HOME", ENV_DEFAULTS["PARA_HOME"]))
    LOG_FILE = str(para_home / "sync" / "sync_daemon.log")

    script_dir = Path(__file__).resolve().parent
    core_path = script_dir.parent / "core.py"

    log("Daemon started (local-first, cloud opt-in)")
    log(f"Interval: {SYNC_INTERVAL//60}min | PARA_HOME: {para_home}")

    cloud_enabled = has_did(para_home)
    log(f"Cloud sync: {'✅ enabled' if cloud_enabled else '❌ disabled (no DID)'}")

    # First run
    report = local_health_check(para_home)
    execute_auto_fix(report, core_path, para_home)
    if cloud_enabled:
        cloud_sync(core_path, para_home)

    while True:
        time.sleep(SYNC_INTERVAL)

        report = local_health_check(para_home)
        execute_auto_fix(report, core_path, para_home)

        if cloud_enabled:
            cloud_sync(core_path, para_home)


if __name__ == "__main__":
    main()
