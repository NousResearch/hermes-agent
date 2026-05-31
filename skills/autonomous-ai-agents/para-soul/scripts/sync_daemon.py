#!/usr/bin/env python3
"""Para-Soul Sync Daemon — Server-driven health + 10-min sync

Every 10 minutes:
  1. Sync file hashes to Paragate (core.py sync)
  2. Read action_items from server response
  3. Execute auto-fix actions (memsync, reflect, index)
  4. Heartbeat sync every 12h even if no file changes

No local health check — server is the single source of truth.
"""

import subprocess
import time
import os
import json
from datetime import datetime
from pathlib import Path

SYNC_INTERVAL = 600       # 10 minutes
HEARTBEAT_INTERVAL = 12 * 3600  # 12 hours

LOG_FILE = None

ENV_DEFAULTS = {
    "PARA_HOME": os.path.expanduser("~/.para"),
    "PARA_KEYS_DIR": os.path.expanduser("~/.config/paragate/keys"),
    "PARAGATE_URL": "http://paragate.cc",
}

# ── Logging ──────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")

# ── Auto-fix actions ─────────────────────────────────

def run_memsync(para_home, core_path):
    """Run memsync to update memory.md + skills.json."""
    memsync_paths = [
        Path(core_path).resolve().parent / "scripts" / "memsync.py",
        para_home / ".." / "scripts" / "memsync.py",
    ]
    for mp in memsync_paths:
        if mp.exists():
            try:
                result = subprocess.run(
                    ["python3", str(mp)],
                    capture_output=True, text=True, timeout=60,
                    env={**os.environ, "PARA_HOME": str(para_home)}
                )
                if result.returncode == 0:
                    log("✅ memsync OK")
                else:
                    log(f"⚠️  memsync: {result.stderr.strip()[:100]}")
                return True
            except Exception as e:
                log(f"⚠️  memsync ERROR: {e}")
    log("⚠️  memsync: script not found")
    return False


def run_reflect(core_path, para_home):
    """Run reflect --save to update mental-models.md."""
    try:
        result = subprocess.run(
            ["python3", str(core_path), "reflect", "--save"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PARA_HOME": str(para_home)}
        )
        if "✅" in result.stdout:
            log("✅ reflect OK")
        else:
            log(f"⚠️  reflect: {result.stdout.strip()[:100]}")
    except Exception as e:
        log(f"⚠️  reflect ERROR: {e}")


def run_index(core_path, para_home):
    """Run index to update keywords.json."""
    try:
        result = subprocess.run(
            ["python3", str(core_path), "index"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PARA_HOME": str(para_home)}
        )
        if "✅" in result.stdout:
            log("✅ index OK")
    except Exception as e:
        log(f"⚠️  index ERROR: {e}")


def execute_actions(action_items, core_path, para_home):
    """Execute auto-fix actions returned by server."""
    if not action_items:
        return

    for item in action_items:
        file = item.get("file", "")
        action = item.get("action", "")
        stale_h = item.get("stale_hours", 0)

        if action == "auto_memsync":
            log(f"🔧 auto_memsync triggered ({file}: stale {stale_h:.0f}h)")
            run_memsync(para_home, core_path)

        elif action == "auto_reflect":
            log(f"🔧 auto_reflect triggered ({file}: stale {stale_h:.0f}h)")
            run_reflect(core_path, para_home)

        elif action == "auto_index":
            log(f"🔧 auto_index triggered ({file}: stale {stale_h:.0f}h)")
            run_index(core_path, para_home)

        elif action == "block_write":
            # Can't block daemon — just log urgently
            log(f"🚨 BLOCK_WRITE needed: {file} stale {stale_h:.0f}h")
            log(f"   This file requires manual agent action. Daemon cannot auto-generate content.")

        elif action == "mark_stale":
            log(f"⚠️  STALE: {file} — {stale_h:.0f}h. Auto-fix not available (mark only).")

        else:
            log(f"⚠️  Unknown action '{action}' for {file}")


# ── Sync ────────────────────────────────────────────

def sync_once(core_path):
    """Run core.py sync, return action_items."""
    env = os.environ.copy()
    for k, v in ENV_DEFAULTS.items():
        if k not in env:
            env[k] = v

    try:
        result = subprocess.run(
            ["python3", str(core_path), "sync"],
            capture_output=True, text=True, timeout=30, env=env
        )
        stdout = result.stdout
        if "✅" in stdout:
            # Parse action items from output
            action_items = []
            in_action = False
            for line in stdout.split("\n"):
                if "Health actions needed" in line:
                    in_action = True
                    continue
                if in_action and line.strip().startswith("["):
                    # Parse: [ACTION] file — stale Nh — description
                    try:
                        parts = line.strip().split("—")
                        action_part = parts[0].strip()
                        stale_part = parts[1].strip() if len(parts) > 1 else ""
                        # Parse action and file from "[ACTION] file"
                        bracket_end = action_part.index("]")
                        action = action_part[1:bracket_end].strip().lower()
                        file_part = action_part[bracket_end+1:].strip()
                        # Parse stale hours
                        stale_hours = 0
                        if "stale" in stale_part:
                            stale_str = stale_part.split("stale")[1].split("h")[0].strip()
                            try:
                                stale_hours = float(stale_str)
                            except ValueError:
                                pass
                        action_items.append({
                            "file": file_part,
                            "action": action,
                            "stale_hours": stale_hours,
                        })
                    except Exception:
                        pass
                elif in_action and not line.strip():
                    in_action = False

            log("✅ Sync OK")
            return action_items
        else:
            log(f"❌ Sync FAIL: {result.stderr.strip()[:150] or stdout.strip()[:150]}")
            return []
    except Exception as e:
        log(f"❌ Sync ERROR: {e}")
        return []


# ── Main ────────────────────────────────────────────

def main():
    global LOG_FILE

    para_home = Path(os.environ.get("PARA_HOME", ENV_DEFAULTS["PARA_HOME"]))
    LOG_FILE = str(para_home / "sync" / "sync_daemon.log")

    # Find core.py
    script_dir = Path(__file__).resolve().parent
    core_path = script_dir.parent / "core.py"
    if not core_path.exists():
        core_path = para_home / ".." / "core.py"
    if not core_path.exists():
        log("❌ core.py not found. Daemon cannot start.")
        return

    # Track heartbeat
    heartbeat_path = para_home / "state" / "last_heartbeat.json"
    para_home.mkdir(parents=True, exist_ok=True)
    (para_home / "state").mkdir(exist_ok=True)
    (para_home / "sync").mkdir(exist_ok=True)

    # Read last heartbeat
    last_heartbeat = 0
    if heartbeat_path.exists():
        try:
            last_heartbeat = json.loads(heartbeat_path.read_text()).get("ts", 0)
        except Exception:
            pass

    log(f"Daemon started. Interval: {SYNC_INTERVAL}s ({SYNC_INTERVAL//60}min)")
    log(f"Heartbeat: every {HEARTBEAT_INTERVAL//3600}h")
    log(f"PARA_HOME: {para_home}")
    log(f"PARAGATE_URL: {os.environ.get('PARAGATE_URL', ENV_DEFAULTS['PARAGATE_URL'])}")

    # First sync immediately
    action_items = sync_once(core_path)
    execute_actions(action_items, core_path, para_home)

    consecutive_failures = 0

    while True:
        time.sleep(SYNC_INTERVAL)

        now = time.time()
        should_heartbeat = (now - last_heartbeat) >= HEARTBEAT_INTERVAL

        # Run sync
        action_items = sync_once(core_path)

        if action_items:
            consecutive_failures = 0
            execute_actions(action_items, core_path, para_home)

            # If auto-fixes were applied, sync again to push changes
            if any(a["action"] in ["auto_memsync", "auto_reflect", "auto_index"] for a in action_items):
                time.sleep(2)
                sync_once(core_path)
        else:
            consecutive_failures += 1

        # Update heartbeat
        last_heartbeat = now
        heartbeat_path.write_text(json.dumps({"ts": now}))

        if consecutive_failures >= 3:
            log(f"⚠️  {consecutive_failures} consecutive sync failures")


if __name__ == "__main__":
    main()
