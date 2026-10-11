#!/usr/bin/env python3
"""keepalive_cli.py — Python CLI manager for Termux Gateway Keep-Alive.

Usage:
  keepalive_cli.py status [--json]
  keepalive_cli.py selfcheck [--json]
  keepalive_cli.py notify "message" [--quiet]
  keepalive_cli.py install [--target DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def get_keepalive_status() -> Dict[str, Any]:
    # Check gateway process
    gw_alive = False
    gw_pid = None
    try:
        res = subprocess.run(["pgrep", "-f", "hermes gateway"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            gw_alive = True
            gw_pid = int(res.stdout.strip().split()[0])
    except Exception:
        pass

    # Check watchdog
    run_flag = HERMES_HOME / "gateway_watchdog.run"
    pid_file = HERMES_HOME / "gateway_watchdog.pid"
    wd_alive = False
    wd_pid = None
    if run_flag.exists() and pid_file.exists():
        try:
            wd_pid = int(pid_file.read_text(encoding="utf-8").strip())
            # Check if process is still running
            os.kill(wd_pid, 0)
            wd_alive = True
        except Exception:
            wd_alive = False

    # Check state file
    state_file = HERMES_HOME / "gateway_state.json"
    state_info: Dict[str, Any] = {"exists": state_file.exists(), "connected": False, "age_seconds": None}
    if state_file.exists():
        try:
            mtime = state_file.stat().st_mtime
            state_info["age_seconds"] = int(time.time() - mtime)
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                tg_state = data.get("platforms", {}).get("telegram", {}).get("state")
                state_info["connected"] = tg_state == "connected"
                state_info["telegram_state"] = tg_state
        except Exception:
            pass

    return {
        "gateway": {"alive": gw_alive, "pid": gw_pid},
        "watchdog": {"running": wd_alive, "pid": wd_pid},
        "state_file": state_info,
        "wake_lock_available": shutil.which("termux-wake-lock") is not None,
    }


def run_selfcheck() -> Dict[str, Any]:
    st = get_keepalive_status()
    ok = True
    issues = []

    if not st["gateway"]["alive"]:
        ok = False
        issues.append("Gateway process is dead")

    if not st["state_file"]["exists"]:
        ok = False
        issues.append("gateway_state.json missing")
    else:
        if not st["state_file"].get("connected"):
            ok = False
            issues.append(f"Telegram state is {st['state_file'].get('telegram_state')}")
        if st["state_file"]["age_seconds"] and st["state_file"]["age_seconds"] > 900:
            ok = False
            issues.append(f"gateway_state.json is stale ({st['state_file']['age_seconds']}s old)")

    return {
        "healthy": ok,
        "issues": issues,
        "status": st,
    }


def install_scripts(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    scripts = [
        "gateway_watchdog.sh",
        "gateway_monitor.sh",
        "telegram_selfcheck.sh",
        "presence_notify.sh",
        "termux_presence.py",
    ]
    for name in scripts:
        src = SCRIPT_DIR / name
        dst = target_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            dst.chmod(0o755)
            print(f"Installed {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Termux Gateway Keep-Alive Manager.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_st = subparsers.add_parser("status", help="Check keepalive and gateway status.")
    p_st.add_argument("--json", action="store_true", help="Output as JSON.")

    # selfcheck
    p_sc = subparsers.add_parser("selfcheck", help="Run delivery and liveness self-check.")
    p_sc.add_argument("--json", action="store_true", help="Output as JSON.")

    # notify
    p_no = subparsers.add_parser("notify", help="Send ambient offline notification.")
    p_no.add_argument("message", type=str, help="Notification message.")
    p_no.add_argument("--quiet", action="store_true", help="Toast and notification only (no vibrate).")

    # install
    p_in = subparsers.add_parser("install", help="Install scripts into ~/.hermes/scripts/.")
    p_in.add_argument("--target", type=Path, default=HERMES_HOME / "scripts", help="Target directory.")

    args = parser.parse_args()

    if args.command == "status":
        st = get_keepalive_status()
        if args.json:
            print(json.dumps(st, indent=2))
        else:
            gw_str = f"ALIVE (pid {st['gateway']['pid']})" if st['gateway']['alive'] else "DEAD"
            wd_str = f"RUNNING (pid {st['watchdog']['pid']})" if st['watchdog']['running'] else "NOT RUNNING"
            print(f"Gateway:  {gw_str}")
            print(f"Watchdog: {wd_str}")
            print(f"WakeLock: {'Available' if st['wake_lock_available'] else 'Not found'}")

    elif args.command == "selfcheck":
        res = run_selfcheck()
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            if res["healthy"]:
                print("✓ Gateway and delivery self-check PASSED (All systems healthy).")
            else:
                print(f"✗ Gateway self-check FAILED: {', '.join(res['issues'])}")

    elif args.command == "notify":
        from termux_presence import fire
        ok = fire(args.message, quiet=args.quiet)
        print("Sent" if ok else "Failed to send notification")

    elif args.command == "install":
        install_scripts(args.target)


if __name__ == "__main__":
    main()
