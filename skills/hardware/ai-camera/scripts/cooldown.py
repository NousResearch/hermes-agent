#!/usr/bin/env python3
"""
Alert Cooldown — Prevent repeated alerts for the same event type.

Simple file-based cooldown system. Tracks when each alert type last fired.

Usage:
    python3 cooldown.py check alert_name --minutes 10
    # Exit 0 = can alert (cooldown expired or never set)
    # Exit 1 = still in cooldown (skip alert)

    python3 cooldown.py set alert_name
    # Mark alert_name as just fired (starts cooldown timer)

    python3 cooldown.py status alert_name
    # Show remaining cooldown time

Storage:
    ~/.hermes/camera_cooldowns.json
"""

import sys
import json
import argparse
import os
import time
from pathlib import Path


COOLDOWN_FILE = os.path.expanduser("~/.hermes/camera_cooldowns.json")


def _load_cooldowns() -> dict:
    """Load cooldown state from disk."""
    if os.path.exists(COOLDOWN_FILE):
        try:
            with open(COOLDOWN_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_cooldowns(cooldowns: dict):
    """Save cooldown state to disk."""
    os.makedirs(os.path.dirname(COOLDOWN_FILE), exist_ok=True)
    with open(COOLDOWN_FILE, "w") as f:
        json.dump(cooldowns, f, indent=2)


def check(name: str, minutes: int) -> bool:
    """
    Check if we can fire an alert for the given name.

    Returns True if cooldown has expired (or never set), False if still in cooldown.
    """
    cooldowns = _load_cooldowns()
    last_fired = cooldowns.get(name, 0)
    elapsed = time.time() - last_fired
    return elapsed >= (minutes * 60)


def set_cooldown(name: str):
    """Mark an alert as just fired (starts cooldown timer)."""
    cooldowns = _load_cooldowns()
    cooldowns[name] = time.time()
    _save_cooldowns(cooldowns)


def status(name: str) -> dict:
    """Get cooldown status for an alert name."""
    cooldowns = _load_cooldowns()
    last_fired = cooldowns.get(name, 0)

    if last_fired == 0:
        return {"name": name, "status": "never_fired", "remaining_seconds": 0}

    elapsed = time.time() - last_fired
    return {
        "name": name,
        "last_fired": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_fired)),
        "elapsed_seconds": round(elapsed, 1),
    }


def clear(name: str = None):
    """Clear cooldown for a specific name, or all if name is None."""
    if name is None:
        _save_cooldowns({})
    else:
        cooldowns = _load_cooldowns()
        cooldowns.pop(name, None)
        _save_cooldowns(cooldowns)


def main():
    parser = argparse.ArgumentParser(description="Alert cooldown manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check
    check_parser = subparsers.add_parser("check", help="Check if we can fire an alert")
    check_parser.add_argument("name", help="Alert name (e.g., 'front_door_motion')")
    check_parser.add_argument("--minutes", type=int, default=10, help="Cooldown duration in minutes")

    # set
    set_parser = subparsers.add_parser("set", help="Mark alert as just fired")
    set_parser.add_argument("name", help="Alert name")

    # status
    status_parser = subparsers.add_parser("status", help="Show cooldown status")
    status_parser.add_argument("name", help="Alert name")

    # clear
    clear_parser = subparsers.add_parser("clear", help="Clear cooldown(s)")
    clear_parser.add_argument("name", nargs="?", default=None, help="Alert name (omit to clear all)")

    args = parser.parse_args()

    if args.command == "check":
        can_alert = check(args.name, args.minutes)
        if can_alert:
            print(json.dumps({"name": args.name, "can_alert": True}))
            sys.exit(0)
        else:
            print(json.dumps({"name": args.name, "can_alert": False, "reason": "still in cooldown"}))
            sys.exit(1)

    elif args.command == "set":
        set_cooldown(args.name)
        print(json.dumps({"name": args.name, "action": "cooldown_set"}))

    elif args.command == "status":
        info = status(args.name)
        print(json.dumps(info, indent=2))

    elif args.command == "clear":
        clear(args.name)
        target = args.name if args.name else "all"
        print(json.dumps({"action": "cleared", "target": target}))


if __name__ == "__main__":
    main()
