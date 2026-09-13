#!/usr/bin/env python3
"""hermes_presence.py — ambient-presence notification channel for Termux/Android.

Provides local alerts without requiring network connectivity:
  vibrate (haptic attention) + toast (on-screen popup) + notification (shade).

Usage:
  python3 hermes_presence.py "message"            # vibrate + toast + notification
  python3 hermes_presence.py --quiet "message"    # toast + notification only
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
LOG = os.path.join(HOME, ".hermes", "logs", "presence.log")
TITLE = "Hermes Agent"


def _run(cmd: list[str], timeout: int = 12) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def fire(msg: str, quiet: bool = False) -> bool:
    results: dict[str, bool] = {}
    if not quiet:
        # Haptic pulse — two short buzzes
        results["vibrate"] = _run(["termux-vibrate", "-d", "250"])
        time.sleep(0.35)
        results["vibrate2"] = _run(["termux-vibrate", "-d", "150"])
    # On-screen transient popup
    results["toast"] = _run(["termux-toast", "-g", "top", msg])
    # Notification shade (persists until dismissed)
    nid = "hermes-%d" % int(time.time())
    results["notify"] = _run(
        [
            "termux-notification",
            "--id",
            nid,
            "--title",
            TITLE,
            "--content",
            msg,
            "--priority",
            "high",
        ]
    )
    ok = any(results.values())
    line = json.dumps(
        {
            "ts": int(time.time()),
            "msg": msg[:120],
            "channels": results,
            "ok": ok,
        }
    )
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    return ok


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--quiet"]
    quiet = "--quiet" in sys.argv
    if not args:
        print('usage: hermes_presence.py [--quiet] "message"')
        sys.exit(1)
    sys.exit(0 if fire(" ".join(args), quiet=quiet) else 1)
