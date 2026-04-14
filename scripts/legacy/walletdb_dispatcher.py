#!/usr/bin/env python3
"""Legacy dispatcher (DEPRECATED).

WalletDB delivery is now owned by systemd service: walletdb-alerts-deliver.service.

This script is kept only for debugging. It is disabled by default to prevent
extra DB contention / duplicate flows.

To run anyway:
  ALLOW_LEGACY_DISPATCH=1 ./walletdb_dispatcher.py
"""

import json
import os
import subprocess
import time
import sys
from datetime import datetime

if os.getenv("ALLOW_LEGACY_DISPATCH") not in {"1", "true", "TRUE", "yes", "YES"}:
    sys.stderr.write(
        "walletdb_dispatcher.py is deprecated and disabled by default. "
        "Use systemd walletdb-alerts-deliver.service + `walletdb alerts status|simulate|analytics`. "
        "Set ALLOW_LEGACY_DISPATCH=1 to run anyway.\n"
    )
    raise SystemExit(2)

CMD = "cd /home/yeqiuqiu/projects/walletdb && . .venv/bin/activate && timeout 8s python -m walletdb.cli alerts dispatch --db-path data/walletdb.sqlite --limit 10 --no-mark --json"
SPOOL_PATH = "/home/yeqiuqiu/clawd-architect/state/walletdb_alert_spool.jsonl"


def now():
    return datetime.now().isoformat(timespec="seconds")


def main():
    while True:
        try:
            p = subprocess.run(
                ["bash", "-lc", CMD],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if p.returncode != 0:
                # minimal error log
                sys.stderr.write(f"[{now()}] walletdb dispatch error rc={p.returncode}\n")
                time.sleep(5)
                continue

            out = (p.stdout or "").strip()
            if not out:
                time.sleep(1)
                continue

            try:
                alerts = json.loads(out)
            except Exception:
                sys.stderr.write(f"[{now()}] walletdb dispatch returned non-JSON output\n")
                time.sleep(5)
                continue

            if isinstance(alerts, dict) and "alerts" in alerts:
                alerts = alerts.get("alerts")

            if not alerts:
                time.sleep(1)
                continue

            # Spool alerts for another process to deliver.
            # Each line: {"ts":..., "text":..., "raw":...}
            with open(SPOOL_PATH, "a", encoding="utf-8") as f:
                for a in alerts:
                    try:
                        text = a.get("text") if isinstance(a, dict) else str(a)
                    except Exception:
                        text = str(a)
                    rec = {"ts": now(), "text": text, "raw": a}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            time.sleep(1)

        except subprocess.TimeoutExpired:
            sys.stderr.write(f"[{now()}] walletdb dispatch timeout\n")
            time.sleep(5)
        except Exception as e:
            sys.stderr.write(f"[{now()}] dispatcher exception: {e}\n")
            time.sleep(5)


if __name__ == "__main__":
    main()
