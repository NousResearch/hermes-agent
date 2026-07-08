#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.jarvis_notification_queue import mark_notification, pending_notifications

DEFAULT_QUEUE = Path.home() / ".hermes" / "handoffs" / "jarvis" / "notifications.jsonl"


def _send_discord(message: str, target: str, timeout: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["hermes", "send", "--to", target, message],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return False, exc.__class__.__name__
    detail = (proc.stdout or proc.stderr or "").strip()[:500]
    return proc.returncode == 0, detail


def process_queue(*, queue: Path, target: str, dry_run: bool, timeout: int) -> int:
    processed = 0
    for item in pending_notifications(queue):
        notification_id = str(item.get("notification_id") or "")
        message = str(item.get("message") or "").strip()
        if not notification_id:
            continue
        if not message:
            mark_notification(queue, notification_id, "failed", delivery_result="empty message")
            processed += 1
            continue
        if dry_run:
            mark_notification(queue, notification_id, "sent", delivery_result=f"dry-run:{target}")
            processed += 1
            continue
        ok, detail = _send_discord(message, target, timeout)
        mark_notification(queue, notification_id, "sent" if ok else "failed", delivery_result=detail)
        processed += 1
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description="Send pending Jarvis handoff notifications via the default profile.")
    parser.add_argument("--queue", default=os.getenv("JARVIS_NOTIFICATION_QUEUE", str(DEFAULT_QUEUE)))
    parser.add_argument("--target", default=os.getenv("JARVIS_NOTIFICATION_TARGET", "discord"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("JARVIS_NOTIFICATION_SEND_TIMEOUT", "60")))
    parser.add_argument("--dry-run", action="store_true", default=os.getenv("JARVIS_NOTIFICATION_DRY_RUN", "") == "1")
    args = parser.parse_args()
    processed = process_queue(queue=Path(args.queue), target=args.target, dry_run=args.dry_run, timeout=args.timeout)
    print(f"processed={processed} queue={args.queue} target={args.target} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
