#!/usr/bin/env python3
"""Dispatch OTP messages queued by the public delivery sandbox.

The delivery sandbox intentionally has no WhatsApp/Telegram secrets. It writes
OTP requests to otp_outbox.jsonl; this trusted Hermes-side script sends pending
codes through registered gateway channels and records delivered event_ids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.send_message_tool import send_message_tool


def load_sent(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data}
    except Exception:
        pass
    return set()


def save_sent(path: Path, sent: set[str]) -> None:
    path.write_text(json.dumps(sorted(sent), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dir", default="/home/jean/zeus-runtime/delivery-sandbox/events")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    outbox = event_dir / "otp_outbox.jsonl"
    sent_path = event_dir / "otp_outbox_sent.json"
    sent = load_sent(sent_path)
    dispatched = 0
    errors: list[dict[str, str]] = []

    if not outbox.exists():
        print(json.dumps({"ok": True, "dispatched": 0, "errors": []}, ensure_ascii=False))
        return

    for line in outbox.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_id = str(item.get("event_id") or "")
        if not event_id or event_id in sent:
            continue
        target = str(item.get("target") or "")
        message = str(item.get("message") or "")
        if not target or not message:
            errors.append({"event_id": event_id, "error": "missing_target_or_message"})
            continue
        if args.dry_run:
            dispatched += 1
            continue
        result_raw = send_message_tool({"action": "send", "target": target, "message": message})
        try:
            result = json.loads(result_raw)
        except Exception:
            result = {"raw": result_raw}
        if isinstance(result, dict) and result.get("error"):
            errors.append({"event_id": event_id, "error": str(result.get("error"))})
            continue
        sent.add(event_id)
        dispatched += 1

    if not args.dry_run:
        save_sent(sent_path, sent)
    print(json.dumps({"ok": not errors, "dispatched": dispatched, "errors": errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
