#!/usr/bin/env python3
"""Email Triage & Summarization Helper for JMAP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add script directory to sys.path to import jmap_client
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from jmap_client import format_addresses, resolve_client


def triage_inbox(client, limit: int = 20, unread_only: bool = True) -> dict:
    """Triage and categorize recent emails."""
    emails = client.list_emails(mailbox_name="INBOX", limit=limit, unread_only=unread_only)

    triage_result = {
        "total_analyzed": len(emails),
        "unread_count": sum(1 for e in emails if "$seen" not in (e.get("keywords") or {})),
        "items": [],
        "categories": {
            "action_required": [],
            "notifications": [],
            "general": [],
        },
    }

    action_keywords = ["urgent", "action required", "important", "please review", "approval", "asap", "deadline", "invoice"]
    notification_keywords = ["no-reply", "noreply", "notification", "alert", "digest", "newsletter", "update", "receipt"]

    for item in emails:
        eid = item.get("id")
        subj = item.get("subject", "")
        sender = format_addresses(item.get("from", []))
        preview = item.get("preview", "")
        date = item.get("receivedAt", "")

        combined_text = f"{subj} {preview}".lower()

        category = "general"
        if any(k in combined_text for k in action_keywords):
            category = "action_required"
        elif any(k in sender.lower() or k in combined_text for k in notification_keywords):
            category = "notifications"

        entry = {
            "id": eid,
            "subject": subj,
            "from": sender,
            "receivedAt": date,
            "preview": preview,
            "category": category,
        }

        triage_result["items"].append(entry)
        triage_result["categories"][category].append(entry)

    return triage_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage and summarize JMAP inbox")
    parser.add_argument("--limit", type=int, default=20, help="Number of emails to inspect")
    parser.add_argument("--all", action="store_true", help="Include read emails (default: unread only)")
    parser.add_argument("--session-url", help="JMAP session discovery URL")
    parser.add_argument("--token", help="Bearer API token")
    parser.add_argument("--account-id", help="JMAP account ID")
    parser.add_argument("--config", help="Path to credentials JSON file")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")

    args = parser.parse_args()

    try:
        client = resolve_client(args)
    except Exception as e:
        sys.stderr.write(f"Configuration error: {e}\n")
        return 1

    try:
        report = triage_inbox(client, limit=args.limit, unread_only=not args.all)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"==================================================")
            print(f"               INBOX TRIAGE REPORT                ")
            print(f"==================================================")
            print(f"Total Emails Analyzed: {report['total_analyzed']}")
            print(f"Unread Count: {report['unread_count']}")
            print()

            action_items = report["categories"]["action_required"]
            print(f"🚨 ACTION REQUIRED ({len(action_items)}):")
            if not action_items:
                print("  (None)")
            for item in action_items:
                print(f"  • [{item['id']}] From: {item['from']}")
                print(f"    Subject: {item['subject']}")
                if item["preview"]:
                    print(f"    Preview: {item['preview'][:100]}...")
            print()

            general_items = report["categories"]["general"]
            print(f"✉️ GENERAL MESSAGES ({len(general_items)}):")
            if not general_items:
                print("  (None)")
            for item in general_items:
                print(f"  • [{item['id']}] From: {item['from']}")
                print(f"    Subject: {item['subject']}")
            print()

            notifs = report["categories"]["notifications"]
            print(f"📢 NOTIFICATIONS & UPDATES ({len(notifs)}):")
            if not notifs:
                print("  (None)")
            for item in notifs:
                print(f"  • [{item['id']}] {item['from']} - {item['subject']}")
            print()

        return 0
    except Exception as e:
        sys.stderr.write(f"Triage error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
