"""
Feed email data into the calumai peer in the calum-selfhost workspace.

Usage:
    python honcho-email-ingest.py path/to/emails.json

Input JSON format — list of objects:
    [
      {
        "subject": "...",
        "from": "sender@example.com",
        "body": "...",
        "timestamp": "2026-06-10T08:00:00Z",  # ISO 8601 UTC
        "thread_id": "unique-thread-id"
      },
      ...
    ]

Schedule via Windows Task Scheduler (daily at 6 AM):
    $action = New-ScheduledTaskAction -Execute "python" -Argument "C:\\dev\\ai\\hermes-agent\\scripts\\honcho-email-ingest.py C:\\Users\\calumai\\exported-emails.json"
    $trigger = New-ScheduledTaskTrigger -Daily -At "06:00AM"
    Register-ScheduledTask -TaskName "HonchoEmailIngest" -Action $action -Trigger $trigger
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from honcho import Honcho

HONCHO_BASE_URL = "http://localhost:8000"
WORKSPACE = "calum-selfhost"
PEER_NAME = "calumai"
BATCH_SIZE = 100


def ingest(emails_path: str) -> None:
    client = Honcho(base_url=HONCHO_BASE_URL)
    workspace = client.workspace(WORKSPACE)
    user = workspace.peer(PEER_NAME)
    agent = workspace.peer("cron_agent", observe_me=False)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session = workspace.session(f"email-import-{today}")
    session.add_peers([user, agent])

    emails = json.loads(Path(emails_path).read_text(encoding="utf-8"))
    print(f"Ingesting {len(emails)} emails into {WORKSPACE}/{PEER_NAME}...")

    messages = [
        user.message(
            f"Subject: {e['subject']}\nFrom: {e['from']}\n\n{e['body']}",
            metadata={"source": "gmail", "thread_id": e["thread_id"]},
            created_at=e["timestamp"],
        )
        for e in emails
    ]

    for i in range(0, len(messages), BATCH_SIZE):
        batch = messages[i : i + BATCH_SIZE]
        session.add_messages(batch)
        print(f"  Ingested {i}-{min(i + BATCH_SIZE, len(messages))}")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python honcho-email-ingest.py path/to/emails.json", file=sys.stderr)
        sys.exit(1)
    ingest(sys.argv[1])
