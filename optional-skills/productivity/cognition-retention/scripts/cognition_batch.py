#!/usr/bin/env python3
"""Batch-log learning events to Cognition from a JSON file.

Usage:
    python scripts/cognition_batch.py concepts.json

The JSON file should be an array of objects:
[
  {"concept": "react-hooks", "label": "React Hooks", "topic": "React", "score": 0.8, "practice_weight": "active"},
  {"concept": "docker-compose", "label": "Docker Compose", "topic": "DevOps", "score": 0.7}
]

Max 200 events per call. No dependencies beyond Python stdlib.
"""

import json
import os
import sys
import urllib.request

COGNITION_URL = os.environ.get(
    "COGNITION_URL",
    "https://www.cognitionus.com/api/integrations/claude-code/mcp",
)
COGNITION_KEY = os.environ.get("COGNITION_API_KEY", "")


def call_mcp(tool_name: str, arguments: dict) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        COGNITION_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {COGNITION_KEY}",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    content = result.get("result", {}).get("content", [{}])
    return json.loads(content[0].get("text", "{}"))


def main():
    if not COGNITION_KEY:
        print("Set COGNITION_API_KEY or run: hermes config set cognition_api_key <key>")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: cognition_batch.py <concepts.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        events = json.load(f)

    if not isinstance(events, list):
        print("Error: JSON must be an array of event objects")
        sys.exit(1)

    # Add source_integration to each event
    for e in events:
        e.setdefault("source_integration", "hermes_agent")

    # Batch in chunks of 200
    total_ingested = 0
    total_created = 0
    total_errors = 0

    for i in range(0, len(events), 200):
        chunk = events[i:i + 200]
        result = call_mcp("log_learning_batch", {"events": chunk})
        total_ingested += result.get("ingested", 0)
        total_created += result.get("created", 0)
        total_errors += len(result.get("errors", []))

    print(f"✓ Batch complete: {total_ingested} ingested, {total_created} new, {total_errors} errors")


if __name__ == "__main__":
    main()
