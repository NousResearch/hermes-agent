#!/usr/bin/env python3
"""Get review recommendations from Cognition — what to study next.

Usage:
    python scripts/cognition_review.py
    python scripts/cognition_review.py --count 10

Returns concepts ranked by urgency (most overdue first).
No dependencies beyond Python stdlib.
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
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    content = result.get("result", {}).get("content", [{}])
    return json.loads(content[0].get("text", "{}"))


def main():
    if not COGNITION_KEY:
        print("Set COGNITION_API_KEY or run: hermes config set cognition_api_key <key>")
        sys.exit(1)

    count = 5
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--count" and i < len(sys.argv) - 1:
            count = int(sys.argv[i + 1])

    data = call_mcp("suggest_review", {"count": count})
    suggestions = data.get("suggestions", [])

    if not suggestions:
        print("✅ Nothing to review — all concepts are fresh!")
        return

    print("📚 Review now (ranked by urgency):")
    print("─" * 45)
    for i, s in enumerate(suggestions, 1):
        urgency = s.get("urgency", "?")
        icon = "🔴" if urgency == "high" else "🟡" if urgency == "medium" else "🟢"
        overdue = "OVERDUE" if s.get("overdue") else ""
        print(f"  {i}. {icon} {s['concept']}  [{s['topic']}]")
        print(f"     Retention: {s['currentRetention']}%  |  Reviews: {s.get('reviewCount', '?')}  {overdue}")


if __name__ == "__main__":
    main()
