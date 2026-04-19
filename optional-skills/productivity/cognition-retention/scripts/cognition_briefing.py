#!/usr/bin/env python3
"""Fetch your Cognition retention briefing at session start.

Usage:
    python scripts/cognition_briefing.py
    python scripts/cognition_briefing.py --max-weak 5 --max-due 5

Calls get_session_context on your Cognition server and prints:
  - Overall retention score
  - Weakest concepts (decaying fastest)
  - Concepts due for review
  - Teammate nudges (things your team knows that you don't)

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


def call_mcp(method: str, tool_name: str, arguments: dict) -> dict:
    """Send a JSON-RPC request to the Cognition MCP server."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
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
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    content = result.get("result", {}).get("content", [{}])
    return json.loads(content[0].get("text", "{}"))


def main():
    if not COGNITION_KEY:
        print("Set COGNITION_API_KEY or run: hermes config set cognition_api_key <key>")
        sys.exit(1)

    max_weak = 3
    max_due = 3
    max_nudges = 2
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--max-weak" and i < len(sys.argv) - 1:
            max_weak = int(sys.argv[i + 1])
        if arg == "--max-due" and i < len(sys.argv) - 1:
            max_due = int(sys.argv[i + 1])

    data = call_mcp("tools/call", "get_session_context", {
        "max_weak": max_weak,
        "max_due": max_due,
        "max_nudges": max_nudges,
    })

    overall = data.get("overallRetention", 0)
    label = "Solid" if overall >= 80 else "Slipping" if overall >= 60 else "At risk"

    print(f"🧠 Brain — {overall}% retention ({label})")
    print("─" * 40)

    weak = data.get("weakest", [])
    if weak:
        print("\n📉 Weakest concepts:")
        for w in weak:
            print(f"  • {w['concept']}  [{w['topic']}]  {w.get('retention', '?')}%")

    due = data.get("dueForReview", [])
    if due:
        print("\n📚 Due for review:")
        for d in due:
            print(f"  • {d['concept']}  [{d['topic']}]")

    nudges = data.get("teammateNudges", [])
    if nudges:
        print("\n👥 Teammate signal:")
        for n in nudges:
            print(f"  • {n['concept']}  [{n['topic']}]")

    briefing = data.get("briefing", "")
    if briefing:
        print(f"\n💡 {briefing}")

    op = data.get("operatorBriefing")
    if op:
        print(f"\n🔬 Operator: {op}")

    study = data.get("shouldStudyOrRest")
    if study:
        print(f"\n⚡ {study}")


if __name__ == "__main__":
    main()
