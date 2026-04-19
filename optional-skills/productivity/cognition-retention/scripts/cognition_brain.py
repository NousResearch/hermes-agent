#!/usr/bin/env python3
"""Full brain health report from Cognition.

Usage:
    python scripts/cognition_brain.py

Combines retention overview + weak topics + review suggestions
into a single dashboard printout.
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

    # 1. Retention overview
    retention = call_mcp("get_user_retention", {})
    overall = retention.get("overallRetention", 0)
    total = retention.get("totalNodes", 0)
    label = "Solid" if overall >= 80 else "Slipping" if overall >= 60 else "At risk"

    print(f"🧠 Brain Health Report")
    print(f"{'─' * 45}")
    print(f"  Overall retention: {overall}% ({label})")
    print(f"  Total concepts tracked: {total}")

    # Topics
    topics = retention.get("topics", [])
    if topics:
        print(f"\n📊 Topics ({len(topics)}):")
        for t in topics[:8]:
            status_icon = "🟢" if t["status"] == "strong" else "🟡" if t["status"] == "fading" else "🔴"
            print(f"  {status_icon} {t['topic']:30s} {t['avgRetention']:>3}%  ({t['nodeCount']} concepts)")

    # 2. Weak topics
    weak = call_mcp("get_weak_topics", {"threshold": 0.6})
    weak_topics = weak.get("weakTopics", [])
    if weak_topics:
        print(f"\n📉 Below 60% threshold ({len(weak_topics)} concepts):")
        for w in weak_topics[:5]:
            print(f"  • {w['concept']:30s} [{w['topic']}]  {w['retention']}%")

    # 3. Review suggestions
    reviews = call_mcp("suggest_review", {"count": 5})
    suggestions = reviews.get("suggestions", [])
    if suggestions:
        print(f"\n📚 Review now:")
        for i, s in enumerate(suggestions, 1):
            icon = "🔴" if s.get("urgency") == "high" else "🟡" if s.get("urgency") == "medium" else "🟢"
            print(f"  {i}. {icon} {s['concept']}  [{s['topic']}]  {s['currentRetention']}%")

    if not weak_topics and not suggestions:
        print(f"\n✅ All concepts above threshold — nothing urgent to review.")

    print(f"\n{'─' * 45}")
    print(f"Powered by Cognition — cognitionus.com")


if __name__ == "__main__":
    main()
