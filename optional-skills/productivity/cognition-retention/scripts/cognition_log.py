#!/usr/bin/env python3
"""Log a learning event to Cognition.

Usage:
    python scripts/cognition_log.py "react-server-components"
    python scripts/cognition_log.py "react-server-components" --label "React Server Components"
    python scripts/cognition_log.py "react-server-components" --topic "React" --score 0.85 --weight active
    python scripts/cognition_log.py "docker-networking" --source debugging --weight active --excerpt "Fixed bridge network DNS resolution"

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

    if len(sys.argv) < 2:
        print("Usage: cognition_log.py <concept> [--label L] [--topic T] [--score S] [--weight W] [--source S] [--excerpt E]")
        sys.exit(1)

    concept = sys.argv[1]
    args = {"concept": concept, "source_integration": "hermes_agent"}

    i = 2
    while i < len(sys.argv):
        flag = sys.argv[i]
        val = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        if flag == "--label":
            args["label"] = val
        elif flag == "--topic":
            args["topic"] = val
        elif flag == "--score":
            args["score"] = float(val)
        elif flag == "--weight":
            args["practice_weight"] = val
        elif flag == "--source":
            args["source"] = val
        elif flag == "--excerpt":
            args["excerpt"] = val
        i += 2

    result = call_mcp("log_learning", args)

    status = "NEW" if result.get("isNew") else "UPDATED"
    ret = result.get("newRetention", "?")
    nxt = result.get("nextReviewAt", "?")
    print(f"✓ {status}: {concept} — retention {ret}%, next review {nxt}")


if __name__ == "__main__":
    main()
