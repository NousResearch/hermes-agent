#!/usr/bin/env python3
"""Optional stdio MCP-like operator surface over the Kubrick CLI.

Does not become authoritative. Weak evidence fails closed. Requires no extra
dependencies beyond the skill's declared Python stack.

Primary tool: kubrick_do → python scripts/kubrick.py do <intent> …

Protocol: newline-delimited JSON-RPC 2.0 messages on stdin/stdout.
Methods: initialize, tools/list, tools/call, ping.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

TOOLS = [
    {
        "name": "kubrick_do",
        "description": (
            "Run a Kubrick intent (compile, retrieve, adapt, visual, learn, check, …)"
        ),
        "inputSchema": {
            "type": "object",
            "required": ["intent"],
            "properties": {
                "intent": {"type": "string"},
                "action": {"type": "string"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "CLI flags after intent/action, e.g. "
                        '["--provider","flux","--packet","p.yaml"]'
                    ),
                },
            },
        },
    }
]


def respond(msg_id: Any, result: Any = None, error: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name != "kubrick_do":
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"unknown tool {name}"}],
        }
    arguments = arguments or {}
    intent = arguments.get("intent")
    if not intent:
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": "kubrick_do requires 'intent' (e.g. adapt, visual, check)",
                }
            ],
            "structuredContent": {
                "status": "NOT_COMPUTABLE",
                "exit_code": 2,
                "fail_closed": True,
            },
        }
    action = arguments.get("action")
    args = arguments.get("args") or []
    if not isinstance(args, list):
        args = [args]

    cli: list[str] = ["do", str(intent)]
    if action:
        cli += ["--action", str(action)]
    cli += [str(a) for a in args]

    proc = subprocess.run(
        [PY, str(ROOT / "scripts/kubrick.py"), *cli],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if proc.returncode != 0:
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": text.strip() or f"tool failed with code {proc.returncode}",
                }
            ],
            "structuredContent": {
                "status": "NOT_COMPUTABLE",
                "exit_code": proc.returncode,
                "fail_closed": True,
            },
        }
    return {
        "isError": False,
        "content": [{"type": "text", "text": text.strip()}],
        "structuredContent": {"status": "OK", "exit_code": 0, "fail_closed": True},
    }


def handle(message: dict[str, Any]) -> None:
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        respond(
            msg_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "kubrick-operators", "version": "0.14.0"},
                "instructions": (
                    "Optional Kubrick operator surface. Use kubrick_do with intent "
                    "(and optional action/args). CLI remains authoritative for local work. "
                    "Forge remains canonical for committed project state. Fail closed on weak evidence."
                ),
            },
        )
        return
    if method == "notifications/initialized":
        return
    if method == "ping":
        respond(msg_id, {})
        return
    if method == "tools/list":
        respond(
            msg_id,
            {
                "tools": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "inputSchema": t["inputSchema"],
                    }
                    for t in TOOLS
                ]
            },
        )
        return
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        respond(msg_id, call_tool(name, arguments))
        return
    if msg_id is not None:
        respond(msg_id, error={"code": -32601, "message": f"Method not found: {method}"})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        handle(message)


if __name__ == "__main__":
    main()
