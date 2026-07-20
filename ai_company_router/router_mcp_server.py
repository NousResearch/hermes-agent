#!/usr/bin/env python3
"""MCP server wrapper for the AI Company LangGraph router.

Exposes a single tool `route_task` that:
1. Writes the task description to the Obsidian vault `input.md`
2. Invokes the router graph
3. Returns the final status and dashboard path.

Runs over stdio. Configure in `~/.hermes/config.yaml`:

    ai_company:
      command: /path/to/venv/bin/python3
      args:
        - /path/to/ai_company_router/router_mcp_server.py
      env:
        OBSIDIAN_VAULT_PATH: /path/to/obsidian_vault
        PYTHONPATH: /path/to/repo/parent
      enabled: true
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow `from ai_company_router.router import build_graph` to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_company_router.obsidian_io import VAULT_PATH, read_task, write_status
from ai_company_router.router import build_graph


def send(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def route_task(description: str, target_os: list[str] | None = None, priority: str = "medium") -> dict:
    os.makedirs(VAULT_PATH, exist_ok=True)
    input_path = Path(VAULT_PATH) / "input.md"
    input_path.write_text(description, encoding="utf-8")

    graph = build_graph()
    final_state = graph.invoke({
        "messages": [],
        "task": None,
        "plan": [],
        "results": {},
        "status": "pending",
        "errors": [],
        "skill_patch": None,
    })

    return {
        "status": final_state.get("status"),
        "task_id": (final_state.get("task") or {}).get("id"),
        "results": final_state.get("results", {}),
        "errors": final_state.get("errors", []),
        "dashboard_path": str(Path(VAULT_PATH) / "output" / "dashboard.md"),
        "input_path": str(input_path),
    }


def main():
    for line in sys.stdin:
        req = json.loads(line)
        req_id = req.get("id")
        method = req.get("method", "")

        if method == "initialize":
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "serverInfo": {"name": "ai-company-router", "version": "0.1.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                },
            })
        elif method == "tools/list":
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [{
                        "name": "route_task",
                        "description": "Dispatch a task to the AI Company LangGraph router. Writes to Obsidian vault and returns execution status.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string", "description": "Task description to route"},
                                "target_os": {"type": "array", "items": {"type": "string"}, "description": "Target OS agents (mac, windows, any)"},
                                "priority": {"type": "string", "default": "medium", "description": "Task priority"},
                            },
                            "required": ["description"],
                        },
                    }]
                },
            })
        elif method == "tools/call":
            params = req.get("params", {})
            args = params.get("arguments", {})
            try:
                result = route_task(
                    description=args.get("description", ""),
                    target_os=args.get("target_os") or ["any"],
                    priority=args.get("priority", "medium"),
                )
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                })
            except Exception as exc:
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": f"Router error: {exc}"},
                })
        elif method == "notifications/initialized":
            pass


if __name__ == "__main__":
    main()
