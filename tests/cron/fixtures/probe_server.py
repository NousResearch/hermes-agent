"""Synthetic JSON-RPC stdio server; only reads/writes its private control files."""

import json
import os
from pathlib import Path
import sys
import threading

home = Path(sys.argv[1])
waiter = threading.Event()
mode = (home / "mode").read_text()
if mode == "down":
    sys.exit(1)
if mode in ("recover", "delayed-list"):
    while not (home / "restore").exists():
        waiter.wait(0.02)
    waiter.wait(1)
(home / "pid").write_text(str(os.getpid()))


def initialize(request):
    return {
        "protocolVersion": request["params"]["protocolVersion"],
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "readiness-fixture", "version": "1"},
    }


def list_tools(_request):
    if mode == "delayed-list":
        waiter.wait(2)
    return {
        "tools": [
            {
                "name": "hello",
                "description": "Synthetic greeting",
                "inputSchema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            }
        ]
    }


def call_tool(request):
    return {
        "content": [
            {"type": "text", "text": "hello " + request["params"]["arguments"]["name"]}
        ],
        "isError": False,
    }


handlers = {
    "initialize": initialize,
    "tools/list": list_tools,
    "tools/call": call_tool,
    "resources/list": lambda _request: {"resources": []},
    "prompts/list": lambda _request: {"prompts": []},
    "ping": lambda _request: {},
}
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    handler = handlers.get(request["method"])
    response = {"jsonrpc": "2.0", "id": request["id"]}
    if handler is None:
        response["error"] = {"code": -32601, "message": "Method not found"}
    else:
        response["result"] = handler(request)
    print(json.dumps(response), flush=True)
