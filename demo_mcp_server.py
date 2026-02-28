#!/usr/bin/env python3
"""
Minimal MCP server for demonstration purposes.

Speaks JSON-RPC 2.0 over stdin/stdout.
Supports: tools, resources, prompts, logging, progress, list_changed.

Usage: python demo_mcp_server.py
(Intended to be spawned by MCPClient via StdioTransport)
"""

import json
import sys
import time


def send(msg):
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def send_response(req_id, result):
    send({"jsonrpc": "2.0", "id": req_id, "result": result})


def send_notification(method, params=None):
    msg = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    send(msg)


# --- Server data ---

TOOLS = [
    {
        "name": "calculate",
        "description": "Perform basic arithmetic operations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current server time",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

RESOURCES = [
    {"uri": "file:///config/settings.json", "name": "settings.json", "mimeType": "application/json"},
    {"uri": "file:///data/users.csv", "name": "users.csv", "mimeType": "text/csv"},
    {"uri": "file:///docs/readme.txt", "name": "readme.txt", "mimeType": "text/plain"},
]

PROMPTS = [
    {
        "name": "summarize",
        "description": "Summarize the given text concisely",
        "arguments": [
            {"name": "text", "description": "The text to summarize", "required": True},
            {"name": "style", "description": "Summary style: brief or detailed", "required": False},
        ],
    },
    {
        "name": "translate",
        "description": "Translate text to a target language",
        "arguments": [
            {"name": "text", "description": "Text to translate", "required": True},
            {"name": "language", "description": "Target language", "required": True},
        ],
    },
]

RESOURCE_CONTENTS = {
    "file:///config/settings.json": '{"theme": "dark", "language": "en", "version": "2.1.0"}',
    "file:///data/users.csv": "id,name,email\n1,Alice,alice@example.com\n2,Bob,bob@example.com",
    "file:///docs/readme.txt": "Welcome to the Hermes MCP Demo Server!\nThis server demonstrates full MCP spec support.",
}

current_log_level = "info"


def handle_tool_call(name, arguments):
    if name == "calculate":
        op = arguments.get("operation", "add")
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        if op == "add":
            result = a + b
        elif op == "subtract":
            result = a - b
        elif op == "multiply":
            result = a * b
        elif op == "divide":
            result = a / b if b != 0 else "Error: division by zero"
        else:
            result = f"Unknown operation: {op}"
        return {"content": [{"type": "text", "text": str(result)}], "isError": False}
    elif name == "get_time":
        return {
            "content": [{"type": "text", "text": time.strftime("%Y-%m-%d %H:%M:%S")}],
            "isError": False,
        }
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        params = msg.get("params", {})
        req_id = msg.get("id")

        # Notifications (no id) -- ignore
        if req_id is None:
            continue

        if method == "initialize":
            send_response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True},
                    "prompts": {"listChanged": True},
                    "logging": {},
                },
                "serverInfo": {
                    "name": "hermes-demo-server",
                    "version": "1.0.0",
                },
            })
            # Send a log notification after init
            send_notification("notifications/message", {
                "level": "info",
                "logger": "demo",
                "data": "Server initialized successfully",
            })

        elif method == "tools/list":
            send_response(req_id, {"tools": TOOLS})

        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments", {})
            # If progress token provided, send progress notifications
            meta = params.get("_meta", {})
            token = meta.get("progressToken")
            if token:
                send_notification("notifications/progress", {
                    "progressToken": token,
                    "progress": 50,
                    "total": 100,
                    "message": "Processing...",
                })
            result = handle_tool_call(name, args)
            if token:
                send_notification("notifications/progress", {
                    "progressToken": token,
                    "progress": 100,
                    "total": 100,
                    "message": "Complete",
                })
            send_response(req_id, result)

        elif method == "resources/list":
            send_response(req_id, {"resources": RESOURCES})

        elif method == "resources/read":
            uri = params.get("uri", "")
            content = RESOURCE_CONTENTS.get(uri)
            if content:
                send_response(req_id, {
                    "contents": [{"uri": uri, "text": content, "mimeType": "text/plain"}],
                })
            else:
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32602, "message": f"Resource not found: {uri}",
                }})

        elif method == "resources/subscribe":
            send_response(req_id, {})

        elif method == "resources/unsubscribe":
            send_response(req_id, {})

        elif method == "prompts/list":
            send_response(req_id, {"prompts": PROMPTS})

        elif method == "prompts/get":
            name = params.get("name", "")
            args = params.get("arguments", {})
            if name == "summarize":
                text = args.get("text", "")
                style = args.get("style", "brief")
                send_response(req_id, {
                    "messages": [
                        {"role": "user", "content": {
                            "type": "text",
                            "text": f"Please provide a {style} summary of:\n\n{text}",
                        }},
                    ],
                })
            elif name == "translate":
                text = args.get("text", "")
                lang = args.get("language", "English")
                send_response(req_id, {
                    "messages": [
                        {"role": "user", "content": {
                            "type": "text",
                            "text": f"Translate the following to {lang}:\n\n{text}",
                        }},
                    ],
                })
            else:
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32602, "message": f"Unknown prompt: {name}",
                }})

        elif method == "logging/setLevel":
            current_log_level = params.get("level", "info")
            send_response(req_id, {})
            send_notification("notifications/message", {
                "level": "info",
                "logger": "demo",
                "data": f"Log level changed to: {current_log_level}",
            })

        else:
            send({"jsonrpc": "2.0", "id": req_id, "error": {
                "code": -32601, "message": f"Method not found: {method}",
            }})


if __name__ == "__main__":
    main()
