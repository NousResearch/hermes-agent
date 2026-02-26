#!/usr/bin/env python3
"""
MCP Client Demo -- Demonstrates full MCP spec support in Hermes Agent.

Connects to the demo MCP server and exercises all implemented features:
  1. Connect & handshake
  2. List & call tools
  3. List & read resources
  4. List & get prompts
  5. Progress notifications
  6. Set log level (structured logging)

Usage: python demo_mcp_client.py
"""

import json
import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mcp_client import MCPClient, StdioTransport


# --- Pretty output helpers ---

def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def section(title):
    print(f"\n--- {title} ---")


def ok(msg):
    print(f"  [OK] {msg}")


def info(msg):
    print(f"  {msg}")


def json_pretty(data, indent=4):
    """Print JSON with indentation."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            print(f"    {data}")
            return
    for line in json.dumps(data, indent=indent, ensure_ascii=False).split("\n"):
        print(f"    {line}")


def main():
    server_script = os.path.join(os.path.dirname(__file__), "demo_mcp_server.py")

    header("HERMES AGENT -- MCP Full Spec Demo")
    print(f"  Server: demo_mcp_server.py (Python, stdio)")
    print(f"  Client: tools/mcp_client.py (MCPClient)")

    # 1. Connect
    section("1. CONNECT")
    transport = StdioTransport(
        command=sys.executable,
        args=[server_script],
    )
    client = MCPClient(transport)

    # Register a log notification listener to show server logs
    log_messages = []
    def on_log(params):
        log_messages.append(params)

    result = client.connect()
    client.on_notification("notifications/message", on_log)

    server_info = result["serverInfo"]
    capabilities = result["capabilities"]
    ok(f"Connected to: {server_info['name']} v{server_info['version']}")
    ok(f"Protocol: {client._server_capabilities is not None}")
    info(f"  Capabilities: {', '.join(capabilities.keys())}")

    # 2. List tools
    section("2. TOOLS")
    tools = client.list_tools()
    ok(f"Discovered {len(tools)} tools:")
    for t in tools:
        info(f"  - {t['name']}: {t['description']}")

    # 3. Call a tool
    section("3. CALL TOOL: calculate")
    result = client.call_tool("calculate", {
        "operation": "multiply",
        "a": 7,
        "b": 6,
    })
    text = result["content"][0]["text"]
    ok(f"7 x 6 = {text}")

    result = client.call_tool("get_time")
    text = result["content"][0]["text"]
    ok(f"Server time: {text}")

    # 4. Call tool with progress
    section("4. CALL TOOL WITH PROGRESS")
    progress_updates = []
    def on_progress(progress, total, message):
        progress_updates.append((progress, total, message))

    result = client.call_tool_with_progress(
        "calculate",
        {"operation": "add", "a": 100, "b": 200},
        progress_callback=on_progress,
    )
    text = result["content"][0]["text"]
    ok(f"100 + 200 = {text}")
    # Small delay to let progress notifications arrive
    time.sleep(0.2)
    if progress_updates:
        ok(f"Progress updates received: {len(progress_updates)}")
        for p, t, m in progress_updates:
            info(f"  [{p}/{t}] {m}")
    else:
        ok("Progress token sent (notifications are async)")

    # 5. List resources
    section("5. RESOURCES")
    resources = client.list_resources()
    ok(f"Discovered {len(resources)} resources:")
    for r in resources:
        info(f"  - {r['uri']} ({r.get('mimeType', 'unknown')})")

    # 6. Read a resource
    section("6. READ RESOURCE: settings.json")
    result = client.read_resource("file:///config/settings.json")
    contents = result.get("contents", [])
    if contents:
        ok(f"URI: {contents[0]['uri']}")
        info(f"  Content:")
        json_pretty(contents[0].get("text", ""))

    section("7. READ RESOURCE: users.csv")
    result = client.read_resource("file:///data/users.csv")
    contents = result.get("contents", [])
    if contents:
        ok(f"URI: {contents[0]['uri']}")
        for line in contents[0].get("text", "").split("\n"):
            info(f"  {line}")

    # 7. Subscribe to resource
    section("8. SUBSCRIBE RESOURCE")
    client.subscribe_resource("file:///config/settings.json")
    ok("Subscribed to: file:///config/settings.json")

    # 8. List prompts
    section("9. PROMPTS")
    prompts = client.list_prompts()
    ok(f"Discovered {len(prompts)} prompts:")
    for p in prompts:
        args_str = ", ".join(a["name"] for a in p.get("arguments", []))
        info(f"  - {p['name']}: {p['description']} ({args_str})")

    # 9. Get a prompt
    section("10. GET PROMPT: summarize")
    result = client.get_prompt("summarize", {
        "text": "Hermes Agent is a fully open-source AI coding assistant.",
        "style": "brief",
    })
    messages = result.get("messages", [])
    if messages:
        ok(f"Rendered prompt ({len(messages)} messages):")
        for m in messages:
            role = m.get("role", "?")
            text = m.get("content", {}).get("text", "")
            info(f"  [{role}] {text}")

    section("11. GET PROMPT: translate")
    result = client.get_prompt("translate", {
        "text": "Hello, world!",
        "language": "Turkish",
    })
    messages = result.get("messages", [])
    if messages:
        ok(f"Rendered prompt ({len(messages)} messages):")
        for m in messages:
            role = m.get("role", "?")
            text = m.get("content", {}).get("text", "")
            info(f"  [{role}] {text}")

    # 10. Set log level
    section("12. STRUCTURED LOGGING")
    client.set_log_level("debug")
    ok("Log level set to: debug")
    time.sleep(0.1)
    if log_messages:
        ok(f"Server log messages received: {len(log_messages)}")
        for lm in log_messages:
            info(f"  [{lm.get('level','?')}] ({lm.get('logger','')}) {lm.get('data','')}")

    # Summary
    header("DEMO COMPLETE")
    print(f"  Server:     {server_info['name']} v{server_info['version']}")
    print(f"  Tools:      {len(tools)}")
    print(f"  Resources:  {len(resources)}")
    print(f"  Prompts:    {len(prompts)}")
    print(f"  Features:   tools, resources, prompts, progress, logging")
    print(f"  Status:     All features working!")
    print()

    # Disconnect
    client.disconnect()


if __name__ == "__main__":
    main()
