#!/usr/bin/env python3
"""Sequence test: EnterPlanMode -> EnterWorktree -> Agent as SubTask.

This probes whether Claude MCP requires prior session state before Agent
subagent types resolve correctly.
"""
import asyncio
import json
import os

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

CLAUDE = os.environ.get("CLAUDE_CLI_PATH", "/Users/tusker/.local/bin/claude")
ENV_KEYS = {"HOME", "USER", "PATH", "TERM", "SHELL", "LANG", "LC_ALL", "TMPDIR"}


def dump_result(label, result):
    print(f"\n=== {label} ===")
    print(json.dumps(result.model_dump(mode="json"), indent=2)[:40000])


async def safe_call(session, tool_name, payload=None):
    payload = payload or {}
    result = await session.call_tool(tool_name, payload)
    dump_result(tool_name, result)
    return result


async def main() -> int:
    env = {k: v for k, v in os.environ.items() if k in ENV_KEYS}
    server = StdioServerParameters(command=CLAUDE, args=["mcp", "serve"], env=env)

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            print("INIT:", init)
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("TOOLS:", names)

            if "EnterPlanMode" in names:
                try:
                    await safe_call(session, "EnterPlanMode", {})
                except Exception as exc:
                    print("EnterPlanMode exception:", repr(exc))

            if "EnterWorktree" in names:
                try:
                    await safe_call(session, "EnterWorktree", {})
                except Exception as exc:
                    print("EnterWorktree exception:", repr(exc))

            agent_payloads = [
                {
                    "description": "General purpose smoke test",
                    "prompt": "Reply with exactly GENERAL_OK",
                    "subagent_type": "general-purpose",
                },
                {
                    "description": "Explore smoke test",
                    "prompt": "Reply with exactly EXPLORE_OK",
                    "subagent_type": "Explore",
                },
                {
                    "description": "Swarm worker smoke test",
                    "prompt": "Reply with exactly SWARM_WORKER_OK",
                    "subagent_type": "swarm:worker",
                },
                {
                    "description": "Swarm coordinator smoke test",
                    "prompt": "Reply with exactly SWARM_COORD_OK",
                    "subagent_type": "swarm:coordinator",
                },
            ]

            if "Agent" in names:
                for payload in agent_payloads:
                    try:
                        result = await session.call_tool("Agent", payload)
                        dump_result(f"Agent {payload['subagent_type']}", result)
                    except Exception as exc:
                        print(f"Agent {payload['subagent_type']} exception:", repr(exc))

            if "ExitPlanMode" in names:
                try:
                    await safe_call(session, "ExitPlanMode", {})
                except Exception as exc:
                    print("ExitPlanMode exception:", repr(exc))

            if "ExitWorktree" in names:
                try:
                    await safe_call(session, "ExitWorktree", {})
                except Exception as exc:
                    print("ExitWorktree exception:", repr(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
