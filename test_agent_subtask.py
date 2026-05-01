#!/usr/bin/env python3
"""Local Claude MCP subtask/agent harness.

Runs Claude Code MCP over stdio using several startup variants, then attempts
Agent/SubTask execution with built-in, plugin, and custom smoke-agent types.
"""
import asyncio
import json
import os

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

CLAUDE = os.environ.get("CLAUDE_CLI_PATH", "/Users/tusker/.local/bin/claude")
ENV_KEYS = {"HOME", "USER", "PATH", "TERM", "SHELL", "LANG", "LC_ALL", "TMPDIR"}

INLINE_AGENTS_JSON = json.dumps(
    {
        "mcp-inline-smoke": {
            "description": "Inline MCP smoke test agent",
            "prompt": "Reply with exactly MCP_INLINE_OK",
            "tools": ["Read"],
            "model": "haiku",
        }
    }
)

VARIANTS = [
    {
        "label": "default",
        "args": ["mcp", "serve"],
    },
    {
        "label": "settings-sources",
        "args": ["--setting-sources", "user,project,local", "mcp", "serve"],
    },
    {
        "label": "inline-agent",
        "args": ["--agents", INLINE_AGENTS_JSON, "mcp", "serve"],
    },
    {
        "label": "settings-and-inline-agent",
        "args": [
            "--setting-sources",
            "user,project,local",
            "--agents",
            INLINE_AGENTS_JSON,
            "mcp",
            "serve",
        ],
    },
]

CASES = [
    {
        "label": "default-agent",
        "payload": {
            "description": "Default agent smoke test",
            "prompt": "Reply with exactly SUBAGENT_OK",
        },
    },
    {
        "label": "built-in-general-purpose",
        "payload": {
            "description": "General purpose smoke test",
            "prompt": "Reply with exactly GENERAL_OK",
            "subagent_type": "general-purpose",
        },
    },
    {
        "label": "built-in-Plan",
        "payload": {
            "description": "Plan smoke test",
            "prompt": "Reply with exactly PLAN_OK",
            "subagent_type": "Plan",
        },
    },
    {
        "label": "built-in-explore",
        "payload": {
            "description": "Explore smoke test",
            "prompt": "Reply with exactly EXPLORE_OK",
            "subagent_type": "Explore",
        },
    },
    {
        "label": "swarm-worker",
        "payload": {
            "description": "Swarm worker smoke test",
            "prompt": "Reply with exactly SWARM_WORKER_OK",
            "subagent_type": "swarm:worker",
        },
    },
    {
        "label": "swarm-coordinator",
        "payload": {
            "description": "Swarm coordinator smoke test",
            "prompt": "Reply with exactly SWARM_COORD_OK",
            "subagent_type": "swarm:coordinator",
        },
    },
    {
        "label": "local-smoke-agent",
        "payload": {
            "description": "Project local smoke agent test",
            "prompt": "Reply with exactly MCP_SMOKE_OK",
            "subagent_type": "mcp-local-smoke",
        },
    },
    {
        "label": "inline-smoke-agent",
        "payload": {
            "description": "Inline smoke agent test",
            "prompt": "Reply with exactly MCP_INLINE_OK",
            "subagent_type": "mcp-inline-smoke",
        },
    },
]


async def run_variant(variant: dict) -> None:
    env = {k: v for k, v in os.environ.items() if k in ENV_KEYS}
    server = StdioServerParameters(command=CLAUDE, args=variant["args"], env=env)

    print(f"\n================ VARIANT: {variant['label']} ================")
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            print("INIT:", init)
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print("TOOLS:", names)

            for case in CASES:
                label = case["label"]
                payload = case["payload"]
                print(f"\n=== CASE: {label} ===")
                try:
                    result = await session.call_tool("Agent", payload)
                    print(json.dumps(result.model_dump(mode="json"), indent=2)[:40000])
                except Exception as exc:
                    print(f"EXCEPTION: {exc!r}")


async def main() -> int:
    for variant in VARIANTS:
        try:
            await run_variant(variant)
        except Exception as exc:
            print(f"\nVARIANT FAILURE {variant['label']}: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
