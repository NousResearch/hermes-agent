#!/usr/bin/env python3
"""Extended sequence test 2: deeper session-state probe for MCP Agent resolution.

Sequence:
- EnterPlanMode
- EnterWorktree
- EnterPlanMode
- EnterWorktree
- Try Agent for built-in/plugin/custom types
- ExitPlanMode / ExitWorktree
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

AGENT_PAYLOADS = [
    {
        "description": "General purpose smoke test",
        "prompt": "Reply with exactly GENERAL_OK",
        "subagent_type": "general-purpose",
    },
    {
        "description": "Plan smoke test",
        "prompt": "Reply with exactly PLAN_OK",
        "subagent_type": "Plan",
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
    {
        "description": "Project local smoke agent test",
        "prompt": "Reply with exactly MCP_SMOKE_OK",
        "subagent_type": "mcp-local-smoke",
    },
    {
        "description": "Inline smoke agent test",
        "prompt": "Reply with exactly MCP_INLINE_OK",
        "subagent_type": "mcp-inline-smoke",
    },
]

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


def dump(label, payload) -> None:
    print(f"\n=== {label} ===")
    if hasattr(payload, "model_dump"):
        print(json.dumps(payload.model_dump(mode="json"), indent=2)[:40000])
    else:
        print(payload)


async def run_variant(variant: dict) -> None:
    env = {k: v for k, v in os.environ.items() if k in ENV_KEYS}
    if os.getenv("CLAUDE_CODE_FORK_SUBAGENT", "0") == "1":
        env["CLAUDE_CODE_FORK_SUBAGENT"] = "1"

    server = StdioServerParameters(command=CLAUDE, args=variant["args"], env=env)
    print(f"\n================ VARIANT: {variant['label']} ================")

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            dump("INIT", init)
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            dump("TOOLS", tool_names)

            for cmd in ["EnterPlanMode", "EnterWorktree", "EnterPlanMode", "EnterWorktree"]:
                if cmd in tool_names:
                    try:
                        res = await session.call_tool(cmd, {})
                        dump(cmd, res)
                    except Exception as exc:
                        dump(f"{cmd} exception", repr(exc))
                else:
                    dump(f"skip {cmd}", "not available")

            if "Agent" in tool_names:
                for payload in AGENT_PAYLOADS:
                    try:
                        res = await session.call_tool("Agent", payload)
                        dump(f"Agent {payload['subagent_type']}", res)
                    except Exception as exc:
                        dump(f"Agent {payload['subagent_type']} exception", repr(exc))
            else:
                dump("Agent", "not available")

            for cmd in ["ExitPlanMode", "ExitWorktree"]:
                if cmd in tool_names:
                    try:
                        res = await session.call_tool(cmd, {})
                        dump(cmd, res)
                    except Exception as exc:
                        dump(f"{cmd} exception", repr(exc))


async def main() -> int:
    for variant in VARIANTS:
        try:
            await run_variant(variant)
        except Exception as exc:
            dump(f"VARIANT FAILURE {variant['label']}", repr(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
