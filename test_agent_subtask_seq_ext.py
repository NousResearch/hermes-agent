#!/usr/bin/env python3
"""Extended sequence test: EnterPlanMode -> EnterWorktree -> Agent as SubTask.

This probes whether Claude MCP requires specific session state to resolve Agent
subagent types correctly.
"""
import asyncio
import json
import os
import logging

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLAUDE = os.environ.get("CLAUDE_CLI_PATH", "/Users/tusker/.local/bin/claude")
ENV_KEYS = {"HOME", "USER", "PATH", "TERM", "SHELL", "LANG", "LC_ALL", "TMPDIR"}

def dump_result(label, result):
    logger.info(f"\n=== {label} ===")
    logger.info(json.dumps(result.model_dump(mode="json"), indent=2)[:40000])

async def safe_call(session, tool_name, payload=None):
    payload = payload or {}
    logger.info(f"Calling tool: {tool_name} with payload: {payload}")
    result = await session.call_tool(tool_name, payload)
    dump_result(tool_name, result)
    return result

async def main() -> int:
    env = {k: v for k, v in os.environ.items() if k in ENV_KEYS}

    # Optional: try with CLAUDE_CODE_FORK_SUBAGENT=1
    if os.getenv("CLAUDE_CODE_FORK_SUBAGENT", "0") == "1":
        env["CLAUDE_CODE_FORK_SUBAGENT"] = "1"
        logger.info("CLAUDE_CODE_FORK_SUBAGENT=1 set for this run.")

    server = StdioServerParameters(command=CLAUDE, args=["mcp", "serve"], env=env)

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            dump_result("INIT", init)
            
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            logger.info(f"TOOLS: {names}")

            # Sequence: EnterPlanMode -> EnterWorktree -> EnterPlanMode (again)
            if "EnterPlanMode" in names:
                try:
                    await safe_call(session, "EnterPlanMode", {})
                except Exception as exc:
                    logger.error(f"EnterPlanMode exception: {exc!r}")

            if "EnterWorktree" in names:
                try:
                    await safe_call(session, "EnterWorktree", {})
                except Exception as exc:
                    logger.error(f"EnterWorktree exception: {exc!r}")
            
            # Repeat EnterPlanMode after worktree
            if "EnterPlanMode" in names:
                try:
                    await safe_call(session, "EnterPlanMode", {})
                except Exception as exc:
                    logger.error(f"EnterPlanMode (re-entry) exception: {exc!r}")

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
                        logger.error(f"Agent {payload['subagent_type']} exception: {exc!r}")
            else:
                logger.warning("Agent tool not available.")

            if "ExitPlanMode" in names:
                try:
                    await safe_call(session, "ExitPlanMode", {})
                except Exception as exc:
                    logger.error(f"ExitPlanMode exception: {exc!r}")

            if "ExitWorktree" in names:
                try:
                    await safe_call(session, "ExitWorktree", {})
                except Exception as exc:
                    logger.error(f"ExitWorktree exception: {exc!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
