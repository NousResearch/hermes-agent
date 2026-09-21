#!/usr/bin/env python3
"""
Rate limited executor for cron jobs.

Usage: rate_limited_executor.py --key <key> --rpm <rpm> -- <command...>

This script enforces rate limiting for cron jobs that hit external APIs.
It works by:
1. Using the adaptive-rate-limiter MCP server to await a slot
2. Running the command
3. Recording the response (if the command outputs JSON headers to stderr)
4. Exiting with the command's exit code

The command can optionally output JSON lines to stderr in the format:
{"status": <int>, "headers": {"Header-Name": "value", ...}}

If the command doesn't output headers, only proactive rate limiting is applied.
"""

import asyncio
import json
import os
import sys
import subprocess
import time
from typing import Dict, Any, Optional

# We'll need to connect to the MCP server. Since we're in the same environment,
# we can import the mcp client directly.

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("ERROR: MCP client not available", file=sys.stderr)
    sys.exit(1)


async def main():
    if len(sys.argv) < 5 or sys.argv[-3] != "--":
        print(__doc__)
        sys.exit(1)

    # Parse arguments
    key = None
    rpm = None
    cmd_start = None
    for i, arg in enumerate(sys.argv[1:]):  # Skip script name
        if arg == "--key":
            key = sys.argv[i + 2]
        elif arg == "--rpm":
            rpm = float(sys.argv[i + 2])
        elif arg == "--":
            cmd_start = i + 1  # Index in sys.argv
            break

    if key is None or rpm is None or cmd_start is None:
        print("ERROR: Missing required arguments", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[cmd_start:]

    # Connect to the MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["/home/kensei/.hermes/mcp/adaptive_rate_limiter/server.py"],
        env={
            **os.environ,
            "RATE_LIMIT_DEFAULT_RPM": str(rpm),  # Override default for this key
        },
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Wait for a slot
                await_slot_result = await session.call_tool(
                    "rate_limiter_await_slot",
                    {"key": key}
                )
                # The result is a list of TextContent; we'll parse the first one.
                import ast
                slot_data = ast.literal_eval(await_slot_result[0].text)
                waited = slot_data.get("waited", 0.0)
                if waited > 0.01:  # Only log if we waited significantly
                    print(f"[rate-limiter] waited {waited:.2f}s for key '{key}'", file=sys.stderr)

                # Run the command
                start_time = time.monotonic()
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                elapsed = time.monotonic() - start_time

                # Try to parse JSON headers from stderr (each line)
                status = None
                headers = {}
                for line in stderr.decode().splitlines():
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            data = json.loads(line)
                            if "status" in data and "headers" in data:
                                status = data["status"]
                                headers = data["headers"]
                                break
                        except json.JSONDecodeError:
                            pass

                # Record the response if we got status/headers
                if status is not None:
                    await session.call_tool(
                        "rate_limiter_record_response",
                        {
                            "key": key,
                            "status": status,
                            "headers": headers,
                        }
                    )

                # Output the command's stdout/stderr
                sys.stdout.buffer.write(stdout)
                sys.stderr.buffer.write(stderr)

                # Exit with the command's exit code
                sys.exit(proc.returncode)

    except Exception as e:
        print(f"ERROR: Rate limiter failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())