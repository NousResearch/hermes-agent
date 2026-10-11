#!/usr/bin/env python3
"""Example ACP client that connects to hermes-acp as an agent server.

Usage:
    uv run python acp_client_example.py "Your prompt here"

This demonstrates how to use the ACP Python client library to connect
to the Hermes ACP server and send prompts.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure hermes-agent root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import acp
from acp import Client


async def main():
    if len(sys.argv) < 2:
        print("Usage: python acp_client_example.py <prompt>", file=sys.stderr)
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])

    # Connect to hermes-acp as an ACP client
    # The server command is launched as a subprocess
    server_cmd = ["uv", "run", "hermes-acp"]
    server_cwd = str(Path(__file__).resolve().parent)

    print(f"Connecting to Hermes ACP server...", file=sys.stderr)
    print(f"Prompt: {prompt}", file=sys.stderr)

    # Use the ACP Client to connect to the hermes-acp server
    client = Client(
        command=server_cmd[0],
        args=server_cmd[1:],
        cwd=server_cwd,
    )

    try:
        # Initialize the connection
        init_response = client.initialize(
            protocol_version=1,
            capabilities={},
            client_info={"name": "hermes-acp-client", "version": "1.0"},
        )
        print(f"Initialized: {init_response}", file=sys.stderr)

        # Create a new session
        session = client.create_session()
        print(f"Session created: {session.id}", file=sys.stderr)

        # Send a prompt and stream the response
        print(f"\n--- Hermes Response ---\n", file=sys.stderr)

        for chunk in session.prompt(message=prompt):
            # Chunk types: text, thinking, tool_call, tool_result, etc.
            if hasattr(chunk, 'text') and chunk.text:
                print(chunk.text, end="", flush=True)
            elif hasattr(chunk, 'thinking') and chunk.thinking:
                print(f"\n[Thinking: {chunk.thinking[:100]}...]", file=sys.stderr)
            elif hasattr(chunk, 'tool_call') and chunk.tool_call:
                print(f"\n[Tool: {chunk.tool_call.name}]", file=sys.stderr)

        print("\n\n--- End Response ---", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
