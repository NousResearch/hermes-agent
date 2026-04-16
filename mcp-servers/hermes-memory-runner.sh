#!/bin/bash
# Runner script for hermes-memory MCP server.
# Uses uv to handle dependency installation automatically.
exec uv run --with "mcp[cli]" "$(dirname "$0")/hermes-memory-mcp.py"
