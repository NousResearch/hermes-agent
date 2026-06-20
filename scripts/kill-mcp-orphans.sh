#!/bin/bash
# Kill orphaned stdio MCP subprocesses from previous gateway runs
# Only targets transcriptor wrapper and open-design MCP stdio clients (not the daemon)

pkill -f 'transcriptor-mcp-wrapper.sh' 2>/dev/null || true
pkill -f 'open-design.*cli.js.*mcp.*daemon-url' 2>/dev/null || true

exit 0
