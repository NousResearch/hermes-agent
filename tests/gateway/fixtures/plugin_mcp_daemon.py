"""Test-only: the ordinary daemon with a fake MCP server in its own registry (no MCP SDK, no network).

``probe-mcp`` answers every tool with the call and the pid of the process holding the registry, so a
test sees that a plugin's ``ctx.call_mcp`` reached THIS gateway's MCP registry.
"""
import asyncio
import json
import os
import runpy
from types import SimpleNamespace

from tools import mcp_tool
from tools.mcp_tool_loop import _ensure_mcp_loop


class _Session:
    async def call_tool(self, name, arguments=None):
        text = json.dumps({'tool': name, 'arguments': arguments, 'registry_pid': os.getpid()})
        return SimpleNamespace(is_error=False, content=[SimpleNamespace(type='text', text=text)],
                               structured_content=None, meta=None)


class _Server:
    def __init__(self):
        self.session = _Session()
        self._rpc_lock = asyncio.Lock()


_ensure_mcp_loop()
mcp_tool._servers['probe-mcp'] = _Server()
runpy.run_module('gateway.run', run_name='__main__')
