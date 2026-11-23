"""Serve one Python Hermes plugin's tools over MCP stdio for the native desktop.

Run with the installed Hermes interpreter: ``python -I bridge.py <hermes-agent root> <plugin dir>``.
The plugin is loaded by Hermes's own PluginManager (so its register(ctx), config and plugin-host
isolation behave as in the Python agent) and the tools it registered are listed and dispatched
through tools.registry. stdout carries the protocol; anything a plugin prints goes to stderr.
"""

import json
import os
import sys


def main() -> int:
    root, plugin_dir = sys.argv[1], os.path.realpath(sys.argv[2])
    wire = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.path.insert(0, root)
    import hermes_bootstrap  # noqa: F401  -- the launcher's own environment setup (site dirs, home pin)

    from pathlib import Path

    from hermes_cli.plugins import get_plugin_manager
    from hermes_cli.plugins_discovery import scan_directory
    from hermes_cli.plugins_manifest import manifest_key
    from tools.registry import registry

    manager = get_plugin_manager()
    manifests = [m for m in scan_directory(Path(plugin_dir).parent, "user")
                 if m.path and os.path.realpath(m.path) == plugin_dir]
    if not manifests:
        print(f"No plugin.yaml in {plugin_dir}", file=sys.stderr)
        return 2
    manager._load_plugin(manifests[0])
    loaded = manager._plugins.get(manifest_key(manifests[0]))
    if loaded is None or loaded.error:
        print(f"The plugin did not load: {getattr(loaded, 'error', 'unknown error')}", file=sys.stderr)
        return 3
    names = list(loaded.tools_registered)

    def tool(name):
        function = (registry.definition_for(name) or {}).get("function", {})
        return {"name": name, "description": function.get("description", ""),
                "inputSchema": function.get("parameters") or {"type": "object", "properties": {}}}

    def answer(request, result=None, error=None):
        if "id" not in request:
            return
        reply = {"jsonrpc": "2.0", "id": request["id"]}
        reply.update({"error": {"code": -32000, "message": error}} if error else {"result": result})
        wire.write(json.dumps(reply) + "\n")

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        method, params = request.get("method"), request.get("params") or {}
        if method == "initialize":
            answer(request, {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                             "capabilities": {"tools": {}},
                             "serverInfo": {"name": manifests[0].name, "version": "1"}})
        elif method == "tools/list":
            answer(request, {"tools": [tool(n) for n in names]})
        elif method == "tools/call":
            name = params.get("name")
            if name not in names:
                answer(request, error=f"Unknown tool: {name}")
                continue
            result = registry.dispatch(name, params.get("arguments") or {})
            text = result if isinstance(result, str) else json.dumps(result)
            failed = isinstance(result, dict) and "error" in result
            answer(request, {"content": [{"type": "text", "text": text}], "isError": failed})
        elif method == "ping":
            answer(request, {})
        elif "id" in request:
            answer(request, error=f"Unsupported method: {method}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
