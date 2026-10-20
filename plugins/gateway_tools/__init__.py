"""Gateway advertised-tools plugin — default consumer for the remote MCP lane.

The browser lane routes the browser verbs onto the remote gateway's MCP tool
surface. Everything ELSE the gateway advertises (extract, crawl, usage,
instances, cookies, ...) had no consumer: each tool needed bespoke wiring,
so capabilities shipped server-side and sat unused client-side.

This plugin closes that by consuming the advertised set wholesale:

* one ``tools/list`` at plugin load (stateless, same transport as the verbs)
* every advertised tool that is NOT already routed natively (the ``browser_``
  verbs) registers as a first-class agent tool, schema carried verbatim from
  the server
* each handler is one stateless ``tools/call`` through the same transport, so
  new server-side tools appear on the next session load with zero wiring

No brand, no per-tool code: the server describes its surface, we consume it.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_DISCOVERY_TIMEOUT = 45  # seconds; first call after an idle cold-boot can take ~60s
_EXCLUDE_PREFIX = "browser_"  # routed natively by the browser tool family


def _cache_path() -> str:
    # PLUGIN_DATA overrides (tests, plugin hosts that provide a data dir); default
    # is a user-shared path so one profile's discovery seeds every sibling load —
    # profile sessions run under one home, and the advertised set is identical.
    base = os.environ.get("PLUGIN_DATA") or os.path.join(
        os.path.expanduser("~/.hermes/cache"), "gateway_tools")
    return os.path.join(base, "advertised_tools.json")


def _cached_tools() -> list[dict]:
    try:
        with open(_cache_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("tools", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _write_cache(tools: list[dict]) -> None:
    try:
        with open(_cache_path(), "w", encoding="utf-8") as fh:
            json.dump({"tools": tools}, fh)
    except Exception:
        pass  # cache is best-effort; discovery failure next load degrades to no tools


def _discover() -> list[dict]:
    """Advertised tools: disk cache first (instant load), live tools/list only on
    a cold start, cache write-through on success.

    # ponytail: cache-first because the remote gateway cold-boots from an idle
    # drain (first request can take ~60s) while the plugin loader's own load
    # window is seconds — a blocking discovery there registers nothing. The
    # cache is shared across profile sessions (one home), so one profile's
    # successful discovery seeds every other. No background refresh: a new
    # server-side tool appears after the first cache-less load. Add a refresh
    # tool if stale schemas ever bite.
    """
    from tools.browser_mcp_transport import _is_gateway_backend, tools_list

    if not _is_gateway_backend():
        return []  # local/community REST backend: no MCP surface to consume
    cached = _cached_tools()
    if cached:
        return cached
    try:
        tools = tools_list(timeout=_DISCOVERY_TIMEOUT)
        if tools:
            _write_cache(tools)
            return tools
    except Exception as exc:
        logger.warning("gateway advertised-tools discovery failed (%s); using cache if any", exc)
    return _cached_tools()


def _normalize_schema(input_schema: dict) -> dict:
    """Advertised inputSchema -> the registry's {"type","parameters"} envelope."""
    params = input_schema if isinstance(input_schema, dict) else {}
    if params.get("type") != "object":
        params = {"type": "object", **params}
    return {"type": "object", "parameters": params}


def register(ctx) -> None:
    from tools.browser_mcp_transport import _call, _result_text
    from tools.registry import tool_error

    advertised = _discover()
    if not advertised:
        print("[gateway_tools] no advertised tools: no remote MCP gateway configured, "
              "or discovery failed with no cache")
        return

    registered = 0
    for tool in advertised:
        name = tool.get("name") or ""
        if not name or name.startswith(_EXCLUDE_PREFIX):
            continue  # browser verbs already route natively — no duplicates
        schema = _normalize_schema(tool.get("inputSchema") or {})
        param_keys = set((schema.get("parameters") or {}).get("properties", {}))

        def _handler(args=None, _name=name, _keys=param_keys, **kwargs):
            # Registry dispatch contract: args dict + task_id-style extras merged.
            # Filter to the advertised parameter set so dispatch extras never
            # cross the wire as unknown tool arguments.
            merged = {**(args or {}), **kwargs}
            arguments = {k: v for k, v in merged.items() if k in _keys}
            try:
                result = _call(_name, arguments)
                if result.get("isError") is True:
                    return tool_error(_result_text(result) or "gateway tool error", success=False)
                return _result_text(result)
            except Exception as exc:
                return tool_error(f"gateway tool '{_name}' failed: {exc}", success=False)

        try:
            ctx.register_tool(
                name=name,
                toolset="gateway",
                schema=schema,
                handler=_handler,
                check_fn=lambda: True,
                description=tool.get("description") or f"Remote gateway tool {name}.",
            )
            registered += 1
        except Exception as exc:
            logger.warning("registering gateway tool '%s' failed: %s", name, exc)

    print(f"[gateway_tools] registered {registered} advertised tool(s) from the remote MCP gateway")