"""Hermes-tools-as-MCP server for the codex_app_server runtime.

Codex owns the loop and tool list there, so a curated subset of Hermes tools is
exposed over stdio MCP; codex registers it via ``~/.codex/config.toml
[mcp_servers.hermes-tools]``. Run: ``python -m agent.transports.hermes_tools_mcp_server``.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# JSON Schema type -> Python type mapping for signature generation
_JSON_TO_PY = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


def _signature_from_schema(schema: dict | None) -> tuple[inspect.Signature, dict[str, type]]:
    """KEYWORD_ONLY signature + annotations from a JSON schema (optional params default to None)."""
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    params, annots = [], {}
    for pname, pspec in props.items():
        if pname.startswith("_"):
            continue
        py = _JSON_TO_PY.get((pspec or {}).get("type"), Any)
        ann, default = (py, inspect.Parameter.empty) if pname in required else (Optional[py], None)
        annots[pname] = ann
        params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY, annotation=ann, default=default))
    return inspect.Signature(params, return_annotation=str), annots


# Each name MUST match a registered Hermes tool ``model_tools.handle_function_call()`` can dispatch.
# NOT exposed: terminal/file/search/process/clarify (codex built-ins + its own approval UI);
# delegate_task/memory/session_search/todo (need the running AIAgent context).
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search", "web_extract",
    "browser_navigate", "browser_click", "browser_type", "browser_press", "browser_snapshot", "browser_scroll",
    "browser_back", "browser_get_images", "browser_console", "browser_vision",
    "vision_analyze", "image_generate", "skill_view", "skills_list", "text_to_speech",
    # Kanban handoff tools: stateless (read HERMES_KANBAN_TASK, write kanban.db).
    # Without them a codex-runtime worker can't report completion and hangs.
    "kanban_complete", "kanban_block", "kanban_request_review", "kanban_request_changes", "kanban_comment",
    "kanban_heartbeat", "kanban_show", "kanban_list",
    # Orchestrator-only (the kanban tool gates them on HERMES_KANBAN_TASK unset).
    "kanban_create", "kanban_unblock", "kanban_link", "kanban_unlink",
)


def _build_server() -> Any:
    """Create the MCP server with Hermes tools attached (lazy imports: importable without ``mcp``)."""
    try:
        # mcp 2.0 renamed `mcp.server.fastmcp` to `mcp.server.MCPServer` (same surface).
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(f"hermes-tools MCP server requires the 'mcp' package: {exc}") from exc

    from model_tools import get_tool_definitions, handle_function_call

    mcp = MCPServer(
        "hermes-tools",
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "session. Use these for capabilities Codex's built-in toolset "
            "doesn't cover: web search/extract, browser automation, "
            "subagent delegation, vision, image generation, persistent "
            "memory, skills, and cross-session search."
        ),
    )

    # Authoritative Hermes schemas so MCP clients see the same parameter docs the model does.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    def _make_handler(tool_name: str, schema: dict | None, description: str):
        # The SDK derives the input schema from the callable's signature, so synthesize it from the JSON Schema.
        sig, annots = _signature_from_schema(schema)

        def _dispatch(**kwargs: Any) -> str:
            try:
                # Drop None so unset optionals aren't forwarded to the handler.
                return handle_function_call(tool_name, {k: v for k, v in kwargs.items() if v is not None})
            except Exception as exc:
                logger.exception("tool %s raised", tool_name)
                return json.dumps({"error": str(exc), "tool": tool_name})

        _dispatch.__name__ = tool_name
        _dispatch.__doc__ = description
        _dispatch.__signature__ = sig
        _dispatch.__annotations__ = {**annots, "return": str}
        return _dispatch

    exposed_count = 0
    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug("skipping %s — not registered in this Hermes process", name)
            continue
        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}
        try:
            mcp.add_tool(_make_handler(name, params_schema, description), name=name, description=description)
        except TypeError:
            # Older mcp SDK: decorator-style registration; __signature__ still drives schema.
            mcp.tool(name=name, description=description)(_make_handler(name, params_schema, description))
        exposed_count += 1

    logger.info("hermes-tools MCP server registered %d/%d tools", exposed_count, len(EXPOSED_TOOLS))
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Keep Hermes' own banners off stdout (the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2
    try:
        server.run()  # defaults to stdio transport, which codex spawns us on
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
