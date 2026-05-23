"""Hermes-tools-as-MCP server for the codex_app_server runtime.

When the user runs `openai/*` turns through the codex app-server, codex
owns the loop and builds its own tool list. By default, that means
Hermes' richer tool surface — web search, browser automation,
delegate_task subagents, vision analysis, persistent memory, skills,
cross-session search, image generation, TTS — is unreachable.

This module exposes a curated subset of those Hermes tools to the
spawned codex subprocess via stdio MCP. Codex registers it as a normal
MCP server (per `~/.codex/config.toml [mcp_servers.hermes-tools]`) and
the user gets full Hermes capability inside a Codex turn.

Scope (what we expose):
  - web_search, web_extract              — Firecrawl, no codex equivalent
  - browser_navigate / _click / _type /  — Camofox/Browserbase automation
    _snapshot / _scroll / _back / _press /
    _get_images / _console / _vision
  - vision_analyze                       — image inspection by vision model
  - image_generate                       — image generation
  - skill_view, skills_list              — Hermes' skill library
  - text_to_speech                       — TTS
  - kanban_* (complete/block/comment/    — kanban worker + orchestrator
    heartbeat/show/list/create/            handoff (stateless: read env var,
    unblock/link)                          write ~/.hermes/kanban.db)

What we DO NOT expose:
  - terminal / shell                     — codex's own shell tool
  - read_file / write_file / patch       — codex's apply_patch + shell
  - search_files / process               — codex's shell
  - clarify                              — codex's own UX
  - delegate_task / memory /             — `_AGENT_LOOP_TOOLS` in Hermes
    session_search / todo                  (model_tools.py). They require
                                           the running AIAgent context to
                                           dispatch (mid-loop state), so a
                                           stateless MCP callback can't
                                           drive them. See the inline
                                           comment on EXPOSED_TOOLS below.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned by: CodexAppServerSession.ensure_started() when the runtime is
            active and config opts in.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)


def _normalize_tool_args(raw: dict[str, Any]) -> dict[str, Any]:
    """Unwrap legacy MCP calls that nested Hermes args under a ``kwargs`` key.

    FastMCP used to register handlers as ``def _dispatch(**kwargs)``, which made
    clients send ``{"kwargs": {"all_boards": true}}`` instead of flat args.
    """
    if not raw:
        return {}
    if set(raw.keys()) == {"kwargs"} and isinstance(raw.get("kwargs"), dict):
        return dict(raw["kwargs"])
    return dict(raw)


def _json_schema_property_type(prop: dict[str, Any]) -> Any:
    """Map a JSON-schema property to a Python type annotation for FastMCP."""
    enum_values = prop.get("enum")
    if enum_values:
        return Literal[tuple(enum_values)]  # type: ignore[valid-type]

    schema_type = prop.get("type")
    if schema_type == "boolean":
        return bool
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "array":
        items = prop.get("items") or {}
        item_type = _json_schema_property_type(items) if isinstance(items, dict) else Any
        return list[item_type]  # type: ignore[valid-type]
    if schema_type == "object":
        return dict[str, Any]

    # Missing type — treat as string (matches Hermes/OpenAI tool conventions).
    return str


def _build_tool_handler(
    tool_name: str,
    params_schema: dict[str, Any],
    description: str,
    *,
    dispatch: Callable[[str, dict[str, Any]], str],
) -> Callable[..., str]:
    """Build a FastMCP-compatible handler with flat parameters from JSON schema."""
    properties = params_schema.get("properties") or {}
    if not isinstance(properties, dict):
        properties = {}
    required = set(params_schema.get("required") or [])

    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {"return": str}

    for param_name, prop in properties.items():
        if not isinstance(prop, dict):
            prop = {}
        annotation = _json_schema_property_type(prop)
        if param_name not in required:
            annotation = Optional[annotation]
            default = prop.get("default", None)
        else:
            default = inspect.Parameter.empty

        parameters.append(
            inspect.Parameter(
                param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
        annotations[param_name] = annotation

    def _dispatch(**arguments: Any) -> str:
        tool_args = _normalize_tool_args(arguments)
        # Drop unset optional params so Hermes handlers see the same shape as native tools.
        tool_args = {k: v for k, v in tool_args.items() if v is not None}
        try:
            return dispatch(tool_name, tool_args)
        except Exception as exc:
            logger.exception("tool %s raised", tool_name)
            return json.dumps({"error": str(exc), "tool": tool_name})

    _dispatch.__name__ = tool_name
    _dispatch.__doc__ = description
    _dispatch.__annotations__ = annotations
    _dispatch.__signature__ = inspect.Signature(parameters, return_annotation=str)
    return _dispatch


def _register_hermes_tool(
    mcp: Any,
    handler: Callable[..., str],
    *,
    name: str,
    description: str,
) -> None:
    """Register a handler and wrap it so legacy nested ``kwargs`` args still work."""
    from mcp.server.fastmcp.tools.base import Tool

    mcp.add_tool(handler, name=name, description=description)
    registered = mcp._tool_manager.get_tool(name)
    if registered is None:
        return

    class _HermesTool(Tool):
        async def run(
            self,
            arguments: dict[str, Any],
            context: Any = None,
            convert_result: bool = False,
        ) -> Any:
            normalized = _normalize_tool_args(dict(arguments or {}))
            return await super().run(normalized, context=context, convert_result=convert_result)

    wrapped = _HermesTool(
        fn=registered.fn,
        name=registered.name,
        title=registered.title,
        description=registered.description,
        parameters=registered.parameters,
        fn_metadata=registered.fn_metadata,
        is_async=registered.is_async,
        context_kwarg=registered.context_kwarg,
        annotations=registered.annotations,
        icons=registered.icons,
        meta=registered.meta,
    )
    mcp._tool_manager._tools[name] = wrapped


# Tools we expose. Each name MUST match a registered Hermes tool that
# `model_tools.handle_function_call()` can dispatch.
#
# What we deliberately DO NOT expose:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — codex's built-ins cover these and approval routes through
#     codex's own UI.
#   - delegate_task / memory / session_search / todo — these are
#     `_AGENT_LOOP_TOOLS` in Hermes (model_tools.py:493). They require
#     the running AIAgent context to dispatch (mid-loop state), so a
#     stateless MCP callback can't drive them. Hermes' default runtime
#     keeps these working; the codex_app_server runtime cannot.
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_extract",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_snapshot",
    "browser_scroll",
    "browser_back",
    "browser_get_images",
    "browser_console",
    "browser_vision",
    "vision_analyze",
    "image_generate",
    "skill_view",
    "skills_list",
    # Ideas boards (Hermes markdown ideas toolset)
    "ideas_list",
    "ideas_boards",
    "ideas_show",
    "ideas_create",
    "ideas_update",
    "ideas_delete",
    "ideas_convert",
    "text_to_speech",
    # Kanban worker handoff tools — gated on HERMES_KANBAN_TASK env var
    # (set by the kanban dispatcher when spawning a worker). Without these
    # in the callback, a worker spawned with openai_runtime=codex_app_server
    # could do the work but couldn't report completion back to the kernel,
    # making it hang until timeout. Stateless dispatch — they just read
    # the env var and write to ~/.hermes/kanban.db.
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # NOTE: kanban_create / kanban_unblock / kanban_link are orchestrator-
    # only — the kanban tool gates them on HERMES_KANBAN_TASK being unset.
    # They're exposed here for orchestrator agents running on the codex
    # runtime that need to dispatch new tasks.
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)


def _build_server() -> Any:
    """Create the FastMCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            f"hermes-tools MCP server requires the 'mcp' package: {exc}"
        ) from exc

    # Discover Hermes tools so dispatch works.
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
    )

    mcp = FastMCP(
        "hermes-tools",
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "session. Use these for capabilities Codex's built-in toolset "
            "doesn't cover: web search/extract, browser automation, "
            "subagent delegation, vision, image generation, persistent "
            "memory, skills, and cross-session search."
        ),
    )

    # Pull authoritative Hermes tool schemas for the ones we expose, so
    # MCP clients see the same parameter docs Hermes gives the model.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    exposed_count = 0

    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "skipping %s — not registered in this Hermes process", name
            )
            continue

        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}

        # FastMCP derives MCP inputSchema from the handler signature. A bare
        # ``**kwargs`` handler exposes a single required ``kwargs`` object, which
        # breaks clients (they send nested args that never reach Hermes). Build a
        # function with one keyword-only parameter per JSON-schema property instead.
        handler = _build_tool_handler(
            name,
            params_schema,
            description,
            dispatch=handle_function_call,
        )

        try:
            _register_hermes_tool(mcp, handler, name=name, description=description)
        except TypeError:
            # Older mcp SDK signature — fall back to decorator-style.
            handler = mcp.tool(name=name, description=description)(handler)
            try:
                _register_hermes_tool(mcp, handler, name=name, description=description)
            except Exception:
                pass

        exposed_count += 1

    logger.info(
        "hermes-tools MCP server registered %d/%d tools",
        exposed_count,
        len(EXPOSED_TOOLS),
    )
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quiet mode: keep Hermes' own banners off stdout (which is the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2

    # FastMCP runs with stdio transport by default when launched as a
    # subprocess.
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
