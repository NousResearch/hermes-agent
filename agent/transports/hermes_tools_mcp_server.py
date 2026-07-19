"""Hermes-tools-as-MCP server for codex_app_server + claude_cli runtimes.

When the user runs turns through an external coding CLI (codex app-server
or Claude Code via ``claude -p``), that CLI owns the model loop and builds
its own tool list. By default Hermes' richer tool surface is unreachable.

This module exposes a curated subset of Hermes tools to the spawned CLI
subprocess via stdio MCP. Codex registers it in
``~/.codex/config.toml [mcp_servers.hermes-tools]``; Claude CLI receives
an equivalent entry via ``--mcp-config`` (see
``agent.transports.claude_cli.build_hermes_mcp_config``).

Profiles (``HERMES_TOOLS_MCP_PROFILE`` env, default ``codex``):
  - ``codex``  — Hermes-only tools that Codex lacks (no terminal/fs;
                 Codex's built-ins cover those).
  - ``claude`` — Hermes core tools INCLUDING terminal/fs/search, because
                 Claude's native Bash/Edit/Write/Read are disallowed so
                 the model must call Hermes tools only (no double-agent).

Shared exclusions (both profiles):
  - delegate_task / memory / session_search / todo —
    ``_AGENT_LOOP_TOOLS`` in Hermes (model_tools.py). They require the
    running AIAgent mid-loop state; a stateless MCP callback can't drive
    them.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned by: Codex (config.toml) or Claude CLI (``--mcp-config``) as a
            stdio MCP child. The tool round-trip is owned by the CLI;
            Hermes hosts this server and projects stream events.
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
_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _signature_from_schema(schema: dict | None) -> tuple[inspect.Signature, dict[str, type]]:
    """Build a Python function signature and annotations from a JSON schema.

    Args:
        schema: JSON Schema dict with "properties" and "required" keys.

    Returns:
        (signature, annotations_dict) where signature has KEYWORD_ONLY params
        and annotations maps param names to Python types.
    """
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    params, annots = [], {}

    for pname, pspec in props.items():
        if pname.startswith("_"):
            continue
        py = _JSON_TO_PY.get((pspec or {}).get("type"), Any)
        ann, default = (
            (py, inspect.Parameter.empty)
            if pname in required
            else (Optional[py], None)
        )
        annots[pname] = ann
        params.append(
            inspect.Parameter(
                pname, inspect.Parameter.KEYWORD_ONLY, annotation=ann, default=default
            )
        )

    return inspect.Signature(params, return_annotation=str), annots


# MCP server name registered with FastMCP / clients. Claude prefixes tools
# as ``mcp__hermes-tools__<name>``; codex uses the bare server name.
MCP_SERVER_NAME = "hermes-tools"

# Agent-loop tools that need live AIAgent mid-loop state — never expose
# via the stateless MCP callback (either profile).
AGENT_LOOP_TOOLS_EXCLUDED: frozenset[str] = frozenset(
    {
        "delegate_task",
        "memory",
        "session_search",
        "todo",
    }
)

# Hermes-only tools shared by both profiles (codex lacks these; claude
# gets them too once native freelancing is blocked).
_HERMES_SPECIFIC_TOOLS: tuple[str, ...] = (
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
    "text_to_speech",
    # Kanban worker handoff tools — gated on HERMES_KANBAN_TASK env var
    # (set by the kanban dispatcher when spawning a worker). Stateless
    # dispatch — they read the env var and write ~/.hermes/kanban.db.
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # Orchestrator-only kanban tools (gated on HERMES_KANBAN_TASK unset).
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)

# Codex profile: do NOT expose terminal/fs — codex's built-ins cover them
# and route approvals through codex's own UI.
CODEX_EXPOSED_TOOLS: tuple[str, ...] = _HERMES_SPECIFIC_TOOLS

# Claude profile: include Hermes terminal/fs/search so the model can act
# without Claude's native Bash/Edit/Write/Read (those are disallowed by
# the claude_cli runtime). Still excludes _AGENT_LOOP_TOOLS.
CLAUDE_CORE_TOOLS: tuple[str, ...] = (
    "terminal",
    "process",
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "execute_code",
)

CLAUDE_EXPOSED_TOOLS: tuple[str, ...] = CLAUDE_CORE_TOOLS + _HERMES_SPECIFIC_TOOLS

# Back-compat alias — existing codex tests + migration import EXPOSED_TOOLS.
EXPOSED_TOOLS: tuple[str, ...] = CODEX_EXPOSED_TOOLS


def get_exposed_tools(profile: Optional[str] = None) -> tuple[str, ...]:
    """Return the tool name tuple for the given MCP profile.

    ``profile`` defaults to ``HERMES_TOOLS_MCP_PROFILE`` env (``codex`` if
    unset). Unknown profiles fall back to the codex surface.
    """
    resolved = (profile or os.environ.get("HERMES_TOOLS_MCP_PROFILE") or "codex").strip().lower()
    if resolved in {"claude", "claude_cli", "claude-cli"}:
        return CLAUDE_EXPOSED_TOOLS
    return CODEX_EXPOSED_TOOLS


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

    profile = (os.environ.get("HERMES_TOOLS_MCP_PROFILE") or "codex").strip().lower()
    tool_names = get_exposed_tools(profile)

    mcp = FastMCP(
        MCP_SERVER_NAME,
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "or Claude Code session. Prefer these Hermes tools over any "
            "native filesystem/exec tools the host CLI may advertise. "
            "Capabilities include terminal/file (claude profile), web "
            "search/extract, browser automation, vision, image generation, "
            "skills, TTS, and kanban handoff."
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

    for name in tool_names:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "skipping %s — not registered in this Hermes process", name
            )
            continue

        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}

        # FastMCP wants a Python callable. Build a closure that takes the
        # arguments dict, dispatches via handle_function_call, and returns
        # the result string. We use add_tool() for full control over the
        # input schema (FastMCP's @tool() decorator inspects type hints,
        # which we can't get from a JSON schema at runtime).
        def _make_handler(tool_name: str, schema: dict | None):
            sig, annots = _signature_from_schema(schema)

            def _dispatch(**kwargs: Any) -> str:
                try:
                    # Filter out None values before dispatch so unset optionals
                    # aren't forwarded to the handler.
                    args = {k: v for k, v in kwargs.items() if v is not None}
                    return handle_function_call(tool_name, args or {})
                except Exception as exc:
                    logger.exception("tool %s raised", tool_name)
                    return json.dumps({"error": str(exc), "tool": tool_name})

            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = description
            _dispatch.__signature__ = sig
            _dispatch.__annotations__ = {**annots, "return": str}
            return _dispatch

        try:
            mcp.add_tool(
                _make_handler(name, params_schema),
                name=name,
                description=description,
            )
        except TypeError:
            # Older mcp SDK signature — fall back to decorator-style. The
            # synthesized __signature__ on the handler still drives schema
            # generation there.
            handler = _make_handler(name, params_schema)
            handler = mcp.tool(name=name, description=description)(handler)

        exposed_count += 1

    logger.info(
        "hermes-tools MCP server (profile=%s) registered %d/%d tools",
        profile,
        exposed_count,
        len(tool_names),
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
