"""Hermes-tools-as-MCP server for the codex_app_server runtime.

Codex owns the loop and tool list there, so a curated subset of Hermes tools is
exposed over stdio MCP; codex registers it via ``~/.codex/config.toml
[mcp_servers.hermes-tools]``. Run: ``python -m agent.transports.hermes_tools_mcp_server``.

Curation (which tools, per launch profile) lives in
:mod:`agent.transports.hermes_tool_exposure` so this stdio server and the
in-process hybrid bridge can never drift apart. ``memory`` and
``session_search`` — ``_AGENT_LOOP_TOOLS`` the generic dispatcher refuses — are
exposed through the stateless shims in
:mod:`agent.transports.hermes_tools_mcp_server_shims` (#26567), which does not
widen that refusal.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from typing import Any, Callable, Optional

from agent.transports.hermes_tool_exposure import exposed_tools_for_profile
from agent.transports.hermes_tools_mcp_server_shims import stateless_shim_definitions

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

# This launcher's public surface: the default (Codex) profile. A host selects a
# profile at launch (``--profile claude-agent-sdk``); unknown values keep this
# default rather than widening the subprocess. ``exposed_tools_for_profile`` is
# the single source of truth, so the hybrid bridge's grant set
# (``HERMES_TOOLS_LEGACY_NAMES``) and this stdio surface stay identical.
EXPOSED_TOOLS: tuple[str, ...] = exposed_tools_for_profile()


def _signature_from_schema(schema: dict | None) -> tuple[inspect.Signature, dict[str, type]]:
    """KEYWORD_ONLY signature + annotations from a JSON schema (optional params default to None)."""
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
                pname,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=ann,
                default=default,
            )
        )
    return inspect.Signature(params, return_annotation=str), annots


def _make_handler(
    tool_name: str,
    schema: dict | None,
    description: str,
    dispatch: Callable[[dict[str, Any]], str],
) -> Callable[..., str]:
    """MCP handler whose ``__signature__`` mirrors the registry JSON Schema.

    The SDK derives the input schema from the callable's signature (neither the
    decorator nor ``add_tool()`` takes an inputSchema), so it is synthesized
    from the Hermes schema. ``None`` kwargs are dropped so unset optionals are
    not forwarded to the handler.
    """
    signature, annotations = _signature_from_schema(schema)

    def _dispatch(**kwargs: Any) -> str:
        try:
            return dispatch(
                {key: value for key, value in kwargs.items() if value is not None}
            )
        except Exception as exc:
            logger.exception("tool %s raised", tool_name)
            return json.dumps({"error": str(exc), "tool": tool_name})

    _dispatch.__name__ = tool_name
    _dispatch.__doc__ = description
    _dispatch.__signature__ = signature
    _dispatch.__annotations__ = {**annotations, "return": str}
    return _dispatch


def _register_tool(
    mcp: Any,
    *,
    name: str,
    description: str,
    schema: dict | None,
    dispatch: Callable[[dict[str, Any]], str],
) -> None:
    """Register a schema-backed handler on MCP 1.x or 2.x."""
    handler = _make_handler(name, schema, description, dispatch)
    try:
        mcp.add_tool(handler, name=name, description=description)
    except TypeError:
        # Older mcp SDK: decorator-style registration; __signature__ still drives schema.
        mcp.tool(name=name, description=description)(handler)


def _build_server(profile: Optional[str] = None) -> Any:
    """Create the MCP server with the selected curated tool profile.

    Unknown profiles intentionally resolve to the Codex-compatible default via
    ``exposed_tools_for_profile`` rather than widening this subprocess. Imports
    stay lazy so importing this module does not require the optional MCP SDK.
    """
    try:
        # mcp 2.0 removed `mcp.server.fastmcp`; `mcp.server.MCPServer` is the
        # same decorator/add_tool surface under the new name.
        from mcp.server import MCPServer
    except ImportError:
        # mcp 1.x (what `claude-agent-sdk<0.2.140` pins) exports the same
        # surface as `FastMCP`: `MCPServer(name, instructions=)`, `add_tool`,
        # `tool` and `run` are signature-identical on both majors, so an
        # install that resolved the older SDK still gets a working stdio
        # server instead of a dead wrapper. (#65982)
        try:
            from mcp.server import FastMCP as MCPServer
        except ImportError as exc:  # pragma: no cover - install hint
            raise ImportError(
                f"hermes-tools MCP server requires the 'mcp' package: {exc}"
            ) from exc

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
    profile_tools = exposed_tools_for_profile(profile)
    exposed_count = 0
    for name in profile_tools:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug("skipping %s — not registered in this Hermes process", name)
            continue
        _register_tool(
            mcp,
            name=name,
            description=spec.get("description") or f"Hermes {name} tool",
            schema=spec.get("parameters") or {"type": "object", "properties": {}},
            dispatch=lambda kwargs, tool_name=name: handle_function_call(tool_name, kwargs),
        )
        exposed_count += 1

    shim_definitions = stateless_shim_definitions()
    for name, description, schema, dispatch in shim_definitions:
        _register_tool(mcp, name=name, description=description, schema=schema, dispatch=dispatch)

    logger.info(
        "hermes-tools MCP server registered %d/%d tools + %d stateless shims",
        exposed_count,
        len(profile_tools),
        len(shim_definitions),
    )
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    profile = None
    if "--profile" in argv:
        profile_index = argv.index("--profile")
        if profile_index + 1 < len(argv):
            profile = argv[profile_index + 1]
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Keep Hermes' own banners off stdout (the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    # Credentials channel (C4-compliant): read ~/.hermes/.env from DISK inside
    # this child, like every other Hermes entry point (run_agent, cli, main,
    # gateway). The spawn env is the ps-visible --mcp-config argv and stays a
    # minimal non-secret allowlist; without this load, tool check_fns that
    # consult raw os.environ miss .env-stored creds and the tools they gate
    # report unavailable (#65982 R3).
    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv()
    except Exception:
        logger.debug("hermes dotenv load failed", exc_info=True)

    try:
        # Keep the profile-less call shape: test doubles of ``_build_server`` take no arguments.
        server = _build_server(profile=profile) if profile is not None else _build_server()
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
