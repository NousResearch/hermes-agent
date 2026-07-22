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
from importlib.metadata import version as distribution_version
from typing import Any, Optional
from urllib.parse import urlsplit

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


class _FastMCP126SchemaAdapter:
    """Install authoritative schemas on FastMCP 1.26's internal Tools.

    FastMCP 1.26.0 has no public API for registering a callable with an
    independent JSON Schema. Keep the unavoidable private access here and fail
    closed if either the exact reviewed SDK version or its internal shape
    changes. Calls are validated with the same schema that ``tools/list``
    advertises; Pydantic only bridges the validated flat object to the existing
    ``**kwargs`` Hermes dispatcher.
    """

    _SUPPORTED_VERSION = "1.26.0"

    def __init__(self, server: Any) -> None:
        installed = distribution_version("mcp")
        if installed != self._SUPPORTED_VERSION:
            raise RuntimeError(
                "Hermes authoritative MCP schema adapter requires "
                f"mcp=={self._SUPPORTED_VERSION}; found {installed}. "
                "Re-review FastMCP's explicit-schema API and Tool internals "
                "before changing the project pin."
            )

        manager = getattr(server, "_tool_manager", None)
        get_tool = getattr(manager, "get_tool", None)
        if manager is None or not callable(get_tool):
            raise RuntimeError(
                "mcp==1.26.0 compatibility failure: expected "
                "FastMCP._tool_manager.get_tool"
            )
        self._get_tool = get_tool

    def install(self, tool_name: str, schema: dict[str, Any]) -> None:
        """Advertise and enforce ``schema`` on one registered FastMCP Tool."""
        from jsonschema import FormatChecker, validators
        from pydantic import ConfigDict, model_validator

        # Private import is deliberately confined to this pinned-version adapter.
        from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase

        registered_tool: Any = self._get_tool(tool_name)
        metadata = getattr(registered_tool, "fn_metadata", None)
        if (
            registered_tool is None
            or not isinstance(getattr(registered_tool, "parameters", None), dict)
            or metadata is None
            or not hasattr(metadata, "arg_model")
        ):
            raise RuntimeError(
                "mcp==1.26.0 compatibility failure: registered Tool must expose "
                "mutable parameters and fn_metadata.arg_model "
                f"({tool_name!r})"
            )

        validator_type = validators.validator_for(schema)
        validator_type.check_schema(schema)
        format_checker = FormatChecker()

        # jsonschema's RFC URI checker is an optional extra and is absent from
        # mcp==1.26.0's dependency set. Register the minimum protocol-relevant
        # check explicitly so advertised ``format: uri`` constraints are not
        # silently ignored on a standard Hermes installation.
        @format_checker.checks("uri")
        def _is_uri(value: object) -> bool:
            if not isinstance(value, str):
                return True
            parsed = urlsplit(value)
            return bool(parsed.scheme) and not any(char.isspace() for char in value)

        validator = validator_type(schema, format_checker=format_checker)

        class _ValidatedHermesArguments(ArgModelBase):
            model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

            @model_validator(mode="before")
            @classmethod
            def _validate_authoritative_schema(cls, value: Any) -> Any:
                error = next(validator.iter_errors(value), None)
                if error is None:
                    return value
                path = "$" + "".join(
                    f"[{part}]" if isinstance(part, int) else f".{part}"
                    for part in error.absolute_path
                )
                raise ValueError(
                    f"{tool_name} arguments fail JSON Schema at {path}: {error.message}"
                )

            def model_dump_one_level(self) -> dict[str, Any]:
                return dict(self.__pydantic_extra__ or {})

        try:
            registered_tool.parameters = schema
            metadata.arg_model = _ValidatedHermesArguments
        except Exception as exc:
            raise RuntimeError(
                "mcp==1.26.0 compatibility failure: could not install Tool "
                f"schema/argument model ({tool_name!r})"
            ) from exc
        if (
            registered_tool.parameters != schema
            or metadata.arg_model is not _ValidatedHermesArguments
        ):
            raise RuntimeError(
                "mcp==1.26.0 compatibility failure: Tool schema/argument model "
                f"mutation did not stick ({tool_name!r})"
            )


def _build_server() -> Any:
    """Create the FastMCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import CallToolResult, TextContent
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

    def _convert_result(result: Any) -> Any:
        """Translate Hermes' JSON-string convention to MCP result semantics."""
        if isinstance(result, CallToolResult):
            return result

        decoded = result
        if isinstance(result, str):
            try:
                decoded = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return result

        if isinstance(decoded, dict):
            # structuredContent must be JSON-native. Normalize native Hermes
            # values (for example datetimes) while preserving Unicode.
            normalized = json.loads(
                json.dumps(decoded, ensure_ascii=False, default=str)
            )
            text = (
                result
                if isinstance(result, str)
                else json.dumps(normalized, ensure_ascii=False)
            )
            return CallToolResult(
                content=[TextContent(type="text", text=text)],
                structuredContent=normalized,
                isError="error" in decoded,
            )

        return result

    # Pull authoritative Hermes tool schemas for the ones we expose, so
    # MCP clients see the same parameter docs Hermes gives the model.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }
    schema_adapter = _FastMCP126SchemaAdapter(mcp)

    exposed_count = 0

    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug("skipping %s — not registered in this Hermes process", name)
            continue

        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters")
        if not isinstance(params_schema, dict):
            raise RuntimeError(
                f"Hermes tool {name!r} has no authoritative object JSON Schema"
            )

        # FastMCP wants a Python callable. Build a closure that takes the
        # arguments dict and dispatches through Hermes. The SDK initially
        # inspects this callable, then we replace that synthetic argument model
        # and schema with validating pass-through plumbing plus Hermes'
        # authoritative JSON Schema. This avoids lossy
        # JSON-Schema-to-Python-signature translation.
        def _make_handler(tool_name: str, tool_description: str) -> Any:
            def _dispatch(**kwargs: Any) -> Any:
                try:
                    result = handle_function_call(tool_name, kwargs or {})
                    return _convert_result(result)
                except Exception as exc:
                    logger.exception("tool %s raised", tool_name)
                    return _convert_result({"error": str(exc), "tool": tool_name})

            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = tool_description
            return _dispatch

        mcp.add_tool(
            _make_handler(name, description),
            name=name,
            description=description,
        )

        schema_adapter.install(name, params_schema)

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
    except (ImportError, RuntimeError) as exc:
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
