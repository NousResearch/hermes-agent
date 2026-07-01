"""Shared curation/mapping layer for exposing Hermes tools to an external
agent runtime as MCP tools.

Two Hermes runtimes hand control of the agent loop to an external engine and
must re-expose Hermes' own tools to it as MCP tools:

* the **codex_app_server** runtime — a separate `python -m
  agent.transports.hermes_tools_mcp_server` subprocess speaking **stdio** MCP,
  dispatching **statelessly** via ``model_tools.handle_function_call``; and
* the **Claude Agent SDK** runtime — an **in-process** MCP server
  (``create_sdk_mcp_server``) whose handlers dispatch **statefully** via
  ``agent_runtime_helpers.invoke_tool`` (so agent-level tools work).

Those two backends legitimately differ on *execution* (stateless vs live
agent), *registration* (FastMCP/stdio vs SDK/in-process) and *result shape*.
But they were independently answering the same three questions — **which
tools, mapped how, wrapped/erred how** — which is duplicated logic and a
drift risk. This module is the single source of truth for exactly those, so
neither execution model is forced on the other.

Deliberately **import-light**: no FastMCP, no ``claude_agent_sdk``, no
``AIAgent`` imports at module scope, so both a stdio subprocess and the
in-process backend can import it without dragging in the other's (often
heavy/optional) dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# Tools that are safe to dispatch through the STATELESS
# ``model_tools.handle_function_call`` path — i.e. everything except the four
# agent-level tools (todo / memory / delegate_task / session_search) that need
# a live ``AIAgent``. The stdio codex backend can only expose these; the
# in-process Claude SDK backend can expose a broader live set because it
# reaches the live agent via ``invoke_tool``.
#
# What we deliberately exclude and why:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — the external engine (codex) has its own built-ins for these.
#   - todo / memory / delegate_task / session_search — ``_AGENT_LOOP_TOOLS``;
#     they require live mid-loop agent state, which a stateless/out-of-process
#     dispatcher cannot provide.
CURATED_STATELESS_TOOLS: Tuple[str, ...] = (
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
    # Kanban worker-handoff tools — stateless (read HERMES_KANBAN_TASK env var,
    # write ~/.hermes/kanban.db). Without these a worker spawned under an
    # external runtime could do the work but not report completion.
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # Orchestrator-only kanban tools (the kanban tool gates them on
    # HERMES_KANBAN_TASK being unset).
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)

_MCP_PREFIX = "mcp__"


def normalize_tool_spec(spec: Any) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Return ``(registry_name, description, json_schema)`` from a tool spec.

    Accepts both formats a Hermes tool schema can arrive in:

    * OpenAI: ``{"type": "function", "function": {name, description, parameters}}``
      (from ``agent.tools`` / ``get_tool_definitions``)
    * Anthropic: ``{name, description, input_schema}``
      (from an Anthropic-shaped ``api_kwargs["tools"]``)

    Strips a leading ``mcp__`` prefix that the OAuth wire adds to tool names,
    so the name matches the registry. Returns ``None`` if the spec is unusable.
    Missing schemas default to an empty object schema.
    """
    if not isinstance(spec, dict):
        return None
    if spec.get("type") == "function" and isinstance(spec.get("function"), dict):
        fn = spec["function"]
        name = fn.get("name")
        description = fn.get("description") or name
        schema = fn.get("parameters") or {"type": "object", "properties": {}}
    else:
        name = spec.get("name")
        description = spec.get("description") or name
        schema = (
            spec.get("input_schema")
            or spec.get("parameters")
            or {"type": "object", "properties": {}}
        )
    if not name:
        return None
    if name.startswith(_MCP_PREFIX):
        name = name[len(_MCP_PREFIX):]
    return name, description, schema


def resolve_curated_specs(
    tool_defs: Optional[List[dict]],
    names: Tuple[str, ...] = CURATED_STATELESS_TOOLS,
) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """Resolve ``names`` against a list of tool definitions.

    ``tool_defs`` is what ``model_tools.get_tool_definitions()`` returns
    (OpenAI-format). Kept as a parameter — rather than importing
    ``get_tool_definitions`` here — so this module stays import-light. Returns
    ``{name: (description, json_schema)}`` for the requested names that are
    actually registered, in the order given by ``names``.
    """
    by_name: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for td in tool_defs or []:
        norm = normalize_tool_spec(td)
        if norm is not None:
            n, desc, schema = norm
            by_name[n] = (desc, schema)
    return {n: by_name[n] for n in names if n in by_name}


def looks_like_tool_error(text: Any) -> bool:
    """Detect Hermes' single-key ``{"error": "..."}`` error envelope.

    Paired with :func:`make_error_envelope` so the producer and detector agree
    on the shape (both single-key), avoiding the historical 2-key vs 1-key
    mismatch between the codex and SDK backends.
    """
    if not isinstance(text, str):
        return False
    try:
        parsed = json.loads(text)
    except Exception:
        return False
    return isinstance(parsed, dict) and "error" in parsed and len(parsed) == 1


def make_error_envelope(exc: Any, tool: Optional[str] = None) -> str:
    """Produce the canonical single-key ``{"error": ...}`` envelope.

    The tool name (when given) is folded into the message rather than added as
    a second key, so :func:`looks_like_tool_error` still matches it.
    """
    msg = f"{tool}: {exc}" if tool else str(exc)
    return json.dumps({"error": msg}, ensure_ascii=False)


def wrap_untrusted(name: str, content: Any) -> Any:
    """Wrap an untrusted-tool result in ``<untrusted_tool_result>`` delimiters.

    Thin lazy pass-through to ``agent.tool_dispatch_helpers._maybe_wrap_untrusted``
    (the native path's promptware defense) so "how a result is wrapped" has a
    single definition. Imported lazily to keep this module import-light.
    """
    from agent.tool_dispatch_helpers import _maybe_wrap_untrusted

    return _maybe_wrap_untrusted(name, content)
