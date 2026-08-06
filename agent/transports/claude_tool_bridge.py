"""In-process MCP bridge from the Claude Agent SDK back into Hermes tools.

When ``api_mode="claude_agent_sdk"`` is active the SDK owns the agent loop,
so Claude's tool calls arrive as MCP invocations instead of OpenAI-shaped
``tool_calls``.  Hermes still owns tool execution: every generated handler
here funnels into :func:`agent.tool_executor.execute_one_tool`, which is the
one place that applies approvals, guardrails, plugin hooks, checkpoints,
progress callbacks, and task/environment routing.  A handler must never call
``model_tools.handle_function_call`` directly — that skips all of it.

The server is built from the schemas the bound ``AIAgent`` already has
enabled (``agent.tools``), so the bridge cannot widen the model's tool
surface beyond what the agent's toolset resolution produced.

``claude_agent_sdk`` is an optional extra and is imported lazily inside
:func:`build_hermes_sdk_mcp_server` — this module imports cleanly without it.

Unrelated to ``hermes_tools_mcp_server``: that one is a stateless stdio
subset for the codex runtime and deliberately excludes the AIAgent-bound
tools.  This bridge is bound to a live agent and exposes the full enabled
set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Optional

from agent.display import _detect_tool_failure
from agent.tool_dispatch_helpers import (
    _is_multimodal_tool_result,
    _multimodal_text_summary,
)
from agent.tool_executor import execute_one_tool, finalize_tool_outcome
from tools.thread_context import propagate_context_to_thread

logger = logging.getLogger(__name__)

MCP_SERVER_NAME = "hermes"
MCP_SERVER_VERSION = "1.0.0"
# The SDK namespaces in-process servers as ``mcp__<server>__<tool>``; PR4 has
# to list every one of these in ``allowed_tools`` or Claude cannot call them.
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

_DATA_URL_PREFIX = "data:"


def mcp_tool_name(tool_name: str) -> str:
    """Return the name Claude sees for a Hermes tool."""
    return f"{MCP_TOOL_PREFIX}{tool_name}"


def agent_tool_specs(agent) -> list[dict[str, Any]]:
    """Return ``{name, description, parameters}`` for the agent's enabled tools.

    Reads ``agent.tools`` — the already-resolved, toolset-filtered definitions
    the agent sends on every API call. Nothing else is exposed.
    """
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in getattr(agent, "tools", None) or []:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        specs.append(
            {
                "name": name,
                "description": function.get("description") or f"Hermes {name} tool",
                "parameters": function.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return specs


def is_read_only_tool(tool_name: str) -> bool:
    """Whether Claude may batch *tool_name* in parallel with other reads.

    Hermes has no per-tool read-only flag on the registry; the closest
    authority is ``_PARALLEL_SAFE_TOOLS`` — the conservative allowlist of
    read-only tools with no shared mutable session state that the batch
    segment planner already trusts to run concurrently.  Anything not on it
    (including MCP tools whose server merely opted into parallel calls)
    defaults to False so a mutating tool is never announced as read-only.
    """
    try:
        from agent.tool_dispatch_helpers import _PARALLEL_SAFE_TOOLS

        return tool_name in _PARALLEL_SAFE_TOOLS
    except Exception:
        return False


def _image_block(data: str, mime_type: str) -> dict[str, Any]:
    return {"type": "image", "data": data, "mimeType": mime_type}


def _image_block_from_url(url: str) -> Optional[dict[str, Any]]:
    """Convert an OpenAI ``image_url`` payload into an MCP image block.

    MCP carries raw base64 with a separate ``mimeType``; the ``data:`` prefix
    an OpenAI content part uses must be stripped, not forwarded.
    """
    if not isinstance(url, str) or not url.startswith(_DATA_URL_PREFIX):
        return None
    header, _, payload = url[len(_DATA_URL_PREFIX):].partition(",")
    if not payload:
        return None
    mime_type = header.split(";", 1)[0] or "image/png"
    return _image_block(payload, mime_type)


def tool_result_content_blocks(result: Any) -> list[dict[str, Any]]:
    """Convert a Hermes tool result into MCP content blocks.

    Registry handlers return a JSON string; multimodal tools return the
    ``{"_multimodal": True, "content": [...]}`` envelope whose image parts
    become MCP image blocks.
    """
    if _is_multimodal_tool_result(result):
        blocks: list[dict[str, Any]] = []
        for part in result.get("content") or []:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                blocks.append({"type": "text", "text": str(part.get("text", ""))})
            elif part_type == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                block = _image_block_from_url(url)
                if block is not None:
                    blocks.append(block)
            elif part_type == "image" and part.get("data"):
                blocks.append(
                    _image_block(
                        str(part["data"]),
                        str(part.get("mimeType") or "image/png"),
                    )
                )
        if not blocks:
            blocks.append({"type": "text", "text": _multimodal_text_summary(result)})
        return blocks

    if isinstance(result, str):
        return [{"type": "text", "text": result}]
    return [{"type": "text", "text": _multimodal_text_summary(result)}]


def build_tool_response(tool_name: str, result: Any) -> dict[str, Any]:
    """Build the MCP tool response for a completed Hermes tool result."""
    is_error, _ = _detect_tool_failure(tool_name, result)
    response: dict[str, Any] = {"content": tool_result_content_blocks(result)}
    if is_error:
        # Python SDK spells this snake_case; the TypeScript SDK uses isError.
        response["is_error"] = True
    return response


def _resolve_task_id(effective_task_id: Any) -> str:
    if callable(effective_task_id):
        return str(effective_task_id() or "")
    return str(effective_task_id or "")


def run_bridged_tool(
    agent,
    tool_name: str,
    args: dict[str, Any],
    effective_task_id: Any,
    *,
    messages: Optional[list] = None,
) -> dict[str, Any]:
    """Execute one Claude tool call through the Hermes lifecycle.

    Synchronous by design: the whole executor stack (approvals, terminal
    backends, checkpoints) is blocking, so handlers run this on a worker
    thread rather than the SDK's event loop.
    """
    tool_call = SimpleNamespace(
        id=f"claude_{uuid.uuid4().hex[:24]}",
        function=SimpleNamespace(
            name=tool_name,
            arguments=json.dumps(args or {}, ensure_ascii=False),
        ),
    )
    outcome = execute_one_tool(
        agent,
        tool_call,
        _resolve_task_id(effective_task_id),
        messages=messages,
    )
    result = finalize_tool_outcome(agent, outcome)
    response = build_tool_response(outcome.function_name, result)
    if (outcome.is_error or outcome.blocked or outcome.cancelled) and not response.get(
        "is_error"
    ):
        response["is_error"] = True
    return response


def build_bridged_sdk_tools(
    agent,
    effective_task_id: Any,
    *,
    messages: Optional[list] = None,
) -> list:
    """Every Hermes tool for *agent*, wrapped as an SDK ``@tool``.

    Split out of :func:`build_hermes_sdk_mcp_server` so callers can inspect
    what the bridge composed without reaching into the value
    ``create_sdk_mcp_server`` returns, which is an SDK-internal shape that
    differs between the real package and a stub.
    """
    try:
        from claude_agent_sdk import ToolAnnotations, tool
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            "api_mode='claude_agent_sdk' requires the claude-agent-sdk package: "
            f"pip install claude-agent-sdk ({exc})"
        ) from exc

    handlers = []
    for spec in agent_tool_specs(agent):
        tool_name = spec["name"]

        def _make_handler(name: str):
            def _invoke(call_args: dict[str, Any]) -> dict[str, Any]:
                return run_bridged_tool(
                    agent, name, call_args, effective_task_id, messages=messages,
                )

            async def _handler(call_args: dict[str, Any]) -> dict[str, Any]:
                # propagate_context_to_thread carries the turn's ContextVars
                # and thread-local approval/sudo callbacks into the worker,
                # the same way the concurrent executor does.
                return await asyncio.to_thread(
                    propagate_context_to_thread(_invoke), call_args or {},
                )

            _handler.__name__ = name
            return _handler

        handlers.append(
            tool(
                tool_name,
                spec["description"],
                spec["parameters"],
                annotations=ToolAnnotations(readOnlyHint=is_read_only_tool(tool_name)),
            )(_make_handler(tool_name))
        )

    logger.info(
        "claude_agent_sdk bridge exposing %d Hermes tool(s) as %s*",
        len(handlers),
        MCP_TOOL_PREFIX,
    )
    return handlers


def build_hermes_sdk_mcp_server(
    agent,
    effective_task_id: Any,
    *,
    messages: Optional[list] = None,
    server_name: str = MCP_SERVER_NAME,
    version: str = MCP_SERVER_VERSION,
) -> Any:
    """Build an in-process ``claude_agent_sdk`` MCP server bound to *agent*.

    ``effective_task_id`` may be a string or a zero-argument callable, so a
    long-lived server can follow the turn's task id (and therefore the
    local / Docker / SSH / Modal / Daytona / Singularity environment the
    tools execute in).

    Raises ``ImportError`` when the optional ``claude-agent-sdk`` extra is
    not installed; the import is deliberately deferred to here so importing
    this module never requires it.
    """
    handlers = build_bridged_sdk_tools(
        agent, effective_task_id, messages=messages,
    )
    from claude_agent_sdk import create_sdk_mcp_server

    return create_sdk_mcp_server(name=server_name, version=version, tools=handlers)


def bridged_allowed_tools(agent) -> list[str]:
    """Every ``mcp__hermes__<tool>`` name the bridge exposes for *agent*.

    PR4 passes this straight into the SDK's ``allowed_tools``.
    """
    return [mcp_tool_name(spec["name"]) for spec in agent_tool_specs(agent)]


__all__ = [
    "MCP_SERVER_NAME",
    "MCP_SERVER_VERSION",
    "MCP_TOOL_PREFIX",
    "mcp_tool_name",
    "agent_tool_specs",
    "is_read_only_tool",
    "tool_result_content_blocks",
    "build_tool_response",
    "run_bridged_tool",
    "build_bridged_sdk_tools",
    "build_hermes_sdk_mcp_server",
    "bridged_allowed_tools",
]
