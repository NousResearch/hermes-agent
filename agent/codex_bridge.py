"""Bridge Hermes context and stateful tools into Codex app-server threads.

This module is used only by the opt-in ``codex_app_server`` runtime. Codex keeps
its native coding loop and tools; Hermes contributes its assembled identity,
recalled memory, resumed transcript, and a small allowlist of live stateful tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable, Optional

from agent.display import _detect_tool_failure
from agent.memory_manager import build_memory_context_block

logger = logging.getLogger(__name__)

HERMES_DYNAMIC_TOOL_NAMESPACE = "hermes"
HERMES_STATEFUL_TOOL_ALLOWLIST = frozenset(
    {"memory", "session_search", "todo", "delegate_task", "skill_manage"}
)


def build_dynamic_tools(
    tool_definitions: Iterable[dict[str, Any]],
    active_tool_names: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """Convert active Hermes stateful tools to Codex ``dynamicTools`` specs.

    Native Codex and MCP tools remain untouched. A namespace avoids collisions
    with Codex's reserved runtime namespaces and makes the ownership boundary
    explicit on the wire.
    """
    active = set(active_tool_names) if active_tool_names is not None else None
    functions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for definition in tool_definitions or []:
        if not isinstance(definition, dict) or definition.get("type") != "function":
            continue
        function = definition.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if (
            name not in HERMES_STATEFUL_TOOL_ALLOWLIST
            or name in seen
            or (active is not None and name not in active)
        ):
            continue
        seen.add(name)
        schema = function.get("parameters")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        functions.append(
            {
                "type": "function",
                "name": name,
                "description": str(function.get("description") or "")[:1024],
                "deferLoading": False,
                "inputSchema": schema,
            }
        )
    if not functions:
        return []
    return [
        {
            "type": "namespace",
            "name": HERMES_DYNAMIC_TOOL_NAMESPACE,
            "description": (
                "Hermes session, memory, task-state, delegation, and skill-lifecycle tools. "
                "Use these when work depends on Hermes state rather than the local filesystem."
            ),
            "tools": functions,
        }
    ]


def build_history_items(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a resumed Hermes transcript to Codex message items.

    Only user/assistant text is restored. Tool traces are intentionally omitted:
    they can be large, may contain stale ephemeral handles, and the assistant
    answers retain the useful outcome. The current user message must be excluded
    by the caller because it is sent separately through ``turn/start``.
    """
    items: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        content_type = "input_text" if role == "user" else "output_text"
        items.append(
            {
                "type": "message",
                "role": role,
                "content": [{"type": content_type, "text": content}],
            }
        )
    return items


def build_turn_input(
    user_message: Any,
    *,
    external_memory_context: str = "",
    plugin_user_context: str = "",
) -> Any:
    """Attach per-turn ephemeral Hermes context without mutating persistence."""
    injections: list[str] = []
    if isinstance(external_memory_context, str) and external_memory_context.strip():
        fenced = build_memory_context_block(external_memory_context)
        if fenced:
            injections.append(fenced)
    if isinstance(plugin_user_context, str) and plugin_user_context.strip():
        injections.append(plugin_user_context)
    if not injections:
        return user_message

    suffix = "\n\n".join(injections)
    if isinstance(user_message, str):
        return f"{user_message}\n\n{suffix}"
    # Rich input is uncommon on this runtime and is collapsed to text by the
    # session adapter. Preserve the original object while adding one text item.
    if isinstance(user_message, list):
        return list(user_message) + [{"type": "text", "text": suffix}]
    return f"{user_message}\n\n{suffix}"


def make_dynamic_tool_handler(
    agent: Any,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a live Codex dynamic-tool dispatcher backed by Hermes execution.

    The callback checks the allowlist and the agent's current registry on every
    call, then uses ``AIAgent._invoke_tool`` so Hermes middleware, plugin hooks,
    stateful managers, delegation semantics, and enabled-toolset policy remain
    authoritative.
    """

    def handle(params: dict[str, Any]) -> dict[str, Any]:
        namespace = params.get("namespace")
        tool_name = params.get("tool")
        call_id = params.get("callId")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(call_id, str) or not call_id.strip():
            return _tool_response(False, "Dynamic tool callId must be a non-empty string")
        if namespace != HERMES_DYNAMIC_TOOL_NAMESPACE:
            return _tool_response(False, f"Unsupported dynamic tool namespace: {namespace!r}")
        if tool_name not in HERMES_STATEFUL_TOOL_ALLOWLIST:
            return _tool_response(False, f"Hermes dynamic tool is not allowlisted: {tool_name!r}")
        if tool_name not in set(getattr(agent, "valid_tool_names", set()) or set()):
            return _tool_response(False, f"Hermes tool is not active in this session: {tool_name}")
        if not isinstance(arguments, dict):
            return _tool_response(False, "Dynamic tool arguments must be an object")

        try:
            guardrails = getattr(agent, "_tool_guardrails", None)
            if guardrails is not None:
                decision = guardrails.before_call(tool_name, arguments)
                if not decision.allows_execution:
                    blocked = agent._guardrail_block_result(decision)
                    return _tool_response(False, blocked)

            if tool_name == "memory":
                agent._turns_since_memory = 0

            resolved_task_id = getattr(agent, "_current_task_id", None)
            result = agent._invoke_tool(
                tool_name,
                arguments,
                resolved_task_id,
                tool_call_id=call_id,
                messages=getattr(agent, "messages", None),
            )
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)
            failed, _ = _detect_tool_failure(tool_name, result)
            append_observation = getattr(agent, "_append_guardrail_observation", None)
            if callable(append_observation):
                result = append_observation(
                    tool_name,
                    arguments,
                    result,
                    failed=failed,
                )
            return _tool_response(not failed, result)
        except Exception:
            logger.exception("Hermes dynamic tool %s failed", tool_name)
            return _tool_response(False, f"Hermes tool {tool_name!r} failed")

    return handle


def _tool_response(success: bool, text: Any) -> dict[str, Any]:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, default=str)
    return {
        "success": bool(success),
        "contentItems": [{"type": "inputText", "text": text}],
    }
