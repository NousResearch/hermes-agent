"""In-process Hermes MCP tools for the Claude Agent SDK runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable, Mapping, Set
from typing import Any


def _tool_specs(
    definitions: Iterable[Mapping[str, Any]],
    allowed_names: Set[str] | None,
) -> list[Mapping[str, Any]]:
    specs: list[Mapping[str, Any]] = []
    for definition in definitions:
        if definition.get("type") != "function":
            continue
        function = definition.get("function")
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name", "") or "")
        if allowed_names is None and not name.startswith("kanban_"):
            continue
        if allowed_names is not None and name not in allowed_names:
            continue
        specs.append(function)
    return sorted(specs, key=lambda spec: str(spec.get("name", "")))


def _tool_result(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return {"content": [{"type": "text", "text": text}]}


def build_hermes_sdk_mcp_server(
    tool_definitions: Iterable[Mapping[str, Any]],
    *,
    dispatch: Callable[..., Any],
    task_id: str | None,
    sdk: Any,
    allowed_names: Set[str] | None = None,
    argument_transform: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    handler_overrides: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> Any:
    """Build the SDK's in-process MCP server from Hermes' live schemas.

    The first slice deliberately exposes only ``kanban_*`` tools. They are
    stateless, worker-safe, and are the capability required for a Claude Max
    Kanban worker to report progress without starting a credential-bearing
    stdio child process.
    """

    tools: list[Any] = []
    for spec in _tool_specs(tool_definitions, allowed_names):
        name = str(spec["name"])
        description = str(spec.get("description") or f"Hermes {name} tool")
        schema = spec.get("parameters") or {"type": "object", "properties": {}}

        def _make_handler(tool_name: str):
            async def _handler(arguments: Mapping[str, Any]) -> dict[str, Any]:
                try:
                    dispatch_arguments = dict(arguments or {})
                    if argument_transform is not None:
                        dispatch_arguments = argument_transform(
                            tool_name, dispatch_arguments
                        )
                    override = (handler_overrides or {}).get(tool_name)
                    if override is not None:
                        value = await asyncio.to_thread(override, dispatch_arguments)
                    else:
                        value = await asyncio.to_thread(
                            dispatch,
                            tool_name,
                            dispatch_arguments,
                            task_id=task_id,
                        )
                    return _tool_result(value)
                except Exception as exc:
                    result = _tool_result(
                        {"error": str(exc), "tool": tool_name, "success": False}
                    )
                    result["is_error"] = True
                    return result

            return _handler

        tools.append(sdk.tool(name, description, schema)(_make_handler(name)))

    return sdk.create_sdk_mcp_server(name="hermes", version="1.0.0", tools=tools)


__all__ = ["build_hermes_sdk_mcp_server"]
