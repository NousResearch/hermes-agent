"""
Hermes external tool adapter for Atropos ToolServer.

These tools reuse Hermes-Agent's existing tool runner (`model_tools.handle_function_call`)
so we don't duplicate external tool implementations.

Important:
- These are marked `external=True` and should be executed ONLY by ToolServer.
- We run `handle_function_call` in a worker thread because the Hermes implementation
  uses `asyncio.run()` internally for some async tools (web_extract, vision, MoA, etc).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import model_tools

from .base import Tool, ToolResult, ToolSchema


def _schema_from_openai_tool_dict(tool: Dict[str, Any], *, external: bool) -> ToolSchema:
    fn = tool.get("function") or {}
    name = str(fn.get("name") or "")
    description = str(fn.get("description") or "")
    params = fn.get("parameters") or {}
    properties = params.get("properties") or {}
    required = params.get("required") or []
    if not isinstance(required, list):
        required = []
    return ToolSchema(
        name=name,
        description=description,
        parameters=dict(properties),
        required=[str(x) for x in required if isinstance(x, (str, int))],
        external=external,
    )


class HermesExternalTool(Tool):
    def __init__(self, schema: ToolSchema):
        self._schema = schema

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def execute(self, task_id: Optional[str] = None, **kwargs: Any) -> ToolResult:
        # `model_tools.handle_function_call` returns a JSON string (success or error).
        # Run in a thread because some Hermes tool handlers call `asyncio.run()`.
        raw = await asyncio.to_thread(model_tools.handle_function_call, self.name, kwargs, task_id)

        try:
            parsed = json.loads(raw)
        except Exception:
            # Keep as plain string.
            return ToolResult(success=True, output=str(raw))

        if isinstance(parsed, dict) and parsed.get("error"):
            return ToolResult(success=False, error=str(parsed.get("error")), output="")

        return ToolResult(success=True, output=json.dumps(parsed, ensure_ascii=False))


def build_external_tools(
    *,
    selected_tool_names: Optional[set[str]] = None,
) -> List[HermesExternalTool]:
    """
    Build external tool wrappers from Hermes tool declarations.

    Filters out sandbox-oriented tools (e.g. `terminal`) since those should run
    inside the sandbox via ToolExecutor.
    """
    # IMPORTANT: Hermes' `model_tools.get_tool_definitions()` only understands Hermes toolsets.
    # Atropos envs add extra toolsets (filesystem/sandbox/stateful). To avoid noisy "Unknown toolset"
    # prints and accidental filtering, we fetch ALL Hermes tool definitions here and filter by name.
    tools = model_tools.get_tool_definitions(enabled_toolsets=None, disabled_toolsets=None, quiet_mode=True)

    wrappers: List[HermesExternalTool] = []
    for t in tools:
        schema = _schema_from_openai_tool_dict(t, external=True)
        if schema.name in {"terminal"}:
            continue
        if selected_tool_names is not None and schema.name not in selected_tool_names:
            continue
        wrappers.append(HermesExternalTool(schema))
    return wrappers
