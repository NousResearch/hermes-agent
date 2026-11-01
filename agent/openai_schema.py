"""Restore JSON-Schema nullability in tool schemas sent to OpenAI models.

``tools.schema_sanitizer`` collapses ``type: [X, "null"]`` to ``type: X`` plus the OpenAPI
``nullable: true`` hint for every provider (llama.cpp and Gemini reject type arrays). OpenAI
reads JSON Schema, not ``nullable``: an optional field then looks like one that needs a value,
and gpt-6-luna / gpt-6-sol / gpt-6.1-sol fill every one of them (``kanban_create`` received
``max_runtime_seconds: 0``, which the dispatcher kills on its first sweep, #38366). Turning the
hint back into the array form lets the model answer ``null``, which handlers read as omitted.
"""

from __future__ import annotations

from typing import Any


def sanitize_openai_tool_parameters(parameters: Any) -> Any:
    """Copy of *parameters* with every ``nullable: true`` node in JSON-Schema form; input is not mutated."""
    from tools.tool_search_validation import _schema_for_local_validation

    return _schema_for_local_validation(parameters)


def sanitize_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply ``sanitize_openai_tool_parameters`` to every tool's parameters.

    Returns the input list object itself when nothing needed repairing.
    """
    if not tools:
        return tools
    sanitized: list[dict[str, Any]] = []
    any_change = False
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict):
            params = fn.get("parameters")
            repaired = sanitize_openai_tool_parameters(params)
            if repaired != params:
                any_change = True
                tool = {**tool, "function": {**fn, "parameters": repaired}}
        sanitized.append(tool)
    return sanitized if any_change else tools


def is_openai_model(model: str | None) -> bool:
    """True for OpenAI flagship slugs (gpt-*, o1/o3/o4), regardless of aggregator prefix.

    ``gpt-oss`` is excluded: its open weights are commonly served by llama.cpp, whose grammar
    converter rejects the type arrays this module restores.
    """
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare.startswith(("gpt-", "o1", "o3", "o4")) and not bare.startswith("gpt-oss")
