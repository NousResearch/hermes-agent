"""Helpers for translating OpenAI-style tool schemas to Gemini's schema subset."""

from __future__ import annotations

import math
from typing import Any, Dict

# Gemini's ``FunctionDeclaration.parameters`` accepts only a subset of OpenAPI 3.0 /
# JSON Schema (the ``Schema`` object); everything else is stripped.
_GEMINI_SCHEMA_ALLOWED_KEYS = {
    "type", "format", "title", "description", "nullable", "enum", "maxItems", "minItems", "properties", "required",
    "minProperties", "maxProperties", "minLength", "maxLength", "pattern", "example", "anyOf", "propertyOrdering",
    "default", "items", "minimum", "maximum",
}


def _stringify_enum_value(item: Any) -> Any:
    """Gemini-safe string for a scalar enum entry, or None to drop it."""
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, (int, float)) and math.isfinite(item):
        return str(item)
    return item if isinstance(item, str) else None


def sanitize_gemini_schema(schema: Any) -> Dict[str, Any]:
    """Gemini-compatible copy of a tool parameter schema: keeps only the documented subset
    (drops e.g. ``$schema`` / ``additionalProperties``) and recursively sanitizes nested
    ``properties`` / ``items`` / ``anyOf``."""
    if not isinstance(schema, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _GEMINI_SCHEMA_ALLOWED_KEYS:
            continue
        if key == "properties":
            if isinstance(value, dict):
                cleaned[key] = {name: sanitize_gemini_schema(sub) for name, sub in value.items() if isinstance(name, str)}
        elif key == "items":
            cleaned[key] = sanitize_gemini_schema(value)
        elif key == "anyOf":
            if isinstance(value, list):
                cleaned[key] = [sanitize_gemini_schema(item) for item in value if isinstance(item, dict)]
        else:
            cleaned[key] = value

    # Gemini requires every ``enum`` entry to be a string even for
    # integer/number/boolean types; the declared type stays intact and Gemini
    # still emits typed tool arguments at runtime. dict.fromkeys = ordered dedupe.
    enum_val = cleaned.get("enum")
    if isinstance(enum_val, list) and cleaned.get("type") in {"integer", "number", "boolean"}:
        if stringified := list(dict.fromkeys(v for v in map(_stringify_enum_value, enum_val) if v is not None)):
            cleaned["enum"] = stringified
        else:
            cleaned.pop("enum", None)

    # Gemini validates ``required`` strictly against the same node's ``properties`` (HTTP 400
    # "property is not defined") and one bad tool schema fails the ENTIRE request. MCP servers
    # routinely emit ``required`` without ``properties``, so keep only names that exist here;
    # the tool handler still validates required fields at execution time.
    required_val = cleaned.get("required")
    if isinstance(required_val, list):
        props_val = cleaned.get("properties")
        prop_names = set(props_val) if isinstance(props_val, dict) else set()
        valid_required = [name for name in required_val if isinstance(name, str) and name in prop_names]
        if not valid_required:
            cleaned.pop("required", None)
        elif len(valid_required) != len(required_val):
            cleaned["required"] = valid_required
    return cleaned


def sanitize_gemini_tool_parameters(parameters: Any) -> Dict[str, Any]:
    """Normalize tool parameters to a valid Gemini object schema."""
    return sanitize_gemini_schema(parameters) or {"type": "object", "properties": {}}
