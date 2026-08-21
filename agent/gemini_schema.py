"""Helpers for translating OpenAI-style tool schemas to Gemini's schema subset."""

from __future__ import annotations

import math
from typing import Any, Dict

# Gemini's ``FunctionDeclaration.parameters`` field accepts the ``Schema``
# object, which is only a subset of OpenAPI 3.0 / JSON Schema.  Strip fields
# outside that subset before sending Hermes tool schemas to Google.
_GEMINI_SCHEMA_ALLOWED_KEYS = {
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "enum",
    "maxItems",
    "minItems",
    "properties",
    "required",
    "minProperties",
    "maxProperties",
    "minLength",
    "maxLength",
    "pattern",
    "example",
    "anyOf",
    "propertyOrdering",
    "default",
    "items",
    "minimum",
    "maximum",
}


def sanitize_gemini_schema(schema: Any) -> Dict[str, Any]:
    """Return a Gemini-compatible copy of a tool parameter schema.

    Hermes tool schemas are OpenAI-flavored JSON Schema and may contain keys
    such as ``$schema`` or ``additionalProperties`` that Google's Gemini
    ``Schema`` object rejects.  This helper preserves the documented Gemini
    subset and recursively sanitizes nested ``properties`` / ``items`` /
    ``anyOf`` definitions.
    """

    if not isinstance(schema, dict):
        return {}

    prefix_items = schema.get("prefixItems")
    cleaned: Dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _GEMINI_SCHEMA_ALLOWED_KEYS:
            continue
        if key == "properties":
            if not isinstance(value, dict):
                continue
            props: Dict[str, Any] = {}
            for prop_name, prop_schema in value.items():
                if not isinstance(prop_name, str):
                    continue
                props[prop_name] = sanitize_gemini_schema(prop_schema)
            cleaned[key] = props
            continue
        if key == "items":
            cleaned[key] = sanitize_gemini_schema(value)
            continue
        if key == "anyOf":
            if not isinstance(value, list):
                continue
            cleaned[key] = [
                sanitize_gemini_schema(item)
                for item in value
                if isinstance(item, dict)
            ]
            continue
        cleaned[key] = value

    # Gemini's Schema validator requires every ``enum`` entry to be a string,
    # even when the parent ``type`` is ``integer`` / ``number`` / ``boolean``.
    # Preserve those constraints by stringifying scalar values while keeping
    # the declared type intact; Gemini uses the strings as schema metadata and
    # still emits typed tool arguments at runtime.
    enum_val = cleaned.get("enum")
    type_val = cleaned.get("type")
    if isinstance(enum_val, list) and type_val in {"integer", "number", "boolean"}:
        stringified = []
        for item in enum_val:
            if isinstance(item, str):
                value = item
            elif isinstance(item, bool):
                value = "true" if item else "false"
            elif (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(item)
            ):
                value = str(item)
            else:
                continue
            if value not in stringified:
                stringified.append(value)
        if stringified:
            cleaned["enum"] = stringified
        else:
            cleaned.pop("enum", None)

    # Gemini validates ``required`` strictly against the same node's
    # ``properties`` — GenerateContentRequest fails with HTTP 400
    # "...items.required[0]: property is not defined" when a required name
    # has no matching property in that node.  MCP servers routinely emit
    # this shape (e.g. the GitHub remote MCP's array item schemas carry
    # ``required`` without ``properties``), and one bad tool schema fails
    # the ENTIRE request before any model output.  Filter ``required`` to
    # names that exist in this node's ``properties`` and drop it when
    # nothing valid remains.  The tool handler still validates required
    # fields at execution time, so this only removes what Gemini couldn't
    # accept anyway.  (Port of Kilo-Org/kilocode#11955.)
    required_val = cleaned.get("required")
    if isinstance(required_val, list):
        props_val = cleaned.get("properties")
        prop_names = set(props_val.keys()) if isinstance(props_val, dict) else set()
        valid_required = [
            name for name in required_val
            if isinstance(name, str) and name in prop_names
        ]
        if not valid_required:
            cleaned.pop("required", None)
        elif len(valid_required) != len(required_val):
            cleaned["required"] = valid_required

    # Gemini's Schema validator requires an ``items`` field on every array
    # node — a bare ``{"type": "array"}`` (valid standard JSON Schema, where
    # element types are simply unconstrained) fails the ENTIRE
    # GenerateContentRequest with HTTP 400
    # ``...parameters.properties[<name>].items: missing field`` before any model
    # output. MCP / plugin / dynamic tool schemas routinely omit ``items``, so
    # populate a schema when it is absent. For a bounded homogeneous
    # ``prefixItems`` tuple, collapse the repeated positional schema into
    # ``items`` without inventing length bounds. Otherwise use an empty schema,
    # which preserves the unconstrained-element meaning of a genuinely bare
    # array. A declared ``items`` is left untouched. The legacy
    # ``FunctionDeclaration.parameters`` allow-list does not pass
    # ``prefixItems`` through; the newer JSON Schema path is separate.
    if cleaned.get("type") == "array" and "items" not in cleaned:
        cleaned_prefix = []
        if (
            isinstance(prefix_items, list)
            and prefix_items
            and all(isinstance(item, dict) for item in prefix_items)
        ):
            cleaned_prefix = [
                sanitize_gemini_schema(item) for item in prefix_items
            ]

        max_items = cleaned.get("maxItems")
        bounded_to_prefix = (
            isinstance(max_items, int)
            and not isinstance(max_items, bool)
            and 0 <= max_items <= len(cleaned_prefix)
        )
        if (
            cleaned_prefix
            and bounded_to_prefix
            and all(item == cleaned_prefix[0] for item in cleaned_prefix[1:])
        ):
            cleaned["items"] = cleaned_prefix[0]
        else:
            cleaned["items"] = {}

    return cleaned


def sanitize_gemini_tool_parameters(parameters: Any) -> Dict[str, Any]:
    """Normalize tool parameters to a valid Gemini object schema."""

    cleaned = sanitize_gemini_schema(parameters)
    if not cleaned:
        return {"type": "object", "properties": {}}
    return cleaned
